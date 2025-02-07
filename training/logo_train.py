import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import torch
import logging
import torch.nn.functional as F
from modelzipper.tutils import *
from typing import List, Tuple, Union, Literal, Dict
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from transformers import (AutoTokenizer, HfArgumentParser, set_seed, TrainingArguments, AutoConfig)
from simpo_trainer import SimPOTrainer
from training.custom_dataset import SimPODataCollator
from utils.utils import create_and_prepare_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class SimORPOTrainer(SimPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def odds_ratio_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute ORPO's odds ratio (OR) loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the ORPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The log odds ratio of the chosen responses over the rejected responses ratio for logging purposes.
            The `log(sigmoid(log_odds_chosen))` for logging purposes.
        """

        # Derived from Eqs. (4) and (7) from https://huggingface.co/papers/2403.07691 by using log identities and exp(log(P(y|x)) = P(y|x)
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
        )
        sig_ratio = F.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
        losses = self.beta * ratio

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()

        return losses, chosen_rewards, rejected_rewards, torch.mean(ratio).item(), torch.mean(log_odds).item()


    def simpo_loss(self, policy_chosen_logps, policy_rejected_logps):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - self.gamma_beta_ratio
        
        if self.loss_type == "sigmoid": 
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )
            
        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        return losses, chosen_rewards, rejected_rewards
    

    def concatenated_forward(self, model, batch): # one chosen, two rejected
        len_chosen = batch['chosen_input_ids'].size(0)
        concatenated_batch = self.move_to_device(batch, device=self.accelerator.device)

        input_ids = torch.concatenate([
            concatenated_batch["chosen_input_ids"], 
            concatenated_batch["reject_1_input_ids"], 
            concatenated_batch["reject_2_input_ids"]
        ], dim=0)

        attention_mask = torch.concatenate([
            concatenated_batch["chosen_attention_mask"], 
            concatenated_batch["reject_1_attention_mask"], 
            concatenated_batch["reject_2_attention_mask"]
        ], dim=0)

        position_ids = torch.concatenate([
            concatenated_batch["chosen_position_ids"], 
            concatenated_batch["reject_1_position_ids"], 
            concatenated_batch["reject_2_position_ids"]
        ], dim=0)

        labels = torch.cat([
            concatenated_batch["chosen_labels"], 
            concatenated_batch["reject_1_labels"], 
            concatenated_batch["reject_2_labels"]
        ], dim=0)

        selected_pos_01 = torch.where(labels[0] != labels[1])[0]
        selected_pos_12 = torch.where(labels[1] != labels[2])[0]
        selected_pos_02 = torch.where(labels[0] != labels[2])[0]
        selected_pos = torch.unique(torch.cat([selected_pos_01, selected_pos_12, selected_pos_02]))
        
        all_logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False, return_dict=True).logits

        all_logps = self.get_batch_logps(
            all_logits[:, -self.max_target_length:, :],
            labels[:, -self.max_target_length:],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        
        chosen_logps = all_logps[:len_chosen]
        prefix_chosen_logps = all_logps[len_chosen:len_chosen*2]
        suffix_chosen_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        prefix_chosen_logits = all_logits[len_chosen:len_chosen*2]
        suffix_chosen_logits = all_logits[len_chosen*2:]

        return (chosen_logps, 
                prefix_chosen_logps, 
                suffix_chosen_logps, 
                chosen_logits, 
                prefix_chosen_logits, 
                suffix_chosen_logits, )

    def get_batch_loss_metrics(self, model, batch, train_eval):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        
        policy_chosen_logps, policy_rejected_logps1, policy_rejected_logps2, policy_chosen_logits, policy_rejected_logits1, policy_rejected_logits2 = self.concatenated_forward(model, batch)
        
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(policy_chosen_logps, (policy_rejected_logps1 + policy_rejected_logps2) / 2)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        loss = losses.mean()

        if self.sft_weight > 0.0:
            loss_func = nn.CrossEntropyLoss() # method 2
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), batch["chosen_labels"].view(-1))
            loss = self.sft_weight * sft_loss + loss
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu()
    
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected1"] = policy_rejected_logps1.detach().mean().cpu()
        metrics[f"{prefix}logps/rejected2"] = policy_rejected_logps2.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected1"] = policy_rejected_logits1.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected2"] = policy_rejected_logits2.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        return loss, metrics


class LOGOTrainer(SimPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"beta value is: {self.beta}")
        
    def simpo_loss(self, policy_chosen_logps, policy_rejected_logps):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - self.gamma_beta_ratio
        
        if self.loss_type == "sigmoid": 
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )
            
        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        return losses, chosen_rewards, rejected_rewards
    

    def concatenated_forward(self, model, batch): # one chosen, two rejected
        len_chosen = batch['chosen_input_ids'].size(0)
        concatenated_batch = self.move_to_device(batch, device=self.accelerator.device)

        input_ids = torch.concatenate([
            concatenated_batch["chosen_input_ids"], 
            concatenated_batch["reject_1_input_ids"], 
            concatenated_batch["reject_2_input_ids"]
        ], dim=0)

        attention_mask = torch.concatenate([
            concatenated_batch["chosen_attention_mask"], 
            concatenated_batch["reject_1_attention_mask"], 
            concatenated_batch["reject_2_attention_mask"]
        ], dim=0)

        position_ids = torch.concatenate([
            concatenated_batch["chosen_position_ids"], 
            concatenated_batch["reject_1_position_ids"], 
            concatenated_batch["reject_2_position_ids"]
        ], dim=0)

        labels = torch.cat([
            concatenated_batch["chosen_labels"], 
            concatenated_batch["reject_1_labels"], 
            concatenated_batch["reject_2_labels"]
        ], dim=0)

        selected_pos_01 = torch.where(labels[0] != labels[1])[0]
        selected_pos_12 = torch.where(labels[1] != labels[2])[0]
        selected_pos_02 = torch.where(labels[0] != labels[2])[0]
        selected_pos = torch.unique(torch.cat([selected_pos_01, selected_pos_12, selected_pos_02]))
        if selected_pos.size(0) == 0:
            selected_pos = torch.arange(labels[0].size(0) - self.max_target_length, labels[0].size(0), device=selected_pos.device)
        min_b, max_b = selected_pos.min(), selected_pos.max()
        selected_pos = torch.arange(min_b, max_b+1, device=selected_pos.device)
        del selected_pos_01, selected_pos_12, selected_pos_02
        
        all_logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False, return_dict=True).logits

        all_logps = self.get_batch_logps(
            all_logits[:, selected_pos, :],
            labels[:, selected_pos],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        
        chosen_logps = all_logps[:len_chosen]
        prefix_chosen_logps = all_logps[len_chosen:len_chosen*2]
        suffix_chosen_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        prefix_chosen_logits = all_logits[len_chosen:len_chosen*2]
        suffix_chosen_logits = all_logits[len_chosen*2:]

        return (chosen_logps, 
                prefix_chosen_logps, 
                suffix_chosen_logps, 
                chosen_logits, 
                prefix_chosen_logits, 
                suffix_chosen_logits, )

    def get_batch_loss_metrics(self, model, batch, train_eval):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        
        policy_chosen_logps, policy_rejected_logps1, policy_rejected_logps2, policy_chosen_logits, policy_rejected_logits1, policy_rejected_logits2 = self.concatenated_forward(model, batch)
        
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(policy_chosen_logps, (policy_rejected_logps1 + policy_rejected_logps2) / 2)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        loss = losses.mean()

        if self.sft_weight > 0.0:
            loss_func = nn.CrossEntropyLoss() # method 2
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), batch["chosen_labels"].view(-1))
            loss = self.sft_weight * sft_loss + loss
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu()
    
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected1"] = policy_rejected_logps1.detach().mean().cpu()
        metrics[f"{prefix}logps/rejected2"] = policy_rejected_logps2.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected1"] = policy_rejected_logits1.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected2"] = policy_rejected_logits2.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        return loss, metrics


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")
    rope_type: Optional[str] = field(default=None)
    factor: int = field(default=10, metadata={"help": "RoPE factor"})
    max_position_embeddings: int = field(default=10, metadata={"help": "max_position_embeddings"})
    rope_theta: float = field(default=None, metadata={"help": "rope_theta"})
    peft_model_path: Optional[str] = field(default=None)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=16)
    

@dataclass
class TrainingArguments(TrainingArguments):
    '''
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    max_target_length: Optional[int] = None
    '''
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataset_path: str = field(default="alpaca")
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed, norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    beta: float = field(default=2.0)
    gamma_beta_ratio: float = field(default=0.25)
    sft_weight: float = field(default=0.0)
    label_smoothing: float = field(default=0)
    loss_type: Literal["sigmoid", "hinge"] = field(default="sigmoid")
    disable_dropout: bool = field(default=True)
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    precompute_ref_log_probs: bool = False
    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None
    max_target_length: int = 512


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    logger.info('load model and tokenizer ...')
    model, tokenizer = create_and_prepare_model(model_args.model_name_or_path, training_args, model_args)
    model.config.use_cache=not training_args.gradient_checkpointing  # required for gradient checkpointing

    logger.info('load datasets ...')
    dataset = load_from_disk(training_args.dataset_path)
    
    logger.info('initialize dpo training arguments ...')
    dpo_trainer = LOGOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],  # just for debug
        tokenizer=tokenizer,
        data_collator=SimPODataCollator(
            max_seq_length=training_args.max_seq_length, tokenizer=tokenizer
        ),
        load_from_cache_path=True,
        cache_path=None,
    )

    logger.info('start training ...')
    dpo_trainer.train()
    
    logger.info('end training ...')
    dpo_trainer.save_model(training_args.output_dir)
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
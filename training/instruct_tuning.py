import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import transformers
import datasets
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from modelzipper.tutils import *
from transformers import Trainer
from utils.utils import create_and_prepare_model
from training.custom_dataset import InstructTuningDataCollator

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
class TrainingArguments(transformers.TrainingArguments):
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
    loss_full_seq_length: bool = field(
        default=True,
        metadata={"help": "Whether calculate full sequence length"},
    )


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # load model and tokenizer
    model, tokenizer = create_and_prepare_model(model_args.model_name_or_path, training_args, model_args)
    model.config.use_cache=not training_args.gradient_checkpointing  # required for gradient checkpointing
    # model.gradient_checkpointing_enable()  # enable gradient checkpointing

    # load train_dataset and data_collator
    eval_dataset = None
    if os.path.isdir(training_args.dataset_path):
        train_dataset = datasets.load_dataset(training_args.dataset_path)['train']
        if training_args.eval_strategy != 'no':
            eval_dataset = datasets.load_dataset(training_args.dataset_path)['validation']
    else:
        train_dataset = auto_read_data(training_args.dataset_path)
        if training_args.eval_strategy != 'no':
            eval_dataset = auto_read_data(training_args.dataset_path)
    
    data_collator = InstructTuningDataCollator(training_args.max_seq_length, tokenizer, training_args.loss_full_seq_length)
    
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
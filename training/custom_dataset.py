import torch
import os
import json, dill
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
# ================== Custom DataCollator ==================

class BasicDataCollator:
    def __init__(self, max_seq_length: int, tokenizer=None, **kwargs):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def auto_cut_length(self, t, is_label=False):
        if t.size(-1) > self.max_seq_length:
            # cut the seq length from the middle
            t = torch.cat([t[:self.max_seq_length // 2], t[-self.max_seq_length // 2:]], dim=0)
        elif t.size(-1) < self.max_seq_length:
            padding_length = self.max_seq_length - t.size(-1)
            if is_label:
                t = F.pad(t, (0, padding_length), 'constant', -100)
            else:
                t = F.pad(t, (0, padding_length), 'constant', 0)
        return t

    def __call__(self, batch):
        wrap_batch = {}
        batch_size = len(batch)
        keys = list(batch[0].keys())
        if isinstance(batch[0][keys[0]], list):
            for key in keys:
                wrap_batch[key] = torch.stack([self.auto_cut_length(torch.tensor(batch[i][key])) for i in range(batch_size)])
        else:
            for key in keys:
                wrap_batch[key] = torch.stack([self.auto_cut_length(batch[i][key]) for i in range(batch_size)])
        return wrap_batch


class LanguageModelingDataCollator(BasicDataCollator):
    def __call__(self, batch):
        wrap_batch = {}
        batch_size = len(batch)
        
        # wrap the bath with tensor
        if isinstance(batch[0]['input_ids'], list):
            wrap_batch['input_ids'] = torch.tensor([batch[i]['input_ids'] for i in range(batch_size)])
            wrap_batch['attention_mask'] = torch.tensor([batch[i]['attention_mask'] for i in range(batch_size)])
            wrap_batch['labels'] = torch.tensor([batch[i]['labels'] for i in range(batch_size)])
        else:
            wrap_batch['input_ids'] = torch.stack([batch[i]['input_ids'] for i in range(batch_size)])
            wrap_batch['attention_mask'] = torch.stack([batch[i]['attention_mask'] for i in range(batch_size)])
            wrap_batch['labels'] = torch.stack([batch[i]['labels'] for i in range(batch_size)])
        
        # trunct the max sequence length
        if wrap_batch['input_ids'].ndim == 1:
            wrap_batch['input_ids'] = wrap_batch['input_ids'][None, :]
            wrap_batch['attention_mask'] = wrap_batch['attention_mask'][None, :]
            wrap_batch['labels'] = wrap_batch['labels'][None, :]
        
        # make sure the data has BxL shape
        if wrap_batch['input_ids'].size(1) > self.max_seq_length:
            wrap_batch['input_ids'] = wrap_batch['input_ids'][:, :self.max_seq_length]
            wrap_batch['attention_mask'] = wrap_batch['attention_mask'][:, :self.max_seq_length]
            wrap_batch['labels'] = wrap_batch['labels'][:, :self.max_seq_length]
        elif wrap_batch['input_ids'].size(1) < self.max_seq_length:
            padding_length = self.max_seq_length - wrap_batch['input_ids'].size(1)
            wrap_batch['input_ids'] = F.pad(wrap_batch['input_ids'], (0, padding_length), 'constant', 0)
            wrap_batch['attention_mask'] = F.pad(wrap_batch['attention_mask'], (0, padding_length), 'constant', 0)
            wrap_batch['labels'] = F.pad(wrap_batch['labels'], (0, padding_length), 'constant', -100)

        return wrap_batch


class InstructTuningDataCollator(BasicDataCollator):
    def __init__(self, max_seq_length: int, tokenizer=None, cal_full_seq_loss=False, **kwargs):
        super().__init__(max_seq_length, tokenizer, **kwargs)
        self.cal_full_seq_loss = cal_full_seq_loss

    def __call__(self, batch):
        instructions = [item['instruction'] for item in batch]
        outputs = [item['output'] for item in batch]
        instruct_toks = [self.tokenizer(item, return_tensors="pt").input_ids for item in instructions]
        output_toks = [self.tokenizer(item, return_tensors="pt").input_ids for item in outputs]
        input_ids_list, attention_masks_list, labels_list = [], [], []

        for inst, out in zip(instruct_toks, output_toks):
            if inst.size(-1) + out.size(-1) > self.max_seq_length:
                remain_seq_length = self.max_seq_length - out.size(-1)
                inst = torch.cat([inst[:, :remain_seq_length // 2], inst[:, -remain_seq_length // 2:]], dim=1)

            full_input_ids = torch.cat([inst, out], dim=1)
            attention_mask = torch.ones_like(full_input_ids, dtype=torch.long)
            if self.cal_full_seq_loss:
                labels = full_input_ids
            else:
                # Constructing labels, only compute loss on output tokens
                labels = torch.full(inst.size(), -100, dtype=torch.long)
                labels = torch.cat([labels, out], dim=1)

            if full_input_ids.size(-1) < self.max_seq_length:
                padding_length = self.max_seq_length - full_input_ids.size(-1)
                labels = F.pad(labels, (0, padding_length), 'constant', -100)
                full_input_ids = F.pad(full_input_ids, (0, padding_length), 'constant', 0)
                attention_mask = F.pad(attention_mask, (0, padding_length), 'constant', 0)

            input_ids_list.append(full_input_ids.squeeze(0))
            attention_masks_list.append(attention_mask.squeeze(0))
            labels_list.append(labels.squeeze(0))

        batch_data = {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_masks_list),
            'labels': torch.stack(labels_list)
        }
        return batch_data


class SimPODataCollator(BasicDataCollator):
    def filter_fun(self, x):
        return True

    def __call__(self, batch):
        wrap_batch = {}
        batch_size = len(batch)
        keys = list(batch[0].keys())
        if isinstance(batch[0][keys[0]], list):
            for key in keys:
                if self.filter_fun(key):
                    wrap_batch[key] = torch.stack([self.auto_cut_length(torch.tensor(batch[i][key])) for i in range(batch_size)])
        else:
            for key in keys:
                if self.filter_fun(key):
                    wrap_batch[key] = torch.stack([self.auto_cut_length(batch[i][key]) for i in range(batch_size)])
        return wrap_batch


class MOSimPODataCollator(BasicDataCollator):
    def filter_fun(self, x):
        return True  # not filtering

    def __call__(self, batch):
        wrap_batch = {}
        batch_size = len(batch)
        keys = list(batch[0].keys())
        if isinstance(batch[0][keys[0]], list):
            for key in keys:
                if self.filter_fun(key):
                    wrap_batch[key] = torch.stack([self.auto_cut_length(torch.tensor(batch[i][key])) for i in range(batch_size)])
        else:
            for key in keys:
                if self.filter_fun(key):
                    wrap_batch[key] = torch.stack([self.auto_cut_length(batch[i][key]) for i in range(batch_size)])
        
        return wrap_batch

class PoseDataCollator(BasicDataCollator):
    
    def __call__(self, batch):
        wrap_batch = {}
        batch_size = len(batch)
        keys = list(batch[0].keys())
        if isinstance(batch[0][keys[0]], list):
            for key in keys:
                if key.startswith('chosen'):
                    wrap_batch[key.split('chosen_')[-1]] = torch.stack(
                        [self.auto_cut_length(torch.tensor(batch[i][key]), is_label=('labels' in key)) for i in range(batch_size)]
                    )
        else:
            for key in keys:
                if key.startswith('chosen'):
                    wrap_batch[key.split('chosen_')[-1]] = torch.stack(
                        [self.auto_cut_length(torch.tensor(batch[i][key]), is_label=('labels' in key)) for i in range(batch_size)]
                    )
        return wrap_batch




class LMDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.input_ids, self.labels = self.process_data(filepath)

    def process_data(self, filepath):
        input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs.npy')))
        labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels.npy')))
        return input_ids, labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return self.input_ids.size(0)

class LMSortDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.input_ids, self.labels = self.process_data(filepath)
    
    def process_data(self, filepath):
        input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs_sort.npy')))
        labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels_sort.npy')))
        return input_ids, labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return self.input_ids.size(0)

class LMPackDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.input_ids, self.attention_masks, self.labels, self.weights, self.nums = self.process_data(filepath)
        self.num_gpus = torch.cuda.device_count()
        
    def process_data(self, filepath):
        input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs_pack.npy')))
        labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels_pack.npy')))
        weights = torch.from_numpy(np.load(os.path.join(filepath, 'weights_pack.npy')))
        attention_masks = json.load(open(os.path.join(filepath, 'attention_masks_pack.json')))
        num_gpus = torch.cuda.device_count()
        l = (input_ids.size(0) // num_gpus) * num_gpus
        input_ids, labels, weights, attention_masks = input_ids[:l, :], labels[:l, :], weights[:l, :], attention_masks[:l]
        nums = [weights[i*num_gpus:(i+1)*num_gpus, :].sum() for i in range(l//num_gpus)]
        return input_ids, attention_masks, labels, weights, nums

    def __getitem__(self, idx):
        if idx < 32: # reduce GPU memory usage during first few steps
            max_length_tmp = 32768
            attention_mask_tmp = []
            for pos in self.attention_masks[idx]:
                if pos < max_length_tmp:
                    attention_mask_tmp.append(pos)
            attention_mask_tmp.append(max_length_tmp)
            return {
                'input_ids': self.input_ids[idx, :max_length_tmp],
                'attention_mask': torch.tensor(attention_mask_tmp, dtype=torch.int32),
                'labels': (self.labels[idx, :max_length_tmp], self.weights[idx, :max_length_tmp]*2, self.nums[idx//self.num_gpus])
            }
        else:
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.int32),
                'labels': (self.labels[idx], self.weights[idx], self.nums[idx//self.num_gpus])
            }

    def __len__(self):
        return self.input_ids.size(0)
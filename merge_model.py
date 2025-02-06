import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="3,4"

model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

adapter_path = "/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt/baseline/Llama-3.1-8B-Instruct/longalpaca/adapter/global_step250" 

model_config = os.path.join(adapter_path,"config.json")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained("/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/Llama-3.1-8B-Instruct-longalpaca-adapter-global_step250/")
# model_config = os.path.join(adapter_path,"config.json")

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, 
use_flash_attention_2="flash_attention_2", 
device_map="auto", config=AutoConfig.from_pretrained(model_config) if (model_config is not None and len(model_config)>0) else None )

if adapter_path:
    model = PeftModelForCausalLM.from_pretrained(model, adapter_path)#.eval()
    print("model_config:", model_config)
    
merged_model = model.merge_and_unload()

os.makedirs("/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/Llama-3.1-8B-Instruct-longalpaca-adapter-global_step250/",exist_ok = True)

merged_model.save_pretrained(f"/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/Llama-3.1-8B-Instruct-longalpaca-adapter-global_step250/")

print("Merged!!!")



import torch
from transformers import AutoModelForCausalLM
from peft import PeftModelForCausalLM
import os

model_path = "meta-llama/Meta-Llama-3-8B"


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained("/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/Llama-3-8B-Scaling-CE-lora-global_step450_hf/")

adapter_path = "/mnt/petrelfs/tangzecheng/local_ckpt/pg19/Llama-3-8B-Scaling-CE/lora/global_step450_hf" 

model_config = os.path.join(adapter_path,"config.json")

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, 
use_flash_attention_2="flash_attention_2", 
device_map="auto", config=AutoConfig.from_pretrained(model_config) if (model_config is not None and len(model_config)>0) else None )

if adapter_path:
    model = PeftModelForCausalLM.from_pretrained(model, adapter_path)#.eval()
    print("model_config:", model_config)
    
merged_model = model.merge_and_unload()

os.makedirs("/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/Llama-3-8B-Scaling-CE-lora-global_step450_hf/",exist_ok = True)

merged_model.save_pretrained(f"/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/Llama-3-8B-Scaling-CE-lora-global_step450_hf/")

print("Merged 2 !!!")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import login
import time
import os

# https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens

hf_token = os.getenv("HF_TOKEN")

# Authenticate using your token
login(token=hf_token)
device_name = "cuda"
device = torch.device(device_name)
# transformers/models/llama/modeling_llama.py
model = LlamaForCausalLM.from_pretrained("HF1BitLLM/Llama3-8B-1.58-100B-tokens", device_map=device_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# input_text = "What is the capital of US?"
input_text = "Is the sky blue?"

start_time = time.time()
print("cuda available? ", torch.cuda.is_available())

print("Start of inference...")
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)#.cuda()
output = model.generate(input_ids, max_length=20, do_sample=False, max_new_tokens=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
print("Time: ", time.time() - start_time)
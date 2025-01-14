import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import *
from torch import Neura

# Load a pretrained BitNet model
model = "merfanian/Bitnet-Llama-700M"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)

i = 0

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class BitLinear(nn.Linear):
    def forward(self, x):
        global i
        w = self.weight  # a weight tensor with shape [d, k]
        print(w.shape)
        x = x.to(w.device)
        RMSNorm = LlamaRMSNorm(x.shape[-1]).to(w.device)
        x_norm = RMSNorm(x)
        # A trick for implementing Straight−Through−Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y

def convert_to_bitnet(model, copy_weights):
    for name, module in model.named_modules():
        # Replace linear layers with BitNet
        if isinstance(module, LlamaSdpaAttention) or isinstance(module, LlamaMLP):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    bitlinear = BitLinear(child_module.in_features, child_module.out_features, child_module.bias is not None).to(device="cuda:0")
                    if copy_weights:
                        bitlinear.weight = child_module.weight
                        if child_module.bias is not None:
                            bitlinear.bias = child_module.bias
                    setattr(module, child_name, bitlinear)
        # Remove redundant input_layernorms
        elif isinstance(module, LlamaDecoderLayer):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, LlamaRMSNorm) and child_name == "input_layernorm":
                    setattr(module, child_name, nn.Identity().to(device="cuda:0"))

convert_to_bitnet(model, copy_weights=True)
model.to(device="cpu")

# Inference with timing
prompt = "What is the capitol of US?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Start timing
start_time = time.time()

generate_ids = model.generate(inputs.input_ids, max_length=100)

# End timing
end_time = time.time()

res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res)

# Print the time taken for inference
print(f"Inference Time: {end_time - start_time:.4f} seconds")

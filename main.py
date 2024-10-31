import hashlib
import os
import time

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.bitnet import log_execution_time

from Singleton import Cache

load_dotenv()
token = os.getenv("HF_ACCESS_TOKEN")

CACHE_FILE = "precomputed_cache.pkl"
original_matmul = torch.matmul

@log_execution_time
def hash_tensor(tensor: torch.Tensor) -> str:
    """Generate a hash for a tensor to use as a cache key."""
    tensor_bytes = tensor.cpu().numpy().tobytes()
    return hashlib.md5(tensor_bytes).hexdigest()


def load_model_and_tokenizer(model_name, tokenizer_name):
    """Load model and tokenizer with given parameters."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token)
    return model, tokenizer


def measure_inference_time(model, tokenizer, input_text):
    """Measure and print inference time for a given model."""
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
    start_time = time.time()  # Start timer
    output = model.generate(input_ids, max_length=128, do_sample=False)
    end_time = time.time()  # End timer

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    print(f"Inference Time: {end_time - start_time:.4f} seconds")
    return end_time - start_time


if __name__ == "__main__":
    cache = Cache()

    # Test input text
    input_texts = [(
        "Daniel went back to the the the garden. Mary travelled to the kitchen. "
        "Sandra journeyed to the kitchen. Sandra went to the hallway. "
        "John went to the bedroom. Mary went back to the garden. Where is Sandra?\nAnswer:"
    )]

    # Load the model and tokenizer
    bit_model_name = "HF1BitLLM/Llama3-8B-1.58-Linear-10B-tokens"
    bit_tokenizer_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    bit_model, bit_tokenizer = load_model_and_tokenizer(bit_model_name, bit_tokenizer_name)

    print("\n1-Bit Model Inference:")
    for i in range(1):
        for input_text in input_texts:
            bit_inference_time = measure_inference_time(bit_model, bit_tokenizer, input_text)

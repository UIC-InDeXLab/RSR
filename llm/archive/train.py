import argparse

import wandb
from datasets import load_dataset
from huggingface_hub import login, create_repo, HfApi
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from transformers.models.llama.modeling_llama import *


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
        w = self.weight
        x = x.to(w.device)
        RMSNorm = LlamaRMSNorm(x.shape[-1]).to(w.device)
        x_norm = RMSNorm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y


def convert_to_bitnet(model, copy_weights):
    for name, module in model.named_modules():
        if isinstance(module, LlamaSdpaAttention) or isinstance(module, LlamaMLP):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    bitlinear = BitLinear(
                        child_module.in_features,
                        child_module.out_features,
                        child_module.bias is not None
                    ).to(device="cuda:0")
                    if copy_weights:
                        bitlinear.weight = child_module.weight
                        if child_module.bias is not None:
                            bitlinear.bias = child_module.bias
                    setattr(module, child_name, bitlinear)
        elif isinstance(module, LlamaDecoderLayer):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, LlamaRMSNorm) and child_name == "input_layernorm":
                    setattr(module, child_name, nn.Identity().to(device="cuda:0"))


def main(args):
    wandb.login(key=args.wandb_token)
    login(token=args.hf_token)
    data = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_config)

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=False,
            max_length=args.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        combined = []
        for tokenized_doc in outputs['input_ids']:
            combined += tokenized_doc + [tokenizer.eos_token_id]
        input_batch = []
        for i in range(0, len(combined) - args.context_length, args.context_length):
            input_batch.append(combined[i:i + args.context_length])
        return {"input_ids": input_batch}

    tokenized_data = data.map(
        tokenize, batched=True, remove_columns=data["train"].column_names,
    )

    total_tokens = tokenized_data['train'].num_rows * args.context_length
    print(f"Training on {total_tokens:_} tokens")

    config = AutoConfig.from_pretrained(
        args.model_config,
        vocab_size=len(tokenizer),
        n_ctx=args.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    config.hidden_size = args.dimensions
    config.max_position_embeddings = args.dimensions
    config.num_attention_heads = args.heads
    config.num_hidden_layers = args.layers
    config.num_key_value_heads = args.heads
    config.intermediate_size = args.intermediate_size

    model = LlamaForCausalLM(config)
    convert_to_bitnet(model, copy_weights=False)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args_train = TrainingArguments(
        output_dir="./out",
        per_device_train_batch_size=args.batch_size,
        logging_steps=100,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=1,
        lr_scheduler_type="cosine",
        learning_rate=args.learning_rate,
        save_steps=0.25,
        fp16=True,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args_train,
        data_collator=data_collator,
        train_dataset=tokenized_data["train"],
    )

    trainer.train()
    trainer.save_model("./out/final_model")

    api = HfApi()
    create_repo(
        repo_id=f"{args.huggingface_id}/{args.new_model}",
        repo_type="model",
        exist_ok=True,
        token=args.hf_token,
    )

    api.upload_folder(
        folder_path="./out/final_model",
        repo_type="model",
        repo_id=f"{args.huggingface_id}/{args.new_model}",
        token=args.hf_token,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and upload a BitNet-Llama model.")
    parser.add_argument("--model_config", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dimensions", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--new_model", type=str, default="Bitnet-Llama-700M")
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--wandb_token", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="abideen/Cosmopedia-100k-pretrain")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.5e-3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--huggingface_id", type=str, required=True)

    args = parser.parse_args()
    main(args)

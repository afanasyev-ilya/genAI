# fine_tune_tinyllama_instruct_iterative.py

# 1) INSTALLS (run once)
# -----------------------
# pip install transformers accelerate peft bitsandbytes datasets trl

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# (Optional) allow TF32 on Ampere+ GPUs for a small speed‑up
torch.backends.cuda.matmul.allow_tf32 = True


# 2) CONFIGURE 4‑BIT QUANTIZATION
# --------------------------------
# We quantize the base model to 4 bits (NF4) in bfloat16 compute dtype
# This drastically cuts GPU memory usage for a ~1.1B‑param model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                 # enable 4-bit loading
    bnb_4bit_quant_type="nf4",         # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16  # computations in bfloat16
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
print(f"\nLoading base model {MODEL_ID} in 4‑bit…")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
# Ensure a PAD token exists so the collator can batch properly
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    base_model.resize_token_embeddings(len(tokenizer))


# 3) PRINT MODEL SIZE
# --------------------
total_params = sum(p.numel() for p in base_model.parameters())
print(f"Base model total parameters: {total_params/1e6:.1f}M")

# LoRA adapters will add only a small fraction of trainable params
def print_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (LoRA adapters): {trainable/1e6:.3f}M "
          f"({100*trainable/total_params:.3f}% of total)")

# 4) RESPONSE GENERATOR
# ----------------------
def generate_response(model, prompt: str, max_new_tokens: int = 64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# 5) EVALUATION SUITE
# --------------------
# A small set of prompts to watch improvement
eval_prompts = [
    "Explain quantum computing in simple terms",
    "What is the capital of France?",
    "How does a car engine work?",
]

def evaluate(model, stage: str):
    print(f"\n=== Evaluating model: {stage} ===")
    for q in eval_prompts:
        print(f"> Prompt: {q}")
        print(generate_response(model, q))
    print("="*40)


# 6) SHOW BASE MODEL RESPONSES
# -----------------------------
print("\n>>> Before any fine‑tuning:")
evaluate(base_model, "BASE")


# 7) PREPARE INSTRUCTION DATA
# ----------------------------
print("\nLoading dataset")
dataset = load_dataset("mlabonne/guanaco-llama2-1k")
train_dataset = dataset["train"]

# 8) CONFIGURE LoRA
# ------------------
# LoRA = Low‑Rank Adaptation of large language models.
# We insert small "adapter" matrices into certain weight projections,
# freeze the base parameters, and train only these adapters.
peft_config = LoraConfig(
    r=16,                   # rank of the low‑rank update matrices
    lora_alpha=32,          # scaling factor: controls adapter initialization scale
    target_modules=[        # which linear layers to adapt
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj",
        "up_proj", "down_proj"
    ],
    lora_dropout=0.05,      # dropout applied to adapter outputs
    bias="none",            # no bias adapters
    task_type="CAUSAL_LM"
)
print("\nApplying LoRA adapters…")
model = get_peft_model(base_model, peft_config)
print_trainable(model)  # show trainable params after LoRA

# 9) TRAINING ARGUMENTS
# ----------------------
# We use SFTTrainer for supervised fine‑tuning (SFT):
# it handles shuffling, batching, and loss computation for causal LM.
training_args = TrainingArguments(
    output_dir="./tinyllama-instruct",
    num_train_epochs=3,                    # one epoch per stage
    per_device_train_batch_size=1,         # batch size
    gradient_accumulation_steps=4,         # accumulate grads to simulate bs=4
    learning_rate=2e-4,                    # initial LR
    optim="paged_adamw_8bit",              # 8‑bit paged AdamW optimizer
    bf16=True,                             # bfloat16 training
    logging_steps=10,                      # log every 10 steps
    save_strategy="no",                    # disable auto‑save during iterative stages
    max_grad_norm=0.3,                     # gradient clipping
    warmup_ratio=0.03,                     # LR warmup
    lr_scheduler_type="cosine"             # cosine decay
)

# 10) ITERATIVE TRAINING LOOP
# ----------------------------
# We fine‑tune in three stages: 50 → 100 → 200 examples,
# evaluating after each stage to see gradual improvements.
stages = [50, 100, 200, 500, 1000]  # cumulative counts
prev_idx = 0

for end_idx in stages:
    # 1) select only the new slice
    new_slice = train_dataset.select(range(prev_idx, end_idx))
    
    print(f"\n>>> Training on samples {prev_idx} to {end_idx} (total {end_idx - prev_idx}) …")
    
    # 2) re‑use the *same* model and args
    trainer = SFTTrainer(
        model=model,
        train_dataset=new_slice,
        args=training_args
    )
    # 3) run training on this slice
    trainer.train()
    
    # 4) evaluate to see the incremental improvement
    evaluate(model, f"after examples {prev_idx}-{end_idx}")
    
    # 5) advance the pointer so we don’t re‑train on these again
    prev_idx = end_idx

total_params = sum(p.numel() for p in model.parameters())
print(f"Fine-tuned model total parameters: {total_params/1e6:.1f}M")

# 11) SAVE FINAL MODEL
# ---------------------
model.save_pretrained("./tinyllama-instruct")
tokenizer.save_pretrained("./tinyllama-instruct")
print("\nAll done! Final model saved to ./tinyllama-instruct")

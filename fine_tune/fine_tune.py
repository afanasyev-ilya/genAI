from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer

# python3 -m pip install transformers accelerate peft bitsandbytes datasets trl

model_url = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_url,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_url)

instruct_prompt = "Explain quantum computing in simple terms "
base_prompt = "Qunatum computing is "

def generate_response(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Base model response (before fine-tuning)
base_response = generate_response(base_model, base_prompt)
print(f"Base Model:\n{base_response}\n")

output_dir = "./tinyllama-instruct"
response_template = "\n### Response:"  # Used to identify response section

# LoRA configuration
peft_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
base_model = get_peft_model(base_model, peft_config)

# ------------------- prepare dataset --------------
# Load and prepare dataset
dataset = load_dataset("mlabonne/guanaco-llama2-1k")
train_dataset = dataset["train"]

#for i in range(0, 2):
#    print(train_dataset[i])
# ------------------- done -------------------------

# ------------------- train --------------
# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset.select(range(200)),
    args=training_args
)

# Start training
trainer.train()

# Save model
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
# ------------------- train --------------


# Test fine-tuned model
instruct_prompt = "Explain quantum computing in simple terms"
ft_response = generate_response(base_model, instruct_prompt)
print(f"Fine-tuned Model:\n{ft_response}")

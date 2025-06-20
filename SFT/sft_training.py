import torch

from datasets import load_dataset
from transformers import Qwen3ForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("pawin205/iclr-2017-2020-peer-review-with-thinking-trace", split="90thPercentile")

dataset = dataset.remove_columns("prompt")
dataset = dataset.rename_column("conversations", "messages")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

model = Qwen3ForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda"    
)
model.gradient_checkpointing_enable()

training_args = SFTConfig(
    output_dir="saves/REMORX-Qwen3-SFT",
    run_name="REMORX-Qwen3-SFT",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine",
    learning_rate=1.0e-4,
    logging_steps=1,
    max_length=32768,
    num_train_epochs=3,
    report_to="wandb",
    gradient_checkpointing=True,
    eos_token=tokenizer.eos_token,
)

trainer = SFTTrainer(
    "Qwen/Qwen3-8B",
    train_dataset=dataset,
    args=training_args
)

trainer.train()

trainer.save_model(training_args.output_dir)
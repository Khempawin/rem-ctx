import torch

from datasets import load_dataset
from transformers import Qwen3ForCausalLM
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("pawin205/iclr-2017-2020-peer-review-with-thinking-trace", split="90thPercentile")

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
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    lr_scheduler_type="cosine",
    learning_rate=1.0e-4,
    logging_steps=1,
    max_length=32768,
    output_dir="tmp",
    num_train_epochs=3,
    report_to="none",
    gradient_checkpointing=True,
    deepspeed=""
)

trainer = SFTTrainer(
    "Qwen/Qwen3-8B",
    train_dataset=dataset,
    args=training_args
)

trainer.train()

trainer.save_model(training_args.output_dir)
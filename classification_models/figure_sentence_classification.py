from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset

import evaluate
import numpy as np
import torch

torch.set_float32_matmul_precision('high')

#model_name = "answerdotai/ModernBERT-base"
model_name = "answerdotai/ModernBERT-large"
hub_model_id = "pawin205/figure-correspondence-classifier-large"
output_dir = "figure_result_large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, device_map="cuda")

print(model.device)

train_ds = load_dataset("pawin205/FigSentRelevance", split="train")
val_ds = load_dataset("pawin205/FigSentRelevance", split="val")

def tokenize_function(examples):
    return tokenizer(examples["text_input"], truncation=True, padding=True)

train_dataset = train_ds.map(tokenize_function, batched=True)
val_dataset = val_ds.map(tokenize_function, batched=True)

f1_metric = evaluate.load("f1")


def compute_metrics(p):
    logits, labels = p.predictions, p.label_ids
    predictions = np.argmax(logits, axis=1)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"f1": f1}


training_args = TrainingArguments(
    output_dir=output_dir,
    hub_model_id=hub_model_id,
    push_to_hub=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=100,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    #logging_dir="figure_logs"
    report_to="wandb",
    run_name="figure_correspondence_classifier"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

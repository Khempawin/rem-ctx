from typing import TypedDict, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import pandas as pd

class ModelConfig(TypedDict):
    model_id: str
    name: str


def create_prompt(full_text: str, figure_details: Optional[str], novelty_assessment: Optional[str]) -> str:
    prompt = f"You are a member of the scientific community tasked with peer review. Review the following paper content.\n\n### Paper Content\n\n{full_text}"
    
    if (figure_details is not None):
        prompt += f"\n\n### Figure Details\n{figure_details}"
    else:
        prompt += f"\n\n### Figure Details\nNone"
        
    if (novelty_assessment is not None):
        prompt += f"\n\n### Novelty Assessment\n{novelty_assessment}"
    else:
        prompt += f"\n\n### Novelty Assessment\nNone"
    
    return prompt


def create_prompt_for_dataset(sample, include_figure_details: bool, include_novelty_assessment: bool):
    prompt = create_prompt(
        full_text=sample["full_text"], 
        figure_details=sample["figure_details"] if include_figure_details else None,
        novelty_assessment=sample["novelty_assessment"] if include_novelty_assessment else None
    )
    return {
        "prompt": prompt
    }


def preprocess_function(sample):
    return {
        "prompt": [
                {
                    "content": sample["prompt"],
                    "role": "user"
                }
            ],
        "full_text": sample["full_text"],
        "figure_details": sample["figure_details"],
        "novelty_assessment": sample["novelty_assessment"],
    }
    
    
def generate_reviews_from_dataset(ds: Dataset, model_name: str) -> pd.DataFrame:
    prompts = [s["prompt"] for s in ds]
    # Truncate prompts
    MAX_PROMPT_LENGTH = 32768
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    truncated_prompts: list[str] = list()
    for prompt in prompts:
        tokens = tokenizer(prompt[0]["content"], return_tensors="pt", truncation=False)
        token_ids = tokens["input_ids"][0]
        token_ids = token_ids[:MAX_PROMPT_LENGTH - 2]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        conversation = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        truncated_prompts.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        
    # Load model    
    llm = LLM(model=model_name, tensor_parallel_size=2)
    sampling_params = SamplingParams(
        max_tokens=4096, repetition_penalty=1, temperature=0.6
    )
    
    # Generate reviews from prompts
    outputs = llm.generate(prompts=truncated_prompts, sampling_params=sampling_params)
    reviews = [response.outputs[0].text for response in outputs]
    
    df = ds.to_pandas()
    
    df["generated_reviews"] = reviews
    
    return df


def main():
    # Load dataset pawin205/PeerRTEx
    dataset: Dataset = load_dataset("pawin205/PeerRTEx", split="train")


    # Define models to load
    model_ids = [
        # ModelConfig(model_id="Qwen/Qwen3-8B", name="q3"),
        # ModelConfig(model_id="pawin205/Qwen3-8B-GRPO-REMOR-U", name="remor"),
        # ModelConfig(model_id="pawin205/REMORX-UREX-Qwen3", name="remorx"),
        # ModelConfig(model_id="pawin205/REMORX-UREX-Qwen3-Cont", name="remorx_ep4"),
        # ModelConfig(model_id="pawin205/Qwen3-8B-REMOR-SFT-lora", name="remor_sft_lora"),
        # ModelConfig(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", name="ds_r1_qwen7b"), # Works better than Qwen3
        # ModelConfig(model_id="pawin205/Qwen3-8B-REMOR-U-SFT", name="q3_sft"), # Works properly with temperature of 0.6
        # ModelConfig(model_id="pawin205/Qwen-7B-REMOR-SFT", name="ds_r1_qwen7b_sft"),
        # ModelConfig(model_id="pawin205/Qwen3-8B-REMOR-U-GRPO-Agg", name="remorx-agg"),
        # ModelConfig(model_id="pawin205/Qwen3-8B-REMOR-U-GRPO-Agg-Cont", name="remorx-agg-ep2"),
        # ModelConfig(model_id="pawin205/Qwen3-8B-REMOR-U-GRPO-Agg-Cont-ep3", name="remorx-agg-ep3"),
        # ModelConfig(model_id="pawin205/Qwen3-8B-REMOR-U-GRPO-Agg-Cont-ep4", name="remorx-agg-ep4"),
        ModelConfig(model_id="pawin205/Qwen3-8B-REMOR-U-GRPO-Agg-Cont-ep5", name="remorx-agg-ep5"),
        ModelConfig(model_id="pawin205/Qwen3-8B-REMOR-U-GRPO-Agg-Cont-ep6", name="remorx-agg-ep6"),
        ModelConfig(model_id="pawin205/Qwen3-8B-REMOR-U-GRPO-Agg-Cont-ep7", name="remorx-agg-ep7"),
        # ModelConfig(model_id="pawin205/REMOR-U-GRPO-Agg-TPR", name="remor_tpr")
    ]

    ablation_configs = [
        {"include_figure_details": True, "include_novelty_assessment": True},
        {"include_figure_details": True, "include_novelty_assessment": False},
        {"include_figure_details": False, "include_novelty_assessment": True},
        {"include_figure_details": False, "include_novelty_assessment": False}        
    ]
    abl_config_names = [
        "full_option", 
        "fig", "novel", "none"
    ]
    
    for config, config_name in zip(ablation_configs, abl_config_names):
        dataset_processed: Dataset = dataset.map(create_prompt_for_dataset, fn_kwargs=config)
        dataset_processed: Dataset = dataset_processed.map(preprocess_function)
        # For each model
        for model_config in model_ids:
            #   Generate Review
            generated_reviews_df = generate_reviews_from_dataset(dataset_processed, model_config["model_id"])
            generated_reviews_df.to_parquet("generated_reviews/{}_reviews_{}.parquet".format(model_config["name"], config_name), engine="pyarrow")
            # print(f"Completed generation for model {model_config['model_id']} with config {config_name}")
            

        
if __name__ == "__main__":
    main()
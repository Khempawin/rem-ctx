import re
import math
import nltk
import torch
import argparse


from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline, AutoModelForCausalLM, Qwen2ForCausalLM, AutoConfig
from typing import List, TypedDict, Literal
from importlib.metadata import version
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import sent_tokenize


print("transformers version: {}".format(version('transformers')))
print("pandas version: {}".format(version('pandas')))


nltk.download("punkt_tab")
nltk.download('wordnet')
nltk.download('omw-1.4')


class ReviewSentenceCriteriaCount(TypedDict):
    criticism: int
    example: int
    importance_and_relevance: int
    materials_and_methods: int
    praise: int
    presentation_and_reporting: int
    results_and_discussion: int
    suggestion_and_solution: int
    total: int

# Load aspect classifier models
JIF_MODELS = [
        "criticism",
        "example",
        "importance_and_relevance",
        "materials_and_methods",
        "praise",
        "presentation_and_reporting",
        "results_and_discussion",
        "suggestion_and_solution"
    ]

## load tokenizer (same for all models handled here)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

## load classifier models
classifier_dict = dict()

for model_idx in JIF_MODELS:
    current_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert_models/" + model_idx, num_labels = 2
    )
    classifier:TextClassificationPipeline = pipeline(
        "text-classification",
        model=current_model,
        tokenizer=tokenizer,
        device="cuda:0"
    )
    classifier_dict[model_idx] = classifier
    
def classify_review_sentences(sentences: List[str]) -> ReviewSentenceCriteriaCount:
    criteria_count = ReviewSentenceCriteriaCount(
        criticism=0,
        example=0,
        importance_and_relevance=0,
        materials_and_methods=0,
        praise=0,
        presentation_and_reporting=0,
        results_and_discussion=0,
        suggestion_and_solution=0,
        total=0
    )
    
    tokenizer_kwarg= { "padding":True, "truncation":True, "max_length":512 }

    for model_idx in JIF_MODELS:
        # run classification pipeline
        results = classifier_dict[model_idx](sentences, **tokenizer_kwarg)
        positive_criteria_count = len(list(filter(lambda x: x["label"] == "LABEL_1", results)))
        criteria_count[model_idx] = positive_criteria_count
    criteria_count["total"] = len(sentences)
    
    return criteria_count


def calculate_meteor_score_review_full_text(review, full_text):
    if(full_text == "ERROR"):
        return 0.0

    # Tokenize your inputs
    candidate = nltk.word_tokenize(review)

    reference_list = [nltk.word_tokenize(full_text)]

    # Calculate METEOR score
    return meteor_score(reference_list, candidate)


def calc_reward(sentence_criteria_count: ReviewSentenceCriteriaCount) -> float:
    if (sentence_criteria_count["total"] == 0):
        return -10
    reward = 0
    for metric in JIF_MODELS:
        reward += sentence_criteria_count[metric] / sentence_criteria_count["total"]
    return reward


def calc_ratio_with_aspect(sentences: List[str], aspect: Literal["criticism",
                                                                            "example",
                                                                            "importance_and_relevance",
                                                                            "materials_and_methods",
                                                                            "praise",
                                                                            "presentation_and_reporting",
                                                                            "results_and_discussion",
                                                                            "suggestion_and_solution"]
    ) -> float:

    if(len(sentences) == 0):
        return -2

    tokenizer_kwarg= { "padding":True, "truncation":True, "max_length":512 }
    classification_result = classifier_dict[aspect](sentences, **tokenizer_kwarg)
    ratio = len(list(filter(lambda x: x["label"] == "LABEL_1", classification_result))) / float(len(sentences))
    
    return ratio


def reward_aspect(completions: List[str], aspect: Literal["criticism",
                                                                            "example",
                                                                            "importance_and_relevance",
                                                                            "materials_and_methods",
                                                                            "praise",
                                                                            "presentation_and_reporting",
                                                                            "results_and_discussion",
                                                                            "suggestion_and_solution"]
    ):

    # Get review after thinking traces
    reviews = [review.split("</think>") for review in completions]
    reviews = [review[1].strip() if len(review) > 1 else review[0].strip() for review in reviews]

    # Tokenize each review at sentences level
    tokenized_sentences = [sent_tokenize(review) for review in reviews]
    
    # Classify sentences and calculate reward for each review
    rewards = [calc_ratio_with_aspect(sentences, aspect) for sentences in tokenized_sentences]
    
    return rewards


def reward_all_aspects_of_review(completions: List[str], **kwargs) -> List[float]:
    # Tokenize each review at sentences level
    tokenized_sentences = [sent_tokenize(review) for review in completions]
    
    # Classify sentences of each review
    classification_results = [classify_review_sentences(sentences) for sentences in tokenized_sentences]
    
    # Calculate reward
    rewards = [calc_reward(sentence_criteria_count) for sentence_criteria_count in classification_results]
    
    #rewards = [math.sqrt((25 - len(sentences))**2/64) * -1 for sentences in tokenized_sentences]

    return rewards


def reward_criticism(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "criticism")


def reward_example(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "example")


def reward_importance_and_relevance(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "importance_and_relevance")


def reward_materials_and_methods(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "materials_and_methods")


def reward_praise(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "praise")


def reward_presentation_and_reporting(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "presentation_and_reporting")


def reward_results_and_discussion(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "results_and_discussion")


def reward_suggestions_and_solution(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "suggestion_and_solution")


def reward_meteor(completions: List[str], **kwargs):
    # Get review after thinking traces
    reviews = [review.split("</think>") for review in completions]
    reviews = [review[1] if len(review) > 1 else review[0] for review in reviews]

    # Calculate meteor score compared to the full text
    rewards = [calculate_meteor_score_review_full_text(review, full_text) for review, full_text  in zip(reviews, kwargs["full_text"])]

    return rewards


def strict_format_reward_func(completions: list[str], **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r"^<think>.*?</think>\n.+?\.$", re.DOTALL)
    matches = [pattern.match(response) for response in completions]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions: list[str], **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r"<think>.*?</think>\s*.+?\.", re.DOTALL)
    matches = [pattern.match(response) for response in completions]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.250
    if text.count("</think>") == 1:
        count += 0.250    
    return count


def xmlcount_reward_func(completions: list[str], **kwargs) -> list[float]:
    return [count_xml(response) for response in completions]


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract add details for TPR records from PDFs"
    )
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--data-split", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--deepspeed-config-path", type=str, required=True)
    parser.add_argument("--report-to", type=str, default="none")
    
    args = parser.parse_args()

    model_path = args.base_model
    dataset_id = args.dataset_id
    data_split = args.data_split
    output_dir = args.output_dir
    run_name = args.run_name
    deepspeed_config_path = args.deepspeed_config_path
    report_to = args.report_to

    dataset = load_dataset(dataset_id, split=f"{data_split}[:80]")

    config = AutoConfig.from_pretrained(model_path)
    config.use_sliding_window = True
    config.max_window_layers = 16
    config.sliding_window = 4096

    model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0"
    )

    reward_tokenizer = AutoTokenizer.from_pretrained(model_path)
    reward_tokenizer.pad_token = tokenizer.eos_token
    reward_tokenizer.model_max_length = 131072
    model.gradient_checkpointing_enable()

    training_args = GRPOConfig(
            output_dir=output_dir,
            run_name=run_name,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=1,
            lr_scheduler_type='cosine',
            logging_steps=1,
            bf16=True,
            num_generations=4,
            max_prompt_length=32768,
            max_completion_length=4096,
            num_train_epochs=1,
            save_steps=100,
            use_vllm=False,
            vllm_gpu_memory_utilization=0.7,
            report_to=report_to,
            gradient_checkpointing=True,
            deepspeed=deepspeed_config_path,
            )

    trainer = GRPOTrainer(
        model=model,
        processing_class=reward_tokenizer,
        reward_funcs=[
            reward_criticism,
            reward_example,
            reward_importance_and_relevance,
            reward_materials_and_methods,
            reward_praise,
            reward_presentation_and_reporting,
            reward_results_and_discussion,
            reward_suggestions_and_solution,
            reward_meteor,
            strict_format_reward_func,
            soft_format_reward_func,
            xmlcount_reward_func
            ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()


from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline, AutoModelForCausalLM, Qwen2ForCausalLM
from typing import TypedDict, Literal, Optional
from importlib.metadata import version
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import sent_tokenize

import os
import re
import math
import nltk
import torch


print("transformers version: {}".format(version('transformers')))
print("pandas version: {}".format(version('pandas')))
print("trl version: {}".format(version('trl')))


nltk.download("punkt_tab")
nltk.download('wordnet')
nltk.download('omw-1.4')


class ConversationItem(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    

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

REWARD_GPU_DEVICE = "cuda:0"

## load figure correspondence classifier
classifier_figure = pipeline("text-classification", model="pawin205/figure-correspondence-classifier-large", dtype=torch.bfloat16, device=REWARD_GPU_DEVICE)

## load external knowledge classifier
classifier_novelty = pipeline("text-classification", model="pawin205/novelty-correspondence-classifier-large", dtype=torch.bfloat16, device=REWARD_GPU_DEVICE)

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
        device=REWARD_GPU_DEVICE
    )
    classifier_dict[model_idx] = classifier
    

def create_prompt(full_text: str, figure_details: Optional[str], novelty_assessment: Optional[str]) -> str:
    prompt = f"You are a member of the scientific community tasked with peer review. Review the following paper content.\n\n### Paper Content\n\n{full_text}"
    
    if (figure_details is not None):
        prompt += f"\n\n### Figure Details\n{figure_details}"
        
    if (novelty_assessment is not None):
        prompt += f"\n\n### Novelty Assessment\n{novelty_assessment}"
    
    return prompt
    

def preprocess_function(sample):
    return {
        "prompt": [
                {
                    "content": create_prompt(sample["full_text"], None, None),
                    "role": "user"
                }
            ],
        "full_text": sample["full_text"],
        "figure_details": sample["figure_details"],
        "novelty_assessment": sample["novelty_assessment"],
    }


def get_review_from_completion(
    completions: list[list[ConversationItem]],
    include_thinking_trace: bool
) -> list[str]:
    # review_after_thinking_trace
    reviews = [completion[0]["content"] for completion in completions]

    if (include_thinking_trace):
        return reviews

    reviews = [review.split("</think>") for review in reviews]
    reviews = [review[1].strip() if len(review) > 1 else review[0].strip() for review in reviews]
    
    return reviews


def calc_ratio_with_aspect(
    sentences: list[str], 
    aspect: Literal[
        "criticism",
        "example",
        "importance_and_relevance",
        "materials_and_methods",
        "praise",
        "presentation_and_reporting",
        "results_and_discussion",
        "suggestion_and_solution"
    ]
) -> float:

    if(len(sentences) == 0):
        return -2

    tokenizer_kwarg= { "padding":True, "truncation":True, "max_length":512 }
    classification_result = classifier_dict[aspect](sentences, **tokenizer_kwarg)
    ratio = len(list(filter(lambda x: x["label"] == "LABEL_1", classification_result))) / float(len(sentences))
    
    return ratio


def reward_aspect(
    completions: list[list[ConversationItem]], 
    aspect: Literal[
        "criticism",
        "example",
        "importance_and_relevance",
        "materials_and_methods",
        "praise",
        "presentation_and_reporting",
        "results_and_discussion",
        "suggestion_and_solution"
    ]
):

    # Get review after thinking traces
    reviews = get_review_from_completion(completions, include_thinking_trace=False)
 
    # Tokenize each review at sentences level
    tokenized_sentences = [sent_tokenize(review) for review in reviews]
    
    # Classify sentences and calculate reward for each review
    rewards = [calc_ratio_with_aspect(sentences, aspect) for sentences in tokenized_sentences]
    
    return rewards


def reward_criticism(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    return reward_aspect(completions, "criticism")


def reward_example(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    return reward_aspect(completions, "example")


def reward_importance_and_relevance(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    return reward_aspect(completions, "importance_and_relevance")


def reward_materials_and_methods(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    return reward_aspect(completions, "materials_and_methods")


def reward_praise(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    return reward_aspect(completions, "praise")


def reward_presentation_and_reporting(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    return reward_aspect(completions, "presentation_and_reporting")


def reward_results_and_discussion(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    return reward_aspect(completions, "results_and_discussion")


def reward_suggestions_and_solution(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    return reward_aspect(completions, "suggestion_and_solution")


def calculate_meteor_score_review_full_text(review: str, full_text: str) -> float:
    if(full_text == "ERROR"):
        return 0.0

    # Tokenize your inputs
    candidate = nltk.word_tokenize(review)

    reference_list = [nltk.word_tokenize(full_text)]
    
    # Calculate METEOR score
    return meteor_score(reference_list, candidate) / 0.25


def reward_meteor(completions: list[list[ConversationItem]], **kwargs):
    # Get review after thinking traces
    reviews = get_review_from_completion(completions, include_thinking_trace=False)
 
    # Calculate meteor score compared to the full text
    rewards = [calculate_meteor_score_review_full_text(review, full_text) for review, full_text  in zip(reviews, kwargs["full_text"])]

    return rewards


def strict_format_reward_func(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r"^<think>.*?</think>\n.+?\.$", re.DOTALL)
    responses = get_review_from_completion(completions, include_thinking_trace=True)
    matches = [pattern.match(response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r"<think>.*?</think>\s*.+?\.", re.DOTALL)
    responses = get_review_from_completion(completions, include_thinking_trace=True)
    matches = [pattern.match(response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.250
    if text.count("</think>") == 1:
        count += 0.250    
    return count


def xmlcount_reward_func(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    responses = get_review_from_completion(completions, include_thinking_trace=True)
    return [count_xml(response) for response in responses]


def calculate_figure_correspondence(review: str, figure_details: str) -> float:
    # classifier_figure
    # Tokenize review at sentences level
    tokenized_sentences = sent_tokenize(review)
    
    if(len(tokenized_sentences) == 0):
        return 0.0
    
    if(len(figure_details) == 0):
        return 1.0
    
    # Classify sentences
    figure_correspondence_prompts = [
        f"### Figure Details\n{figure_details}\n\n### Conclusion\n{sentence}" for sentence in tokenized_sentences
    ]
    
    classification_results = classifier_figure(figure_correspondence_prompts)
    
    involve_figure_count_no_conflict = len(list(filter(lambda x: x["label"] == "LABEL_0", classification_results)))
    involve_figure_count_conflict = len(list(filter(lambda x: x["label"] == "LABEL_1", classification_results)))
    
    involve_figure_count = involve_figure_count_conflict + involve_figure_count_no_conflict
    
    reward = 0.0 if involve_figure_count == 0 else (involve_figure_count_no_conflict / involve_figure_count)
    return reward


def reward_figure_correspondence(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    reviews = [completion[0]["content"] for completion in completions]
    
    # Get review after thinking traces
    reviews = [review.split("</think>") for review in reviews]
    reviews = [review[1] if len(review) > 1 else review[0] for review in reviews]

    # Calculate meteor score compared to the full text
    rewards = [calculate_figure_correspondence(review, figure_details) for review, figure_details  in zip(reviews, kwargs["figure_details"])]

    return rewards


def calculate_novelty_correspondence(review: str, novelty_assessment: str) -> float:
    # classifier_figure
    # Tokenize review at sentences level
    tokenized_sentences = sent_tokenize(review)
    
    if(len(tokenized_sentences) == 0):
        return 0.0
    
    if(len(novelty_assessment) == 0):
        return 1.0
    
    # Classify sentences
    novelty_correspondence_prompts = [
        f"### Novelty Assessment\n{novelty_assessment}\n\n### Conclusion\n{sentence}" for sentence in tokenized_sentences
    ]
    
    classification_results = classifier_novelty(novelty_correspondence_prompts)
    
    involve_novelty_count_no_conflict = len(list(filter(lambda x: x["label"] == "LABEL_0", classification_results)))
    involve_novelty_count_conflict = len(list(filter(lambda x: x["label"] == "LABEL_1", classification_results)))
    
    involve_novelty_count = involve_novelty_count_conflict + involve_novelty_count_no_conflict
    
    reward = 0.0 if involve_novelty_count == 0 else (involve_novelty_count_no_conflict / involve_novelty_count)
    return reward


def reward_novelty_correspondence(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    reviews = [completion[0]["content"] for completion in completions]
    
    # Get review after thinking traces
    reviews = [review.split("</think>") for review in reviews]
    reviews = [review[1] if len(review) > 1 else review[0] for review in reviews]

    # Calculate meteor score compared to the full text
    rewards = [calculate_novelty_correspondence(review, novelty_assessment) for review, novelty_assessment  in zip(reviews, kwargs["novelty_assessment"])]

    return rewards


def classify_review_sentences(sentences: list[str]) -> ReviewSentenceCriteriaCount:
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


def calc_reward(sentence_criteria_count: ReviewSentenceCriteriaCount) -> float:
    if (sentence_criteria_count["total"] == 0):
        return -10
    reward = 0
    for metric in JIF_MODELS:
        reward += sentence_criteria_count[metric] / sentence_criteria_count["total"]
    return reward


# def reward_all_aspects_of_review(completions: List[str], **kwargs) -> List[float]:
def reward_all_aspects_of_review(completions: list[list[ConversationItem]], **kwargs) -> list[float]:
    reviews = [completion[0]["content"] for completion in completions]
    
    # Get review after thinking traces
    reviews = [review.split("</think>") for review in reviews]
    reviews = [review[1].strip() if len(review) > 1 else review[0].strip() for review in reviews]

    # Tokenize each review at sentences level
    tokenized_sentences = [sent_tokenize(review) for review in reviews]
    
    # Classify sentences of each review
    classification_results = [classify_review_sentences(sentences) for sentences in tokenized_sentences]
    
    # Calculate reward
    rewards = [calc_reward(sentence_criteria_count) for sentence_criteria_count in classification_results]
    
    return rewards


def main():
    os.environ["WANDB_PROJECT"] = "REMORX"
    # model_path = "pawin205/Qwen-7B-Review-ICLR-90th-sft"
    # model_path = "pawin205/Qwen-7B-Review-ICLR-GRPO-UR"
    # model_path = "pawin205/REMORX-UREX-Qwen3"
    # model_path = "pawin205/Qwen3-8B-REMOR-SFT-lora"
    model_path = "pawin205/Qwen3-8B-REMOR-U-SFT"
    # model_path = "pawin205/Qwen3-8B-REMOR-U-GRPO-Agg"
    # model_path = "pawin205/Qwen3-8B-REMOR-U-GRPO-Agg-Cont"
    # model_path = "pawin205/Qwen3-8B-REMOR-U-GRPO-Agg-Cont-ep3"
    # model_path = "pawin205/Qwen3-8B-REMOR-U-GRPO-Agg-Cont-ep4"
    hub_model_id = "pawin205/REMOR-U-GRPO-Agg-TPR"

    dataset: Dataset = load_dataset("pawin205/PeerRTEx", split="train")
    dataset: Dataset = dataset.map(preprocess_function, remove_columns=[
        'title', 'abstract', 'major_discipline', 'minor_discipline', 'source', 'year', 'pdf_file_path', 'full_text_length', 'list_of_reference', 'prompt_length'
    ])
    
    reward_tokenizer = AutoTokenizer.from_pretrained(model_path)
    # reward_tokenizer.pad_token = tokenizer.eos_token

    training_args = GRPOConfig(
            run_name="EKA-separate-metric-agg-REMOR-U",
            hub_model_id=hub_model_id,
            push_to_hub=True,
            report_to="wandb",
            seed=42,
            data_seed=42,
            output_dir="saves/remorx-checkpoints/REMORX-UREX-test",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            lr_scheduler_type='cosine',
            logging_steps=1,
            bf16=True,
            num_generations=4,
            max_prompt_length=32768,
            max_completion_length=4096,
            num_train_epochs=1,
            save_steps=200,
            model_init_kwargs={
                "dtype": torch.bfloat16
            },
            #reward_weights=[
            #    1,
            #    1,
            #    1,
            #    1,
            #    1,
            #    1,
            #    1
            #]
            )

    trainer = GRPOTrainer(
        model=model_path,
        processing_class=reward_tokenizer,
        reward_funcs=[
            # reward_all_aspects_of_review,
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
            xmlcount_reward_func,
            # reward_figure_correspondence,
            # reward_novelty_correspondence
            ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()


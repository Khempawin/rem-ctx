import nltk
import pandas as pd
import torch

from pathlib import Path
from typing import TypedDict, Any
from nltk.tokenize import sent_tokenize
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline
from tqdm import tqdm

nltk.download("punkt_tab")


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

classifier_dict = dict()

# Load classifier models to reduce loading overhead but will consume more VRAM
for model_idx in JIF_MODELS:
    current_model = AutoModelForSequenceClassification.from_pretrained(
        f"distilbert_models/{model_idx}", num_labels=2
    )
    classifer:TextClassificationPipeline = pipeline(
        "text-classification",
        model=current_model,
        tokenizer=tokenizer
    )
    classifier_dict[model_idx] = classifer
    

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


def calculate_meteor_score(review: str, full_text: str) -> float:
    # Tokenize your inputs
    candidate = nltk.word_tokenize(review)

    reference_list = [nltk.word_tokenize(full_text)]

    # Calculate METEOR score
    return meteor_score(reference_list, candidate)


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


def process_review_record(review_record: dict[str, Any], include_thinking_trace: bool=True) -> dict[str, Any]:
    # tokenize review into sentences
    if (include_thinking_trace):
        review = review_record["generated_reviews"]
    else:
        review = review_record["generated_reviews"].split("</think>")
        review = review[1] if len(review) > 1 else review[0]
    
    review_sentences = sent_tokenize(review)
    
    # get criteria count
    criteria_count = classify_review_sentences(review_sentences)
    
    # calculate criteria scores
    remor_u_score = 0
    for criteria in JIF_MODELS:
        review_record[criteria] = criteria_count[criteria] / criteria_count["total"] if criteria_count["total"] > 0 else 0
        remor_u_score += review_record[criteria]
        
    # calculate meteor score
    review_record["meteor_score"] = calculate_meteor_score(review, review_record["full_text"])
    remor_u_score += review_record["meteor_score"]
    
    review_record["remor_u"] = remor_u_score
    review_record["figure_correspondence"] = calculate_figure_correspondence(review, review_record["figure_details"])
    review_record["novelty_correspondence"] = calculate_novelty_correspondence(review, review_record["novelty_assessment"])
       
    return review_record
    
    # title
    # abstract
    # major_discipline
    # minor_discipline
    # source
    # year
    # full_text
    # pdf_file_path
    # figure_details
    # novelty_assessment
    # prompt
    # full_text_length
    # list_of_reference
    # prompt_length
    # generated_reviews
    ############ metrics
    # criticism
    # example
    # importance and relevance
    # materials and methods
    # praise
    # presentation and reporting
    # results and discussion
    # suggestion and solution
    # meteor score
    # remor_u
    # figure correspondence
    # novelty correspondence





def main():
    data_file_dir = Path("generated_reviews")
    result_dir = Path("review_with_metrics")
    # review_files = [entry for entry in data_file_dir.glob("*.parquet")]
    review_files = [
        # data_file_dir / "mamorx_reviews.parquet",
        # data_file_dir / "remorx_ep4_reviews_full_option.parquet",
        # data_file_dir / "remorx_ep4_reviews_fig.parquet",
        # data_file_dir / "remorx_ep4_reviews_novel.parquet",
        # data_file_dir / "remorx_ep4_reviews_none.parquet",
        # data_file_dir / "remorx-agg-ep2_reviews_full_option.parquet",
        # data_file_dir / "remorx-agg-ep2_reviews_fig.parquet",
        # data_file_dir / "remorx-agg-ep2_reviews_novel.parquet",
        # data_file_dir / "remorx-agg-ep2_reviews_none.parquet",
        # data_file_dir / "q3_reviews_full_option.parquet",
        # data_file_dir / "q3_reviews_fig.parquet",
        # data_file_dir / "q3_reviews_novel.parquet",
        # data_file_dir / "q3_reviews_none.parquet",
        # data_file_dir / "remor_reviews_full_option.parquet",
        # data_file_dir / "remor_reviews_fig.parquet",
        # data_file_dir / "remor_reviews_novel.parquet",
        # data_file_dir / "remor_reviews_none.parquet",
        # data_file_dir / "remorx-agg-ep3_reviews_full_option.parquet",
        # data_file_dir / "remorx-agg-ep3_reviews_fig.parquet",
        # data_file_dir / "remorx-agg-ep3_reviews_novel.parquet",
        # data_file_dir / "remorx-agg-ep3_reviews_none.parquet",
        # data_file_dir / "remorx-agg-ep4_reviews_full_option.parquet",
        # data_file_dir / "remorx-agg-ep4_reviews_fig.parquet",
        # data_file_dir / "remorx-agg-ep4_reviews_novel.parquet",
        # data_file_dir / "remorx-agg-ep4_reviews_none.parquet",
        # data_file_dir / "remorx-agg-ep5_reviews_full_option.parquet",
        # data_file_dir / "remorx-agg-ep5_reviews_fig.parquet",
        # data_file_dir / "remorx-agg-ep5_reviews_novel.parquet",
        # data_file_dir / "remorx-agg-ep5_reviews_none.parquet",
        # data_file_dir / "remorx-agg-ep6_reviews_full_option.parquet",
        # data_file_dir / "remorx-agg-ep6_reviews_fig.parquet",
        # data_file_dir / "remorx-agg-ep6_reviews_novel.parquet",
        # data_file_dir / "remorx-agg-ep6_reviews_none.parquet",
        # data_file_dir / "remorx-agg-ep7_reviews_full_option.parquet",
        # data_file_dir / "remorx-agg-ep7_reviews_fig.parquet",
        # data_file_dir / "remorx-agg-ep7_reviews_novel.parquet",
        # data_file_dir / "remorx-agg-ep7_reviews_none.parquet",
        # data_file_dir / "remor_tpr_reviews_full_option.parquet",
        # data_file_dir / "remor_tpr_reviews_fig.parquet",
        # data_file_dir / "remor_tpr_reviews_novel.parquet",
        # data_file_dir / "remor_tpr_reviews_none.parquet",
        data_file_dir / "barebones.parquet",
        data_file_dir / "liangetal.parquet",
        data_file_dir / "mamorx_full.parquet",
        data_file_dir / "marg.parquet"
    ]
    
    """
    For each data file
        load data file
        convert data file to list of dict
        calculate remor metrics
        calculate meteor
        calculate remor_u
        calculate figure correspondence
        calculate novelty correspondence
    """
    for review_file in review_files:
        review_list = pd.read_parquet(review_file, engine="pyarrow").to_dict(orient="records")
        processed_review_list = [process_review_record(record, include_thinking_trace=True) for record in tqdm(review_list)]
        
        pd.DataFrame(processed_review_list).to_parquet(f"{result_dir}/{review_file.name}", engine="pyarrow", index=False)



if __name__ == "__main__":
    main()
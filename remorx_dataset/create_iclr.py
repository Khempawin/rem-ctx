import argparse
import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from tqdm import tqdm
from grobid_client.grobid_client import GrobidClient

from .utils import add_details_from_pdf, Article


def get_pdf_details_for_iclr(
    client: GrobidClient,
    metadata_list: list[Article],
    tokenizer: BertTokenizerFast,
    pdf_base_dir: str,
):
    # Process each TPR article
    processed_records = [add_details_from_pdf(record, client, tokenizer, pdf_base_dir, "ICLR") for record in tqdm(metadata_list)]
    
    return pd.DataFrame(processed_records) 


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract details from ICLR PDFs"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-pdf-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    
    grobid_config_path = args.config
    input_pdf_dir_path = args.input_pdf_dir
    output_data_path = args.output

    # Create client for Grobid
    client = GrobidClient(config_path=grobid_config_path)

    # List all ICLR pdf files
    iclr_pdf_dir = Path(input_pdf_dir_path)
    iclr_records = [Article(
        title="",
        abstract=None,
        major_discipline=None,
        minor_discipline=None,
        source="ICLR",
        year=None,
        full_text=None,
        pdf_file_path=str(entry),
        figure_details=None,
        novelty_assessment=None,
        prompt=None,
        full_text_length=None
        ) for entry in iclr_pdf_dir.glob("*.pdf")]
    
    # load tokenizer (same for all models handled here)
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    
    # Process articles
    processed_articles = get_pdf_details_for_iclr(
        client=client,
        metadata_list=iclr_records,
        tokenizer=tokenizer,
        pdf_base_dir=""
    )
    
    # Save records to parquet file
    processed_articles.to_parquet(output_data_path, engine="pyarrow", index=False)


if __name__ == "__main__":
    main()
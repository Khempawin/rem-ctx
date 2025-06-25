import argparse
import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from tqdm import tqdm
from grobid_client.grobid_client import GrobidClient
from typing import Literal, Optional

from .utils import add_details_from_pdf, Article


def get_pdf_details_from_file(
    client: GrobidClient,
    metadata_list: list[Article],
    tokenizer: BertTokenizerFast,
    pdf_base_dir: str,
    source: Literal["TPR", "ICLR", "ACL", "NeurIPS"],
    year: Optional[int]=None
):
    # Process each article
    processed_records = [add_details_from_pdf(
        record, client, tokenizer, pdf_base_dir, 
        source=source,
        year=year
        ) for record in tqdm(metadata_list)]
    
    return pd.DataFrame(processed_records) 


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract details from PDFs directory"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-pdf-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--source", type=str, required=False, default=None)
    parser.add_argument("--year", type=int, required=False, default=None)
    
    args = parser.parse_args()
    
    grobid_config_path = args.config
    input_pdf_dir_path = args.input_pdf_dir
    output_data_path = args.output
    source = args.source
    year = args.year

    # Create client for Grobid
    client = GrobidClient(config_path=grobid_config_path)

    # List all PDF files
    pdf_dir = Path(input_pdf_dir_path)
    records = [Article(
        title="",
        abstract=None,
        major_discipline=None,
        minor_discipline=None,
        source=source,
        year=year,
        full_text=None,
        pdf_file_path=str(entry),
        figure_details=None,
        novelty_assessment=None,
        prompt=None,
        full_text_length=None
        ) for entry in pdf_dir.glob("*.pdf")]
    
    # load tokenizer (same for all models handled here)
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    
    # Process articles
    processed_articles = get_pdf_details_from_file(
        client=client,
        metadata_list=records,
        tokenizer=tokenizer,
        pdf_base_dir="",
        source=source,
        year=year
    )
    
    # Save records to parquet file
    processed_articles.to_parquet(output_data_path, engine="pyarrow", index=False)


if __name__ == "__main__":
    main()
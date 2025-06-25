import argparse
import pandas as pd

from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from tqdm import tqdm
from grobid_client.grobid_client import GrobidClient
from typing import Literal, Optional

from .utils import add_details_from_pdf


def get_pdf_details_for_metadata(
    client: GrobidClient,
    metadata_df: pd.DataFrame,
    tokenizer: BertTokenizerFast,
    pdf_base_dir: str,
    source: Literal["TPR", "ICLR", "ACL", "NeurIPS"],
    year: Optional[int]=None
) -> pd.DataFrame:

    # Process each TPR article
    processed_records = [add_details_from_pdf(
        row, client, tokenizer, pdf_base_dir, 
        source=source,
        year=year
        ) for _, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0])]
    
    return pd.DataFrame(processed_records) 
    

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract add details for metadata records from PDFs"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-metadata", type=str, required=True)
    parser.add_argument("--input-pdf-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--source", type=str, required=False, default=None)
    parser.add_argument("--year", type=int, required=False, default=None)
    
    args = parser.parse_args()
    
    grobid_config_path = args.config
    input_metadata_path = args.input_metadata
    input_pdf_dir_path = args.input_pdf_dir
    output_data_path = args.output
    source = args.source
    year = args.year

    # Create client for Grobid
    client = GrobidClient(config_path=grobid_config_path)

    # Load metadata file for TPR
    tpr_df = pd.read_parquet(input_metadata_path, engine="pyarrow")
    
    # load tokenizer (same for all models handled here)
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    
    # Process articles
    processed_articles = get_pdf_details_for_metadata(
        client=client,
        metadata_df=tpr_df,
        tokenizer=tokenizer,
        pdf_base_dir=input_pdf_dir_path,
        source=source,
        year=year
    )
    
    # Save records to parquet file
    processed_articles.to_parquet(output_data_path, engine="pyarrow", index=False)


if __name__ == "__main__":
    main()
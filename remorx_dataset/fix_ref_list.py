import pandas as pd

from pathlib import Path
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from grobid_client.grobid_client import GrobidClient
from tqdm import tqdm

from .utils import add_details_from_pdf



def main():
    grobid_config_path = "config.json"
    manifest_dir = Path("data/figure_details_added")
    pdf_base_dir = "data"
    output_dir = Path("data/ref_list_added")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List all metadata files in input directory
    metadata_files = [entry for entry in manifest_dir.glob("*.parquet")]
    metadata_dfs = [pd.read_parquet(str(entry), engine="pyarrow") for entry in metadata_files]
    
    # Create client for Grobid
    client = GrobidClient(config_path=grobid_config_path)
    
    # load tokenizer (same for all models handled here)
    tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased')
    
    for filename, df in zip(metadata_files, metadata_dfs):
        save_path = f"{output_dir}/{filename.name}"
        processed_articles = [add_details_from_pdf(
            row, client, tokenizer, pdf_base_dir, 
            source=None,
            year=None
            ) for row in tqdm(df.to_dict(orient="records"))]
        pd.DataFrame(processed_articles).to_parquet(save_path, engine="pyarrow", index=False)
    


if __name__ == "__main__":
    main()
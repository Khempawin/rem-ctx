import pandas as pd
import base64

from pathlib import Path
from typing import TypedDict
from tqdm import tqdm
from anthropic import AnthropicBedrock
from ratelimit import limits, sleep_and_retry

from .utils import Article


client = AnthropicBedrock(
    aws_access_key="__ACCESS_KEY__",
    aws_secret_key="__SECRET_KEY__",
    aws_region="us-west-2",
)


class Dataset(TypedDict):
    data_path: Path
    dataframe: pd.DataFrame


@sleep_and_retry
@limits(calls=1, period=1)
def add_figure_details_text(record: Article) -> Article:
    
    pdf_file_path = "data/{}".format(record["pdf_file_path"])
    
    # Load PDF from local file system
    with open(pdf_file_path, "rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
    try:
        message = client.messages.create(
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
            max_tokens=4096,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_data
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please extract the figures from the PDF along with their paired captions. Create a detailed description for each image as well. It should be detailed so that a blind person can imagine the figure. Format the results in the form of a JSON file where each record represents a figure. Each record should have the following properties Figure number+name, detailed description, and related caption."
                        }
                    ]
                }
            ],
        )
        
        record["figure_details"] = message.content[0].text
    except Exception as e:
        record["figure_details"] = "ERROR"
    
    return record


def main():
    # List all metadata files
    # data_path = Path("data/cleaned")
    data_path = Path("data/processed")
    data_files = [entry for entry in data_path.glob("*.parquet")]
    output_path = Path("data/figure_details_added")
    output_path.mkdir(parents=True, exist_ok=True)
    
    loaded_data = [
        Dataset(
            data_path=entry,
            dataframe=pd.read_parquet(entry, engine="pyarrow")
        ) 
        for entry in data_files
    ]
    
    for ds in loaded_data:
        print(ds["data_path"], ds["dataframe"].shape, ds["data_path"].name)
        
        # Transform dataframe to list
        records: list[Article] = ds["dataframe"].to_dict(orient="records")
        
        # Add figure descriptions
        processed_records: list[Article] = [add_figure_details_text(record) for record in tqdm(records)]
        
        save_path = "{}/{}".format(output_path, ds["data_path"].name)
        print(f"\t{save_path}")
        pd.DataFrame(processed_records).to_parquet(save_path, engine="pyarrow", index=False)
        

if __name__ == "__main__":
    main()
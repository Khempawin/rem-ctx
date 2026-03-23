import pandas as pd

from typing import TypedDict, Optional, Literal
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from tqdm import tqdm
from pathlib import Path

from grobid_client.grobid_client import GrobidClient
from doc2json.grobid2json.tei_to_json import convert_tei_xml_soup_to_s2orc_json


class Article(TypedDict):
    title: str
    abstract: Optional[str]
    major_discipline: Optional[str]
    minor_discipline: Optional[str]
    source: Literal["TPR", "ICLR", "ACL", "NeurIPS"]
    year: Optional[int]
    full_text: Optional[str]
    pdf_file_path: str
    figure_details: Optional[str]
    novelty_assessment: Optional[str]
    prompt: Optional[str]
    full_text_length: Optional[int]


class PDFArticleContent(TypedDict):
    paper_id: str
    title: str
    abstract: str
    organized_text: str
    list_of_reference: list[str]


def extract_organized_text(json_data: str) -> PDFArticleContent:
    organized_text = ""
    seen_sections = set()
    list_of_reference = []

    # Ensure we're working with the correct structure
    pdf_parse = json_data.get('pdf_parse', json_data)

    # Extract paper ID
    paper_id = json_data.get('paper_id', 'No paper ID found')


    # Extract title by accessing the 'title' key in the JSON data
    title = None
    for key in ['title', 'pdf_parse.title']:
        try:
            temp = json_data
            for k in key.split('.'):
                temp = temp[k]
            title = temp
            break
        except (KeyError,TypeError):
            continue

    organized_text += f"Title: {title or 'No title found'}\n\n"
            

    # Extract abstract
    abstract = None
    for key in ['abstract', 'pdf_parse.abstract.text']:
        try:
            temp = json_data
            for k in key.split('.'):
                temp = temp[k]
            if isinstance(temp, list) and temp and 'text'in temp[0]:
                abstract = temp[0]['text']
            elif isinstance(temp, str):
                abstract = temp
            break
        except (KeyError,TypeError):
            continue
    organized_text += f"Abstract: {abstract or 'No abstract found'}\n\n"

    # Extract body text
    if 'body_text' in pdf_parse:
        for body_item in pdf_parse['body_text']:
            section = body_item.get('section', 'Unnamed Section')
            sec_num = body_item.get('sec_num')
            
            if section not in seen_sections:
                seen_sections.add(section)
                if sec_num:
                    organized_text += f"{sec_num}. {section}:\n\n"
                else:
                    organized_text += f"{section}:\n\n"
            
            organized_text += body_item['text'] + "\n\n"

    # Extract a list of references titles in strings
    if 'bib_entries' in pdf_parse:
        for bibref in pdf_parse['bib_entries']:
            try :
                list_of_reference.append(pdf_parse['bib_entries'][bibref]['title'])
            except (KeyError, TypeError):
                continue



    # Extract figures and tables
    if 'ref_entries' in pdf_parse:
        organized_text += "Figures and Tables:\n\n"
        for ref_key, ref_value in pdf_parse['ref_entries'].items():
            if ref_value['type_str'] in ['figure', 'table']:
                organized_text += f"{ref_value['text']}\n\n"
                if ref_value['type_str'] == 'table' and 'content' in ref_value:
                    organized_text += f"Table content: {ref_value['content']}\n\n"

    return organized_text.strip(), paper_id, title, abstract, list_of_reference


def parse_article_pdf_file(input_file_path: str, client: GrobidClient):
    
    source_path, status_code, tei_xml = client.process_pdf(
        "processFulltextDocument",
        input_file_path,
        tei_coordinates=True, 
        generateIDs=False,
        consolidate_header=False,
        consolidate_citations=False,
        include_raw_citations=True,
        include_raw_affiliations=False,
        segment_sentences=False
    )
    
    tei_soup = BeautifulSoup(tei_xml, "xml")

    paper_id = source_path.split('/')[-1].split('.')[0]

    pdf_hash = ""

    paper = convert_tei_xml_soup_to_s2orc_json(tei_soup, paper_id, pdf_hash)

    paper_json = paper.release_json()
    
    # Extract information from paper
    organized_text, paper_id, title, abstract, list_of_reference = extract_organized_text(paper_json)
    
    return PDFArticleContent(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        organized_text=organized_text,
        list_of_reference=list_of_reference
    )


def get_tokenized_length(message: str, tokenizer: BertTokenizerFast):
    return len(tokenizer(message)["input_ids"])


def process_record(record: Article, client: GrobidClient, tokenizer: BertTokenizerFast) -> Article:
    # Process PDF to get full_text
    try:
        pdf_article_content = parse_article_pdf_file(record["pdf_file_path"], client)
    except Exception as e:
        print("Error processing: {}".format(record["pdf_file_path"]))
        return record

    # Add title if missing
    if not record["title"]:
        record["title"] = pdf_article_content["title"]

    record["abstract"] = pdf_article_content["abstract"]
    record["full_text"] = pdf_article_content["organized_text"]
    
    # Get full_text_length
    record['full_text_length'] = get_tokenized_length(record["full_text"], tokenizer)
    
    return record


def main():
    # Create client for Grobid
    client = GrobidClient(config_path="./config.json")
    
    # List all ICLR pdf files
    iclr_pdf_dir = Path("iclr_pdfs")
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
    
    # Process each ICLR article
    processed_records = [process_record(record, client, tokenizer) for record in tqdm(iclr_records)]
    
    # Save records to parquet file
    pd.DataFrame(processed_records).to_parquet("iclr_processed.parquet", engine="pyarrow", index=False)


if __name__ == "__main__":
    main()

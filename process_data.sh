#!/bin/bash

# TPR dataset
# Create TPR
python -m remorx_dataset.create_from_metadata \
    --input-metadata data/tpr_metadata.parquet \
    --input-pdf-dir data \
    --output data/processed/processed_tpr_metadata.parquet\
    --config config.json

# Select TPR records to be used


# ICLR dataset
# Get Article names for each PDF files
python -m remorx_dataset.create_iclr \
    --input-pdf-dir data/iclr_pdfs \
    --output data/processed/pdf_iclr.parquet \
    --config config.json

# Get other metadata from existing dataset from huggingface
# Merge the metadata with the PDF data manually

# Select Data records to be used

# ACL 2017 dataset
# Get Article names for each PDF files
python -m remorx_dataset.create_from_pdf_dir \
    --config config.json \
    --input-pdf-dir data/acl_2017_pdfs \
    --output data/processed/acl2017_processed.parquet \
    --source ACL \
    --year 2017
    
# ACL 2024 dataset
# Get Article names for each PDF files
python -m remorx_dataset.create_from_pdf_dir \
    --config config.json \
    --input-pdf-dir data/acl_2024_pdfs \
    --output data/processed/acl2024_processed.parquet \
    --source ACL \
    --year 2024

# NeurIPS 2019 dataset
# Get Article names for each PDF files
python -m remorx_dataset.create_from_pdf_dir \
    --config config.json \
    --input-pdf-dir data/neurips_2019_pdfs \
    --output data/processed/neurips_2019_processed.parquet \
    --source NeurIPS \
    --year 2019

# Get Auxiliary Information for each dataset
# Get Figure details for each dataset
python -m remorx_dataset.get_figure_details

# Fix missing metadata for each dataset
python -m remorx_dataset.fix_ref_list

# Add novelty assessment for each dataset
python -m remorx_dataset.get_novelty_assessment

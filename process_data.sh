#!/bin/bash

# TPR dataset
# Create TPR
python -m remorx_dataset.create_tpr \
    --input-metadata data/tpr_metadata.parquet \
    --input-pdf-dir data \
    --output data/test_out/processed_tpr_metadata.parquet\
    --config config.json

# Select TPR records to be used


# ICLR dataset
# Get Article names for each PDF files
python -m remorx_dataset.create_iclr \
    --input-pdf-dir data/iclr_pdfs \
    --output data/test_out/pdf_iclr.parquet \
    --config config.json

# Get other metadata from existing dataset from huggingface

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
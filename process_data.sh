#!/bin/bash

# TPR dataset
# Create TPR
# python -m remorx_dataset.create_tpr \
#     --input-metadata data/tpr_metadata.parquet \
#     --input-pdf-dir data \
#     --output data/processed_tpr_metadata.parquet\
#     --config config.json

# Select TPR records to be used


# ICLR dataset
# Get Article names for each PDF files
# python -m remorx_dataset.create_iclr \
#     --input-pdf-dir data/iclr_pdfs \
#     --output data/pdf_iclr.parquet \
#     --config config.json

# Get other metadata from existing dataset from huggingface

# Select Data records to be used
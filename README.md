# REMORX

# Setup compute environment

## Vanilla TRL
```
conda create -n remorx python=3.12 
conda activate remorx

pip install uv

uv pip install vllm --torch-backend=auto
uv pip install -U nltk
uv pip install deepspeed
uv pip install trl
uv pip install git+https://github.com/huggingface/transformers.git@main
uv pip install git+https://github.com/huggingface/accelerate.git@main
uv pip install wandb
```

## Unsloth
```
conda create -n remorx-unsloth python=3.12 
conda activate remorx-unsloth

pip install uv

uv pip install vllm --torch-backend=auto
uv pip install -U nltk
uv pip install trl
uv pip install unsloth
uv pip install datasets
uv pip install wandb
```
# REMORX

# Setup compute environment
```
conda create -n remorx python=3.12 
conda activate remorx

pip install uv

uv pip install vllm --torch-backend=auto
uv pip install trl
uv pip install unsloth
```

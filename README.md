# Install

First, create a uv venv with the necessary packages:
```
uv venv
uv pip install diffusers transformers accelerate torch
```

Check the cuda version:
```
nvcc --version
```

Then install the corresponding version of xformers (here for cuda 12.8, adapt by changing "128"):
```
uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
```

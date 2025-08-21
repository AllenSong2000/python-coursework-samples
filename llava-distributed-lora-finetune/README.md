# LLaVA Fine-Tuning Prototype

This project demonstrates **fine-tuning LLaVA (Large Language and Vision Assistant)** using **LoRA/QLoRA** techniques on instruction-following datasets (e.g., LLaVA-Instruct-Mix).  
It includes both a Jupyter Notebook for step-by-step exploration and a distributed training script (`DDP.py`) for multi-GPU setups.

## Features
- **Notebook demo**: single-GPU fine-tuning with LoRA/QLoRA, data preprocessing, and inference examples.  
- **Distributed script (DDP.py)**: PyTorch Distributed Data Parallel (DDP) for scalable multi-GPU training.  
- **Data pipeline**: custom collator for aligning image-text pairs into LLaVA’s input format.  
- **Parameter-efficient training**: 4-bit quantization + LoRA for reduced memory footprint.  
- **Evaluation**: simple inference examples and hooks for VQA-style assessment.

## Project Structure
```
notebook/
└─ LLaVA_notebook.ipynb # Step-by-step prototype fine-tuning
scripts/
└─ DDP.py # Distributed multi-GPU fine-tuning script
data/
└─ samples/ # Placeholder for images/annotations
outputs/ # Checkpoints and logs (not included in repo)
```


## Quick Start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```
### 2) Run the notebook

Open notebook/LLaVA_notebook.ipynb in JupyterLab / Colab to try fine-tuning on a small sample.

### 3) Multi-GPU training with DDP
```bash
torchrun --nproc_per_node=4 scripts/DDP.py
```
Adjust --nproc_per_node based on your available GPUs.
(Training hyperparameters are defined directly inside DDP.py for simplicity — no extra config file required.)
## 4) Inference

After training, load the adapter and run:
```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

model = AutoModelForCausalLM.from_pretrained("outputs/checkpoints/last", device_map="auto")
processor = AutoProcessor.from_pretrained("outputs/checkpoints/last")

image = Image.open("data/samples/placeholder.jpg").convert("RGB")
inputs = processor(text="Describe the image briefly.", images=image, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=128)
print(processor.tokenizer.decode(output[0], skip_special_tokens=True))
```
## Notes

The real dataset is not included due to size restrictions. Replace data/samples/ with your own dataset (e.g., LLaVA-Instruct-Mix).

Notebook is suitable for teaching / assignment-scale experiments.

DDP.py is intended for scalable fine-tuning with multiple GPUs.

## Author

Yihang Song
Master of IT (Data Science), UNSW



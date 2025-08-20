# LLaVA Distributed Fine-tuning with LoRA

This project demonstrates **multi-GPU fine-tuning of LLaVA-1.5-7B** using:

- PyTorch **DistributedDataParallel (DDP)**
- **LoRA** (parameter-efficient fine-tuning)
- **BitsAndBytes 4-bit quantization**
- Hugging Face **Transformers** + **TRL**

> ⚠️ Note: This repo is shared as a **coursework-style Python project**.  
> It may require a specific CUDA/NCCL setup. Even if it cannot be run directly,  
> the code shows how to structure distributed training with custom collators.

---

## Quick Start

```bash
pip install -r requirements.txt

# Example paths (replace with your own)
MODEL_ID="liuhaotian/llava-v1.5-7b"
TRAIN_DATA="./data/train"   # HuggingFace datasets 'load_from_disk' folder
EVAL_DATA="./data/eval"

python -m torch.distributed.run --nproc_per_node=1 train_llava_distributed.py \
  --model_id "$MODEL_ID" \
  --train_dataset_path "$TRAIN_DATA" \
  --eval_dataset_path "$EVAL_DATA" \
  --output_dir "./output" \
  --tensorboard_log_dir "./runs" \
  --lr 1.4e-5 \
  --epochs 1 \
  --per_device_train_batch_size 8 \
  --grad_accum 2


# train_llava_distributed.py
# Clean, argparse-driven script for fine-tuning LLaVA with LoRA + 4-bit.
# Launch with torchrun, e.g.:
# python -m torch.distributed.run --nproc_per_node=1 train_llava_distributed.py \
#   --model_id liuhaotian/llava-v1.5-7b \
#   --train_dataset_path ./data/train \
#   --eval_dataset_path ./data/eval

import argparse
from typing import List, Dict, Any

import torch
from datasets import load_from_disk
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    TrainingArguments,
    set_seed,
)


class LLavaDataCollator:
    """
    Collator for chat-style multimodal samples:
    each example should have:
      - "messages": list of chat messages (HF chat template compatible)
      - "images":   list of images (take first image per example)
    """

    def __init__(self, processor: AutoProcessor):
        self.processor = processor
        # If model uses a special image token id, we can optionally mask it from labels.
        # Try to resolve; if not present, keep None.
        try:
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            self.image_token_id = None

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts, images = [], []
        for ex in examples:
            text = self.processor.tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            # assume first image per sample
            images.append(ex["images"][0])

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        # Build labels from input_ids and mask pads (and optionally image token) to -100
        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        if self.image_token_id is not None and self.image_token_id != -1:
            labels[labels == self.image_token_id] = -100
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune LLaVA with LoRA (4-bit) using HF Trainer.")
    p.add_argument("--model_id", required=True, help="Base model id or local path (e.g. liuhaotian/llava-v1.5-7b)")
    p.add_argument("--train_dataset_path", required=True, help="Path to HF dataset (load_from_disk) for training")
    p.add_argument("--eval_dataset_path", required=True, help="Path to HF dataset (load_from_disk) for eval")
    p.add_argument("--output_dir", default="./output", help="Output dir for checkpoints")
    p.add_argument("--tensorboard_log_dir", default="./runs", help="TensorBoard log dir")
    p.add_argument("--lr", type=float, default=1.4e-5, help="Learning rate")
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    p.add_argument("--per_device_train_batch_size", type=int, default=8, help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Tokenizer & Processor
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    # Ensure pad_token exists for proper loss masking
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer = tokenizer

    # Model (4-bit quantization via bitsandbytes); let HF Trainer handle device placement/DDP.
    quant = BitsAndBytesConfig(load_in_4bit=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant,
        torch_dtype="auto",
    )

    # Data
    train_ds = load_from_disk(args.train_dataset_path)
    eval_ds = load_from_disk(args.eval_dataset_path)
    data_collator = LLavaDataCollator(processor)

    # Training args (HF Trainer will auto-use DDP when launched with torchrun)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to="tensorboard",
        logging_dir=args.tensorboard_log_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        logging_steps=5,
        gradient_checkpointing=True,
        remove_unused_columns=False,  # keep image columns
        fp16=True,
        bf16=False,
        do_train=True,
        do_eval=True,
    )

    # LoRA (use a broad target; swap to explicit list if your transformers/llava build requires)
    lora = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_kwargs={"skip_prepare_dataset": True},  # we provide our own collator
    )

    trainer.train()


if __name__ == "__main__":
    # HF Trainer reads LOCAL_RANK etc. from torchrun, so no manual init_process_group needed.
    main()

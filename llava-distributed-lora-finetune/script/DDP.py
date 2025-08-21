import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from torch.multiprocessing import spawn
from torch.utils.data import DataLoader, DistributedSampler

class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["images"][0])

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 尝试初始化进程组
    try:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)

    except Exception as e:
        print(f"Error initializing the distributed environment: {e}")

def cleanup():
    # 销毁进程组
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"Error during cleanup: {e}")

def train(rank, world_size):
    setup(rank, world_size)
    
    model_id = "/data01/Data/yhluo/syh/llava-1.5-7b-hf"
    train_dataset_path = "/data01/Data/yhluo/syh/LLAVA-Finetune/lava-instruct-mix-vsft-train"
    eval_dataset_path = "/data01/Data/yhluo/syh/LLAVA-Finetune/lava-instruct-mix-vsft-test"
    output_dir = "/data01/Data/yhluo/syh/LLAVA-Finetune/output"
    tensorboard_log_dir = "/data01/Data/yhluo/syh/LLAVA-Finetune/tensorboard"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )

    model2 = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map = {"": 1},
    )

    model2 = DDP(model2, device_ids=[rank])

    LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. \
                            The assistant gives helpful, detailed, and polite answers to the user's questions. \
                            {% for message in messages %}{% if message['role'] == 'user' %}\
                            USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}\
                            {% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer = tokenizer

    data_collator = LLavaDataCollator(processor)

    train_dataset = load_from_disk(train_dataset_path)
    eval_dataset = load_from_disk(eval_dataset_path)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, collate_fn=data_collator)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, collate_fn=data_collator)

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="tensorboard",
        learning_rate=1.4e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=5,
        num_train_epochs=1,
        push_to_hub=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        fp16=True,
        bf16=False,
        do_train=True,
        do_eval=True
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules="all-linear"
    )

    trainer = SFTTrainer(
        model=model2.module,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",  # need a dummy field
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer.train()
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    
    spawn(train, nprocs=world_size, args=(world_size,))

if __name__ == "__main__":
    main()

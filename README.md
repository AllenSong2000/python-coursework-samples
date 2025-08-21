# Yihang Song – Project Portfolio

This repository collects two representative projects showcasing my experience in **data engineering, distributed systems, and AI model fine-tuning**.  
Both projects were implemented as part of coursework and research exploration, and demonstrate my ability to work with **PySpark**, **multimodal LLMs**, and **scalable training pipelines**.

---

## 📂 Projects Overview

### 1. LLaVA Fine-Tuning Prototype
- **Goal**: Fine-tune **LLaVA (Large Language and Vision Assistant)** with LoRA/QLoRA for instruction-following multimodal tasks.  
- **Highlights**:
  - Jupyter notebook demo for single-GPU fine-tuning & inference.
  - `DDP.py` script for **multi-GPU distributed training** with PyTorch DDP.
  - Custom data pipeline to align image–text pairs.
  - Parameter-efficient training (4-bit quantization + LoRA).  
- **Folder**: [`llava-distributed-lora-finetune/`](./llava-distributed-lora-finetune)  
- **Tech stack**: PyTorch, Hugging Face Transformers, LoRA/QLoRA, DDP, Jupyter

---

### 2. PySpark Transaction Similarity Join
- **Goal**: Implement a scalable **similarity join** over e-commerce transaction logs using **Jaccard similarity** and prefix filtering.  
- **Highlights**:
  - Transaction pairs with similarity ≥ τ.
  - Cross-year filtering and correct output formatting.
  - Efficient Spark DataFrame operations with UDFs and partitioning.  
- **Folder**: [`pyspark-similarity-join/`](./pyspark-similarity-join)  
- **Tech stack**: PySpark, DataFrames, Jaccard similarity, distributed data processing

---

## 🔧 How to Use
Each project has its own README with setup instructions, datasets, and run examples:
- [LLaVA Fine-Tuning README](./llava-distributed-lora-finetune/README.md)  
- [PySpark Similarity Join README](./pyspark-similarity-join/README.md)

---

## 👤 Author
**Yihang Song**  
Master of IT (Data Science), UNSW  


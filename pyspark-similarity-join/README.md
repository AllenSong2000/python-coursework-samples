# PySpark Transaction Similarity Join

This project implements a **similarity join** over e-commerce transaction logs using **Jaccard similarity** and **prefix filtering** in **PySpark**.  
The goal is to find all transaction pairs with similarity ≥ τ (tau).

## Features
- Transaction = set of purchased items
- **Cross-year pairs only**; pairs are unique with `id1 < id2`
- Output format: `(InvoiceNo1,InvoiceNo2):similarity`, sorted by IDs
- Scales to large datasets using Spark DataFrame APIs and UDFs

## Project Structure
project3.py # main PySpark implementation
data/ # sample & test datasets
expected/ # example outputs
output/ # Spark output folder (created at runtime)
logs/ # runtime logs (optional)
docs/ # assignment brief/context

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with Spark (example: τ = 0.5)
spark-submit project3.py data/sample.csv output 0.5

```
**Result:** Spark writes text files under `output/` (e.g., `part-00000...txt`) with lines like:
(1,3):0.75
(2,3):0.75

## Technical Highlights
- **Algorithm**: Jaccard similarity + prefix filtering
- **Optimizations**: broadcast item ordering, Kryo serialization, caching, partitioning
- **Framework**: PySpark (DataFrame API, UDFs, window functions)

## Datasets
- `data/sample.csv` – minimal runnable example
- `data/testcase1.csv`, `data/testcase2.csv` – validation sets
- Expected outputs in `expected/`

## Author
**Yihang Song**  
Master of IT (Data Science), UNSW

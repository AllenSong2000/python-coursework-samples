# PySpark Transaction Similarity Join (Jaccard + Prefix Filtering)

This project finds **similar transaction pairs across different years** using
**Jaccard similarity** with **prefix filtering** in PySpark.

> Note: Coursework-style sample for demonstrating Python + PySpark skills.
> It includes a tiny sample so you can run locally without a cluster.

## Quick Start
```bash
pip install -r requirements.txt

# Local run (Spark will use local[*])
spark-submit project3.py sample_data/mini.csv output 0.5
# Output will be written as text files under ./output

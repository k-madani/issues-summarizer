# GitHub Issue Summarizer

Fine-tuning Flan-T5-base for automatic GitHub issue summarization.

## Overview

Fine-tuned Flan-T5-base (248M parameters) to automatically summarize GitHub issues, achieving **100.5% improvement** over baseline (ROUGE-1: 14.26 → 28.59).

**Key Results:**
- Test ROUGE-1: 25.82
- Inference: 306ms per summary
- Time savings: 97% (50 issues in 0.3 min vs 2.5 hours)

## Quick Start

```
pip install -r requirements.txt
```

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("./config_3")
tokenizer = AutoTokenizer.from_pretrained("./config_3")

def summarize(text):
    inputs = tokenizer(f"summarize: {text}", max_length=512, 
                      truncation=True, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=64, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
issue = "Application crashes with OutOfMemoryError..."
print(summarize(issue))
```

## Results

| Configuration | LR | ROUGE-1 | Loss | Time |
|--------------|-----|---------|------|------|
| Baseline | - | 14.26 | - | - |
| Config 1 | 5e-5 | 26.14 | 2.99 | 6.4 min |
| Config 2 | 3e-5 | 24.25 | 3.01 | 6.4 min |
| **Config 3** | **1e-4** | **28.59** | **2.95** | **8.3 min** |
| Test | 1e-4 | 25.82 | - | - |

**Improvement:** +14.33 points (+100.5%)

## Technical Details

**Model:** Flan-T5-base (248M parameters)  
**Dataset:** 3,866 train / 967 val / 500 test samples  
**Training:** Manual PyTorch loop with dynamic padding  
**Optimization:** AdamW + linear warmup + gradient clipping  
**Hardware:** Google Colab Tesla T4 GPU

## Examples

**Input:** "Application crashes with OutOfMemoryError when processing large CSV files over 100MB..."  
**Output:** "OutOfMemoryError when processing large CSV files over 100MB"  
**Compression:** 86%

## Project Structure

```
notebooks/
├── 01_dataset_exploration.ipynb
├── 02_data_preprocessing.ipynb
└── 03_model_training_and_evaluation.ipynb

results/
├── baseline_results.json
├── test_results.json
└── training_summary.json

docs/
└── technical_report.pdf
```

## Reproducibility

1. Run `01_dataset_exploration.ipynb`
2. Run `02_data_preprocessing.ipynb`
3. Run `03_model_training_and_evaluation.ipynb`

All notebooks run on Google Colab with Tesla T4 GPU.

## Acknowledgments

- Dataset: [mlfoundations-dev/github-issues](https://huggingface.co/datasets/mlfoundations-dev/github-issues)
- Base Model: [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)

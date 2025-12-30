# Transformer Translation (DE-EN)

A high-performance German-to-English translation model based on the **Transformer-Base** architecture, optimized for efficiency and stability.

## Key Features
- **BPE Tokenization:** Handles complex vocabulary and compound words via Byte-Pair Encoding.
- **Modern Architecture:** Implements Pre-Layer Normalization and GELU activations for superior training stability.
- **Hardware Optimized:** Utilizes **FlashAttention**, **Automatic Mixed Precision (AMP)**, and a pre-tokenized pipeline to maximize GPU throughput.
- **Advanced Scheduling:** Features **OneCycleLR** for faster and more reliable convergence.


## Project Structure
- `data.py`: BPE tokenization, data cleaning, and pre-tokenized Dataset/DataLoader.
- `config.yaml`: Configuration for hyperparameters and training.
- `model.py`: Transformer architecture.
- `train.py`: Training logic.


## Quick Start
1. **Install dependencies:**
```bash
pip install torch datasets tokenizers pyyaml matplotlib
```


2. **Training:**
```bash
python train.py
```
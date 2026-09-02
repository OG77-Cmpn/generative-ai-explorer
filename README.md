# generative-ai-explorer
# Generative AI Exploration with DistilGPT-2

This repository contains a simple Python application demonstrating text generation using Hugging Face's open-source `distilgpt2` model.

## Features
- **Model**: Lightweight Causal Language Model (`distilgpt2`) running locally via PyTorch and Hugging Face Transformers.
- **Task**: Uses temperature-based sampling and top-p/top-k filtering to complete user-provided text prompts.

## Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch transformers
# Vietnamese Text Summarization

A comprehensive research project evaluating and implementing multiple models for Vietnamese text summarization, including Extractive (Lead-3, TextRank), Abstractive Seq2Seq (BARTpho), Causal LLMs (Qwen2.5), and custom Transformer architectures.

## Project Overview

This repository contains the complete pipeline for dataset preparation, model training, evaluation, and a full-stack web demonstration for Vietnamese news summarization.

### Key Features
- **Extensive Benchmarking**: Evaluated traditional models, fine-tuned BARTpho (FFT and LoRA), Qwen2.5 (LoRA), and Custom Transformer/Mamba architectures on the VietNews dataset.
- **Custom Architecture**: Features an \'Entity Guided Pure Transformer\' trained from scratch, achieving competitive performance.
- **LLM-as-a-Judge Evaluation**: Integrated GPT-4o for qualitative evaluation across Relevance, Coherence, Consistency, and Fluency.
- **Interactive Web Demo**: A modern React-based frontend and FastAPI backend allowing side-by-side model comparisons on arbitrary URLs or pre-computed samples.

## Repository Structure

- \data/\: Dataset samples and preparation scripts.
- \demo/\: Source code for the web demonstration.
  - \ackend/\: FastAPI backend handling model inference and web scraping.
  - \rontend/\: React + Vite frontend with Tailwind CSS.
- \docs/\: Project documentation, including the final evaluation results (\EVALUATION_RESULTS.md\).
- \models/\: Directory for storing pre-trained weights and fine-tuned checkpoints (ignored in git).
- otebooks/\: Jupyter notebooks covering EDA, training (FFT, LoRA), evaluation, and decoding strategies.
- \paper/\: LaTeX source code for the research paper.
- esults/\: Aggregated evaluation datasets and model predictions.
- \scripts/\: Utility scripts for processing and evaluation.

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+

### Running the Web Demo

1. **Backend Setup**
\\ash
cd demo/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
\
2. **Frontend Setup**
\\ash
cd demo/frontend
npm install
npm run dev
\The application will be available at http://localhost:5173.

## Evaluation Results
For comprehensive metrics, please refer to [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md).

## Acknowledgments
- VietNews Dataset
- Hugging Face Transformers & PEFT
- VinAI (BARTpho)
- Qwen Team (Qwen2.5)

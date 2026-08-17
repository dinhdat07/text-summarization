# Tổng hợp Kết quả Đánh giá Mô hình (Evaluation Results)

## 1. Đánh giá trên toàn bộ tập Test (1,000 bài báo)
Mục tiêu: So sánh năng lực tóm tắt tổng thể giữa các phương pháp.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-LSUM | BLEU | BERTScore - F1 |
|--------|---------|---------|---------|------------|------|----------------|
| Lead-3 | 45.75 | 21.87 | 28.99 | 28.98 | 8.79 | 72.69 |
| TextRank | 36.88 | 18.99 | 24.94 | 24.93 | 6.37 | 71.52 |
| LoRA - BARTpho | 50.17 | 14.93 | 30.40 | 30.41 | 2.34 | 66.29 |
| Full-Finetuning - BARTpho | 50.24 | 15.27 | 30.52 | 30.51 | 2.25 | 66.46 |
| LoRA - Qwen2.5 | 47.00 | 20.24 | 29.99 | 30.31 | 8.77 | 70.34 |

## 2. Đánh giá trên tập bài báo Dài (Long Documents > 800 từ)
Mục tiêu: Đánh giá khả năng xử lý ngữ cảnh dài (Long-Context) và hiện tượng Truncation.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-LSUM | BLEU | BERTScore - F1 |
|--------|---------|---------|---------|------------|------|----------------|
| FFT - BARTpho (Normal - 1024 tokens) | 47.79 | 12.05 | 26.94 | 26.93 | 1.22 | 63.64 |
| FFT - BARTpho (Sliding Window) | 47.05 | 12.49 | 26.16 | 26.16 | 0.64 | 62.98 |
| LoRA - Qwen2.5 (Native 2048 tokens) | 45.62 | 17.11 | 27.27 | 27.35 | 6.14 | 68.94 |

LLM Judge
               model    id  Relevance  Coherence  Consistency  Fluency
0  bartpho_full_beam  49.5       2.80       3.55         4.06     2.73
1  bartpho_lora_beam  49.5       2.73       3.48         3.89     2.67
2     qwen_lora_beam  49.5       2.44       2.22         3.39     2.73

--- Kết quả cho Qwen2.5 (Sau Post-Process) ---
ROUGE-1: 47.85
ROUGE-2: 20.42
ROUGE-L: 30.22
ROUGE-Lsum: 30.25
BLEU-4: 9.19

--- Kết quả cho Qwen2.5 Long Context (Sau Post-Process) ---
ROUGE-1: 45.38
ROUGE-2: 16.81
ROUGE-L: 27.06
ROUGE-Lsum: 27.08
BLEU-4: 6.27
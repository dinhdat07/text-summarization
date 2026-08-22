# Tổng hợp Kết quả Đánh giá Mô hình (Evaluation Results)

Tài liệu này trình bày các chỉ số đánh giá chất lượng tóm tắt văn bản tiếng Việt sử dụng các mô hình và phương pháp khác nhau trên tập dữ liệu VietNews.

## 1. Đánh giá trên toàn bộ tập Test (1,000 bài báo)

**Mục tiêu:** So sánh năng lực tóm tắt tổng thể giữa các phương pháp tiếp cận (Extractive và Abstractive) trên tập Test chung.

| Phương pháp | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BLEU-4 | BERTScore (F1) |
|-------------|---------|---------|---------|------------|--------|----------------|
| Lead-3 | 45.75 | 21.87 | 28.99 | 28.98 | 8.79 | 72.69 |
| TextRank | 36.88 | 18.99 | 24.94 | 24.93 | 6.37 | 71.52 |
| BARTpho (Full Fine-tuning) | 50.24 | 15.27 | 30.52 | 30.51 | 2.25 | 66.46 |
| BARTpho (LoRA) | 50.17 | 14.93 | 30.40 | 30.41 | 2.34 | 66.29 |
| Qwen2.5-0.5B (LoRA)* | 47.85 | 20.42 | 30.22 | 30.25 | 9.19 | 70.34 |

*(Lưu ý: Chỉ số của Qwen2.5 là kết quả sau khi đã áp dụng Post-Process để làm sạch prompt)*

## 2. Đánh giá trên tập bài báo Dài (Long Documents > 800 từ)

**Mục tiêu:** Đánh giá khả năng xử lý ngữ cảnh dài (Long-Context) và tác động của hiện tượng cắt xén (Truncation) đối với các mô hình seq2seq giới hạn 1024 token so với các mô hình Causal LLM.

| Phương pháp | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BLEU-4 | BERTScore (F1) |
|-------------|---------|---------|---------|------------|--------|----------------|
| BARTpho FFT (Normal - 1024 tokens) | 47.79 | 12.05 | 26.94 | 26.93 | 1.22 | 63.64 |
| BARTpho FFT (Sliding Window) | 47.05 | 12.49 | 26.16 | 26.16 | 0.64 | 62.98 |
| Qwen2.5-0.5B LoRA (Native 2048 tokens)* | 45.38 | 16.81 | 27.06 | 27.08 | 6.27 | 68.94 |

*(Lưu ý: Chỉ số của Qwen2.5 là kết quả sau khi đã áp dụng Post-Process để làm sạch prompt)*

## 3. Đánh giá của LLM Judge (GPT-4o)

**Mục tiêu:** Đánh giá định tính các khía cạnh về nội dung mà các độ đo tự động (ROUGE/BLEU) không thể phản ánh chính xác, thông qua thang điểm 1-5.

| Mô hình | Relevance (Độ liên quan) | Coherence (Độ mạch lạc) | Consistency (Độ nhất quán) | Fluency (Độ trôi chảy) |
|---------|:---:|:---:|:---:|:---:|
| **BARTpho (Full Fine-tuning)** | 2.80 | 3.55 | 4.06 | 2.73 |
| **BARTpho (LoRA)** | 2.73 | 3.48 | 3.89 | 2.67 |
| **Qwen2.5-0.5B (LoRA)** | 2.44 | 2.22 | 3.39 | 2.73 |
## 4. Đánh giá mô hình Transformer (Train from scratch)

**Mục tiêu:** Đánh giá các biến thể kiến trúc Transformer và Mamba được huấn luyện từ đầu (train from scratch) trên tập dữ liệu.

| Mô hình | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore (F1) |
|---------|---------|---------|---------|------|----------------|
| Baseline Transformer | 44.61 | 11.06 | 27.10 | 1.41 | 70.01 |
| Improved Baseline | 48.52 | 11.78 | 27.93 | 1.70 | 70.12 |
| Soft Prompt Mamba | 45.92 | 10.23 | 27.03 | 1.28 | 69.89 |
| Entity Gated Mamba | 47.44 | 11.24 | 28.31 | 1.20 | 69.38 |
| Transformer Hypersphere | 48.04 | 13.18 | 29.01 | 2.25 | 70.43 |
| Entity Guided Hybrid Mamba | 49.17 | 13.40 | 29.16 | 2.21 | 70.85 |
| **Entity Guided Pure Transformer** | **50.00** | **14.01** | **29.74** | **2.57** | **70.88** |

*(Lưu ý: Các chỉ số trên là kết quả tốt nhất khi áp dụng penalty trong quá trình giải mã)*

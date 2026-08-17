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
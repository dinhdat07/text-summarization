# Vietnamese Abstractive Text Summarization
## Project Proposal (Draft v1.0)

> Đây là bản đề xuất dự án cho Bài tập lớn môn NLP.
>
> Mục tiêu: xây dựng một bài nghiên cứu (mini research paper) thay vì chỉ fine-tune một mô hình có sẵn.
>
> Thời gian: **2 tuần**
>
> Nhân lực: **3 thành viên + AI Agents**
>
> Hạ tầng: **Google Colab / Kaggle**

---

# 1. Motivation

Text Summarization là một trong những bài toán quan trọng nhất của NLP hiện đại.

Trong vài năm gần đây, các mô hình pre-trained như BART, T5, BARTpho hay Qwen đã giúp chất lượng tóm tắt tăng đáng kể. Tuy nhiên, vẫn còn nhiều câu hỏi thực tế:

- Có nhất thiết phải Full Fine-tune?
- LoRA có đủ tốt không?
- Decoding Strategy ảnh hưởng bao nhiêu?
- Văn bản dài nên xử lý như thế nào?
- Có thể tối ưu chất lượng mà vẫn tiết kiệm tài nguyên?

Đây sẽ là trọng tâm của dự án.

---

# 2. Research Questions

## RQ1

How do traditional Seq2Seq architectures (BARTpho) compare to modern Decoder-only LLMs (Qwen) for Vietnamese abstractive text summarization?

---

## RQ2

Can parameter-efficient fine-tuning (LoRA) achieve comparable performance to Full Fine-tuning?

---

## RQ3

How much do decoding strategies affect summary quality?

---

## RQ4

How do chunking-based post-hoc methods (Hierarchical Sliding Window with BARTpho) compare to native long-context processing in modern LLMs (Qwen2.5) for long documents?

---

# 3. Objectives

Xây dựng một benchmark đầy đủ cho Vietnamese Text Summarization bao gồm:

- Traditional baseline
- Neural baseline
- Modern PLM
- Parameter Efficient Fine-tuning
- Decoding Analysis
- Long Document Strategy
- Error Analysis

---

# 4. Scope

## Không làm

Để đảm bảo hoàn thành trong 2 tuần, nhóm sẽ KHÔNG nghiên cứu:

- Transformer from scratch
- Mamba
- RLHF
- DPO
- Reward Model
- NER Guidance
- Reinforcement Learning

Các nội dung trên rất thú vị nhưng vượt quá phạm vi BTL.

---

## Tập trung

- Benchmark
- Fine-tuning
- LoRA
- Decoding
- Sliding Window
- Evaluation
- Error Analysis

---

# 5. Overall Pipeline

                    Dataset
                       │
             Data Cleaning + EDA
                       │
        ┌──────────────┴──────────────┐
        │                             │
     Lead-3                     TextRank
        │                             │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
  BARTpho Baseline             Qwen2.5-0.5B Baseline
        │                             │
  Full Fine-tuning        Parameter Efficient (LoRA)
        │                             │
  BARTpho + LoRA               Qwen2.5-0.5B + LoRA
        │                             │
      ┌────────────────┼────────────────┐
      │                │                │
 Decode Strategy   Sliding Window   Length Control
      │                │                │
      └────────────────┼────────────────┘
                       │
              Comprehensive Evaluation
           (ROUGE, BERTScore, LLM-as-a-Judge)
                       │
         Error Analysis + Visualization
                       │
                     Demo
```

---

# 6. Dataset

Sử dụng dataset: **`nam194/vietnews`** (từ Hugging Face Hub).

Tiêu chí lựa chọn:

- Là tập dữ liệu chuẩn, phổ biến trong nhiều bài báo tóm tắt tiếng Việt.
- Phù hợp cho bài toán tóm tắt văn bản tin tức (News Summarization).
- Cấu trúc dữ liệu rõ ràng, dễ dàng tiền xử lý.

**Thiết lập tập dữ liệu (Dataset Split):**
Nhằm tối ưu phần cứng (Kaggle GPU T4 16GB) và đảm bảo thời gian huấn luyện Full Fine-Tuning không vượt quá giới hạn 12h/session, nhóm sẽ trích xuất một subset ngẫu nhiên (cố định seed) như sau:
- **Train set:** 10,000 mẫu
- **Validation set:** 1,000 mẫu
- **Test set:** 1,000 mẫu

**Cấu hình Huấn luyện (Training Configuration):**
- **Epochs:** 4 epochs (Thời gian ước tính 4-6 tiếng trên T4, chừa khoảng trống an toàn cho Evaluation và lưu Checkpoint).
- Toàn bộ các thực nghiệm (Baseline, Full FT, LoRA, Decoding) đều bắt buộc chạy trên đúng subset này để đảm bảo tính công bằng (fairness) khi so sánh.

---

# 7. Data Preprocessing

Bao gồm

- Unicode normalization
- Remove duplicated samples
- Sentence segmentation
- Remove invalid records
- Length filtering
- **Data-centric Filtering (Anti-Lead Bias):** Loại bỏ các bài báo mà tóm tắt (abstract) trùng lặp quá 80% với 3 câu đầu tiên (sapo) để ép mô hình học cách đọc hiểu toàn cục.
- Statistics

EDA gồm:

- Number of documents
- Average length
- Summary length
- Vocabulary
- Token distribution

---

# 8. Baseline Models

## Baseline 1

Lead-3

Lấy 3 câu đầu làm summary.

Mục đích:

Traditional baseline.

---

## Baseline 2

TextRank

Graph-based Extractive Summarization.

Mục đích:

So sánh Extractive vs Abstractive.

---

## Baseline 3

BARTpho

Fine-tune theo cấu hình chuẩn.

Đây sẽ là Neural Baseline.

---

# 9. Main Models

## BARTpho (Full FT & LoRA)

Lý do:
- State-of-the-art (Seq2Seq) cho tiếng Việt
- Tiết kiệm VRAM
- Huấn luyện nhanh

---

## Qwen2.5-0.5B (LoRA)

Lý do:
- Đại diện cho dòng Decoder-only LLM hiện đại.
- Size 0.5B tương đương BARTpho (0.4B), tạo ra fair comparison.
- Cho phép đánh giá khả năng Generation mạnh mẽ của kiến trúc LLM so với Seq2Seq.

---

# 10. Research Direction 1

## Parameter Efficient Fine-tuning

So sánh

| Full FT | LoRA |
|----------|------|
| ROUGE | |
| BERTScore | |
| GPU Memory | |
| Training Time | |
| Parameters | |

Mục tiêu:

Liệu LoRA có thể đạt gần Full Fine-tuning?

---

# 11. Research Direction 2

## Decoding Strategy

Benchmark

- Beam Search
- Diverse Beam Search
- Top-k Sampling
- Top-p Sampling

So sánh

- ROUGE
- BERTScore
- Summary Length
- Repetition
- Fluency

---

# 12. Research Direction 3

Dựa trên kết quả EDA cho thấy nhiều bài báo có độ dài từ 1000-2000 tokens (vượt quá max_length của mô hình), kỹ thuật này được đưa thành một thử nghiệm chính nhằm xử lý Long Document Summarization.

## Long-Context Native vs Hierarchical Sliding Window

Benchmark

- BARTpho: No Sliding (Truncation at 1024)
- BARTpho: Hierarchical Sliding Window (Map-Reduce logic: Tóm tắt từng chunk 512, sau đó nối lại và tóm tắt lần 2).
- Qwen2.5-0.5B: Native Long-Context (max_length=2048, Gradient Checkpointing).

Mục tiêu

Đánh giá tác động của Sliding Window đến chất lượng tóm tắt các văn bản vượt quá Context Window của mô hình Seq2Seq, và so sánh sức mạnh xử lý nguyên bản của kiến trúc LLM (RoPE).

---

# 13. Optional Research Direction 2

Nếu còn thời gian.

## Length Control

Benchmark

- 30 words
- 50 words
- 80 words
- 120 words

Quan sát

ROUGE

vs

Summary Length

---

# 14. Evaluation

## Automatic Metrics

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore

---

## LLM-as-a-Judge (Prompt-based Evaluation)

Sử dụng LLM mạnh (Gemini / GPT-4) qua API để chấm điểm (thang 1-5).

Tiêu chí:

- Relevance (Sự liên quan)
- Coherence (Tính mạch lạc)
- Consistency (Tính nhất quán / Faithfulness)
- Fluency (Tính trôi chảy)

---

# 15. Error Analysis

Đây là phần trọng tâm.

Xây dựng Error Taxonomy.

Ví dụ

- Hallucination
- Missing Entity
- Missing Important Information
- Repetition
- Wrong Numbers
- Wrong Dates
- Too Short
- Too Long

Thống kê tỷ lệ từng loại lỗi.

---

# 16. Visualization

Bao gồm

- Training Loss
- ROUGE Comparison
- BERTScore
- Summary Length Distribution
- Attention Visualization (nếu khả thi)

---

# 17. Ablation Study

Đây là phần quan trọng nhất của báo cáo.

Ví dụ

| Model | LoRA | Sliding | Decode |
|---------|------|----------|---------|
| Baseline | ❌ | ❌ | Beam |
| +LoRA | ✅ | ❌ | Beam |
| +Sliding | ✅ | ✅ | Beam |
| +Top-p | ✅ | ✅ | Top-p |

Qua đó đánh giá đóng góp của từng thành phần.

---

# 18. Demo

Xây dựng giao diện đơn giản bằng

Gradio

hoặc

Streamlit.

Input

↓

Article

↓

Summary

↓

Metric

↓

Model Comparison

---

# 19. Team Assignment

## Member A

Literature Review

Dataset

EDA

Baselines

Report

---

## Member B

Fine-tuning

LoRA

Training

Evaluation

---

## Member C

Visualization

Error Analysis

Demo

Presentation

---

AI Agents hỗ trợ

- đọc paper
- sinh baseline code
- giải thích paper
- review notebook
- sinh biểu đồ
- hỗ trợ viết báo cáo

---

# 20. Expected Contributions

Dự án không hướng tới phát minh thuật toán mới.

Đóng góp chính là:

1. Benchmark nhiều baseline trên cùng dataset.

2. So sánh kiến trúc Seq2Seq (BARTpho) và Decoder-only LLM (Qwen).

3. So sánh Full Fine-tuning và LoRA.

4. Đánh giá ảnh hưởng của Decoding Strategy.

5. Triển khai kiến trúc Hierarchical Sliding Window (Map-Reduce) và so sánh với Native Long-Context (Qwen2.5).

6. Đánh giá chất lượng sinh bằng phương pháp hiện đại LLM-as-a-Judge.

7. Làm sạch dữ liệu theo hướng Data-Centric (chống Lead-Bias).

8. Xây dựng Error Taxonomy cho Vietnamese Summarization.

9. Phân tích định lượng và định tính thay vì chỉ báo cáo ROUGE.

---

# 21. Expected Deliverables

- Source Code
- Trained Models
- Report (25–40 pages)
- Slide Presentation
- Demo Application
- GitHub Repository
- Experiment Logs
- Evaluation Results

---

# 22. Why This Project?

So với đồ án tham khảo, dự án này:

✅ Đơn giản hơn.

✅ Tập trung hơn.

✅ Có câu hỏi nghiên cứu rõ ràng.

✅ Có benchmark đầy đủ.

✅ Có ablation study.

✅ Có error analysis.

✅ Có demo.

Quan trọng nhất, toàn bộ các thí nghiệm đều trả lời cùng một nhóm câu hỏi nghiên cứu thay vì thử quá nhiều kỹ thuật độc lập. Điều này giúp báo cáo có tính học thuật cao hơn, dễ trình bày hơn và phù hợp với quy mô của một bài tập lớn trong 2 tuần.
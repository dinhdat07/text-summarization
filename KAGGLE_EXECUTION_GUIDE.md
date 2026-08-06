# Hướng dẫn thực thi dự án trên Kaggle (Execution Guide)

Tài liệu này cung cấp sơ đồ phụ thuộc (Dependency Graph) và hướng dẫn chi tiết cách chạy 7 Notebook của dự án trên nền tảng Kaggle để đảm bảo không gặp lỗi thiếu file hoặc quá tải tài nguyên.

---

## 1. Sơ đồ Luồng dữ liệu (Data Flow)

Để chạy thành công, các Notebook bắt buộc phải nhận đúng Output Data từ các Notebook trước đó.

```text
       [ Hugging Face: nam194/vietnews ]
                     │
                     ▼
                 [ BƯỚC 1 ]
   (Tạo ra: train_10k.csv, val_1k.csv, test_1k.csv)
                     │
      ┌──────────────┴───────────────┬────────────────┐
      ▼              ▼               ▼                ▼
   [NB 2]         [NB 3]          [NB 4]           [NB 6]
 (Baselines) (BARTpho Full) (BARTpho LoRA)   (Qwen LoRA)
      │              │               │                │
      │        (Model Full)    (Adapter)       (Adapter)
      │              └───────┬───────┴────────────────┘
      │                      ▼
      │                   [NB 5] (Đánh giá ROUGE & Sinh text)
      │                      │
      │         (Tạo ra: all_models_predictions.csv)
      │                      ▼
      └───────────────►   [NB 7] (LLM-as-a-Judge)
```

---

## 2. Kế hoạch chạy chi tiết

### 🔴 BƯỚC 1: Khởi tạo dữ liệu (Chạy đầu tiên & bắt buộc)
*   **Notebook:** `01_Data_Preparation_and_EDA.ipynb`
*   **Hardware:** CPU
*   **Input:** Tự động tải từ HuggingFace (không cần Add Data thủ công).
*   **Output:** Tạo ra 3 file CSV (`train_10k.csv`, `val_1k.csv`, `test_1k.csv`). 
*   **Hành động trên Kaggle:** Nhấn `Save Version` (chọn Save & Run All). Đợi Notebook hoàn thành, Output của Notebook này sẽ được dùng làm Dataset (thông qua nút Add Data) cho tất cả các Notebook sau.
*   **Thời gian ước tính:** 5-10 phút.

---

### 🟡 BƯỚC 2: Huấn luyện & Baseline (Có thể chạy SONG SONG)
Sau khi hoàn thành Bước 1, tạo 4 Notebook riêng biệt trên Kaggle. Trong mỗi Notebook, chọn **Add Data -> Your Work** -> Chọn Output của Bước 1. 

**Lưu ý:** Kaggle miễn phí cho phép chạy **tối đa 2 sessions GPU cùng lúc**. Bạn nên chạy song song NB3 và NB4 trước.

1.  **`02_Traditional_Baselines.ipynb`**
    *   **Hardware:** CPU
    *   **Input Data:** Cần file `test_1k.csv`.
    *   **Thời gian ước tính:** ~ 15 phút.
2.  **`03_Full_Finetuning_BARTpho.ipynb`**
    *   **Hardware:** Bắt buộc GPU T4 x2 (nếu có) hoặc T4 x1
    *   **Input Data:** Cần file `train_10k.csv`, `val_1k.csv`.
    *   **Output:** Trọng số mô hình Full.
    *   **Thời gian ước tính:** ~ 4 - 5 tiếng.
3.  **`04_LoRA_Finetuning_BARTpho.ipynb`**
    *   **Hardware:** Bắt buộc GPU T4
    *   **Input Data:** Cần file `train_10k.csv`, `val_1k.csv`.
    *   **Output:** Trọng số LoRA Adapter (vài chục MB).
    *   **Thời gian ước tính:** ~ 1.5 - 2 tiếng.
4.  **`06_LoRA_Finetuning_Qwen2.5.ipynb`**
    *   **Hardware:** Bắt buộc GPU T4
    *   **Input Data:** Cần file `train_10k.csv`, `val_1k.csv`.
    *   **Output:** Trọng số LoRA Adapter.
    *   **Thời gian ước tính:** ~ 2 tiếng.

> **💡 Mẹo:** Hãy sử dụng chế độ `Save Version -> Save & Run All (Commit)` khi train GPU. Máy chủ Kaggle sẽ chạy ngầm và tự lưu model cho dù bạn có tắt trình duyệt.

---

### 🟢 BƯỚC 3: Đánh giá và Xuất kết quả
Chỉ chạy sau khi Bước 1 và Bước 2 đã hoàn thiện và sinh ra các Models.
*   **Notebook:** `05_Evaluation_and_Decoding.ipynb`
*   **Hardware:** GPU T4
*   **Input Data (Add Data):** 
    1. Output của Bước 1 (lấy `test_1k.csv`).
    2. Output của NB 3 (lấy model full).
    3. Output của NB 4 (lấy lora adapter).
    4. Output của NB 6 (lấy qwen adapter).
*   **Output:** File `all_models_predictions.csv` (lưu kết quả sinh tóm tắt của 1000 bài báo để chấm điểm LLM sau này).
*   **Thời gian ước tính:** ~ 1 tiếng.

---

### 🔵 BƯỚC 4: Chấm điểm bằng AI (LLM Judge)
*   **Notebook:** `07_LLM_as_a_Judge.ipynb`
*   **Hardware:** CPU (Bắt buộc bật **Internet access** trong Settings của Kaggle).
*   **Input Data (Add Data):** File `all_models_predictions.csv` sinh ra từ Bước 3.
*   **Output:** Biểu đồ Radar Chart và file `llm_judge_scores.csv`.
*   **Thời gian ước tính:** ~ 10-15 phút (tuỳ thuộc vào tốc độ phản hồi của API).
*   **Yêu cầu:** Cần có API Key của Google Gemini hoặc Groq/OpenAI (nhập vào biến trong code).

---

## Tóm tắt quy trình làm việc nhóm
- **Thành viên A:** Phụ trách chạy Bước 1 và NB2.
- **Thành viên B:** Phụ trách cắm máy (Save & Run All) cho NB3 và NB4.
- **Thành viên C:** Phụ trách cắm máy NB6. Sau đó thu thập output để chạy NB5 và NB7.

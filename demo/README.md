# Vietnamese Text Summarization — Web Demo

Giao diện web demo so sánh 3 mô hình tóm tắt văn bản tiếng Việt:
- **BARTpho (Full Fine-Tuning)**: Seq2Seq, vinai/bartpho-syllable
- **BARTpho (LoRA)**: Seq2Seq, r=16, alpha=32
- **Qwen2.5-0.5B (LoRA)**: Decoder-only, r=16, alpha=16

## Yêu cầu hệ thống

- Python >= 3.9
- Node.js >= 18
- RAM >= 4GB (Sample Browser) hoặc >= 8GB (Live Inference)

## Cài đặt

### Backend

```bash
cd demo/backend
pip install -r requirements.txt
```

### Frontend

```bash
cd demo/frontend
npm install
```

## Chạy Demo

### 1. Khởi động Backend

```bash
cd demo/backend
python -m uvicorn app.main:app --port 8000
```

Backend sẽ tự động:
- Load dữ liệu pre-computed (1,000 mẫu) từ `results/all_models_predictions_postprocessed.csv`
- Load model checkpoints từ `models/` (nếu có)

### 2. Khởi động Frontend

```bash
cd demo/frontend
npm run dev
```

### 3. Mở trình duyệt

Truy cập: **http://localhost:5173**

## Chế độ hoạt động

### Sample Browser (mặc định, không cần GPU/model)

Duyệt qua 1,000 mẫu kiểm thử đã được tính toán sẵn. Click **"Mẫu ngẫu nhiên"** hoặc nhập số thứ tự (0-999) để xem kết quả của 3 mô hình.

### Live Inference (cần model checkpoints)

Đặt các checkpoint vào thư mục `models/`:
```
models/
├── bartpho_full_ft_final/
├── bartpho_lora_final/
└── qwen-lora-vietnews/
```

Sau đó paste URL bài báo hoặc nội dung text → click **"Tóm tắt"** để chạy inference realtime.

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| GET | `/health` | Health check |
| GET | `/api/models` | Danh sách models đã load |
| POST | `/api/scrape` | Crawl nội dung từ URL |
| GET | `/api/samples/total` | Tổng số mẫu |
| GET | `/api/samples/random` | Mẫu ngẫu nhiên |
| GET | `/api/samples/{index}` | Mẫu theo index |
| POST | `/api/summarize` | Chạy inference + metrics |
| POST | `/api/metrics` | Tính metrics từ predictions |

## Công nghệ

- **Backend**: FastAPI, Transformers, PEFT, evaluate
- **Frontend**: React 18, Vite, Vanilla CSS
- **Models**: BARTpho (vinai/bartpho-syllable), Qwen2.5 (Qwen/Qwen2.5-0.5B)

# Hệ thống Tóm tắt Văn bản Tiếng Việt (Vietnamese Text Summarization)

Một dự án nghiên cứu và phát triển hệ thống tóm tắt văn bản tự động dành riêng cho tiếng Việt. Dự án tập trung vào việc cài đặt, tinh chỉnh (fine-tuning) và đánh giá toàn diện nhiều kiến trúc mô hình học sâu khác nhau, từ các phương pháp truyền thống, mô hình Seq2Seq, mô hình ngôn ngữ lớn (LLM), cho đến các biến thể kiến trúc tự xây dựng (train from scratch).

## 1. Tổng quan Dự án

Dự án này cung cấp một pipeline hoàn chỉnh bao gồm các khâu: thu thập và xử lý dữ liệu, huấn luyện mô hình, đánh giá tự động, và một ứng dụng Web (Web Demo) trực quan để thử nghiệm.

### Các hướng tiếp cận đã thực nghiệm:
- **Phương pháp trích xuất (Extractive)**: Lead-3, TextRank.
- **Mô hình Seq2Seq (Abstractive)**: BARTpho (Sử dụng cả Full Fine-Tuning và PEFT/LoRA).
- **Mô hình Causal LLM**: Qwen2.5-0.5B-Instruct (Tinh chỉnh bằng LoRA).
- **Mô hình Custom**: Kiến trúc Entity Guided Pure Transformer và một số biến thể Mamba được huấn luyện hoàn toàn từ đầu.

### Các tính năng nổi bật:
- **Đánh giá đa chiều (Comprehensive Benchmarking)**: Đánh giá mô hình sử dụng các thang đo phổ biến (ROUGE, BLEU, BERTScore).
- **Đánh giá định tính bằng LLM (LLM-as-a-Judge)**: Tích hợp GPT-4o để đánh giá chất lượng tóm tắt dựa trên 4 tiêu chí: Độ liên quan (Relevance), Độ mạch lạc (Coherence), Độ nhất quán (Consistency), và Độ trôi chảy (Fluency).
- **Web Demo tương tác cao**: Giao diện ReactJS hiện đại kết hợp FastAPI backend. Cho phép người dùng tóm tắt thông qua việc dán URL trực tiếp (tự động cào dữ liệu) hoặc nhập văn bản tùy ý. Tính năng so sánh kết quả sinh văn bản của 3-4 mô hình song song cùng lúc.

## 2. Cấu trúc Repository

Dự án được tổ chức theo cấu trúc sau để đảm bảo tính rõ ràng và dễ bảo trì:

- \data/\: Chứa các script tiền xử lý và mẫu dữ liệu trích xuất từ tập VietNews.
- \demo/\: Mã nguồn của ứng dụng Web Demo.
  - \backend/\: API server viết bằng FastAPI, chịu trách nhiệm tải mô hình, thực hiện suy luận (inference), và cào dữ liệu báo chí (web scraping).
  - \frontend/\: Giao diện người dùng viết bằng React + Vite + Tailwind CSS.
- \docs/\: Chứa tài liệu dự án, nổi bật nhất là file \EVALUATION_RESULTS.md\ tổng hợp toàn bộ kết quả đánh giá mô hình.
- \models/\: Thư mục chứa các trọng số mô hình đã huấn luyện (được đưa vào .gitignore do dung lượng lớn).
- \notebooks/\: Các file Jupyter Notebook ghi lại toàn bộ quá trình nghiên cứu: Khám phá dữ liệu (EDA), Huấn luyện (FFT, LoRA), Đánh giá và Phân tích chiến lược giải mã (Decoding Strategies).
- \paper/\: Mã nguồn LaTeX của báo cáo khoa học/bài báo nghiên cứu.
- \results/\: Chứa các file CSV lưu trữ kết quả dự đoán của mô hình để phục vụ tính toán metrics và hiển thị mẫu trên Web Demo.
- \scripts/\: Các đoạn script tiện ích hỗ trợ phân tích và tự động hóa.

## 3. Hướng dẫn Cài đặt và Sử dụng

### Yêu cầu hệ thống
- Python 3.11+
- Node.js 18+
- RAM: Khuyến nghị tối thiểu 8GB để tải các mô hình cục bộ. (Có hỗ trợ chạy trên CPU/MPS/CUDA).

### Bước 1: Khởi động Backend
Backend đóng vai trò cung cấp API suy luận cho các mô hình và tự động cào bài báo.

\\ash
cd demo/backend
# Khuyến khích tạo môi trường ảo (virtual environment) trước khi cài đặt
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
\*(Backend sẽ khởi chạy tại \http://127.0.0.1:8000\)*

### Bước 2: Khởi động Frontend
Frontend là giao diện tương tác người dùng.

\\ash
cd demo/frontend
npm install
npm run dev
\*(Frontend sẽ khởi chạy tại \http://localhost:5173\)*

## 4. Kết quả Đánh giá
Các mô hình đều cho thấy hiệu năng tốt trên tập dữ liệu VietNews. Sự kết hợp giữa các kiến trúc tự chú ý (Self-Attention) và phương pháp giải mã có kiểm soát (penalty) đã giúp triệt tiêu hiện tượng lặp từ và cải thiện rõ rệt chất lượng văn bản sinh.

Để xem bảng tổng hợp kết quả chi tiết của tất cả các biến thể mô hình (ROUGE, BLEU, BERTScore, LLM Judge), vui lòng tham khảo tài liệu: **[docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md)**.

## 5. Lời cảm ơn và Ghi nhận
- Tập dữ liệu [VietNews](https://github.com/ThanhChinhBK/vietnews) cho tác vụ tóm tắt văn bản.
- Đội ngũ VinAI Research với mô hình pre-trained BARTpho.
- Đội ngũ Qwen (Alibaba Cloud) với dòng mô hình Qwen2.5.
- Thư viện Transformers và PEFT từ Hugging Face.

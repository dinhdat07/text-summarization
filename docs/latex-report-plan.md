# Research Report LaTeX Plan

## Goal
Viết báo cáo Nghiên cứu Khoa học bằng LaTeX đạt chuẩn học thuật quốc tế về chủ đề "Tóm tắt văn bản tiếng Việt" nhằm so sánh các phương pháp truyền thống, Seq2Seq (BARTpho), và mô hình Causal LLM thế hệ mới (Qwen2.5) với bối cảnh ngữ cảnh dài.

## Tasks
- [ ] Task 1: Khởi tạo Template LaTeX → Verify: Biên dịch thành công file PDF sườn báo cáo (đủ các phần Abstract, Introduction, Method, Result, Conclusion).
- [ ] Task 2: Viết phần Abstract & Introduction → Verify: Trình bày rõ bối cảnh bài toán, sự khó khăn của việc tóm tắt văn bản dài và đóng góp chính của nghiên cứu.
- [ ] Task 3: Viết phần Related Work → Verify: Có trích dẫn và review các kỹ thuật nền tảng: TextRank, Seq2Seq, cơ chế Attention, và xu hướng dùng LLM (LoRA).
- [ ] Task 4: Trình bày Methodology (Phương pháp) → Verify: Giải thích chi tiết các thiết lập Fine-tuning (Full vs LoRA), giải thuật Sliding Window (Map-Reduce) và Native Long-context.
- [ ] Task 5: Mô tả Experimental Setup (Thực nghiệm) → Verify: Cung cấp đủ thông số về Dataset (VietNews), config huấn luyện (lr, batch size, epochs) và các metrics (ROUGE, BLEU, BERTScore).
- [ ] Task 6: Phân tích Results & Discussion (Kết quả) → Verify: Chuyển bảng kết quả `evaluation_results.md` vào LaTeX, phân tích hiện tượng Lead-Bias và sự vượt trội của Qwen ở văn bản dài.
- [ ] Task 7: Viết Conclusion & Future Work → Verify: Đúc kết lại ưu điểm/nhược điểm của từng kỹ thuật và đề xuất hướng nghiên cứu tiếp theo.
- [ ] Task 8: Định dạng References (Trích dẫn) → Verify: File `.bib` chứa đầy đủ tài liệu tham khảo đúng chuẩn IEEE/APA và hiển thị chính xác trong PDF.

## Done When
- [ ] File mã nguồn `.tex` biên dịch trơn tru không báo lỗi (Zero errors).
- [ ] File PDF xuất ra có đầy đủ văn bản khoa học, bảng biểu (2 bảng so sánh kết quả), công thức nếu có, và danh mục trích dẫn chuẩn mực.
- [ ] Luận điểm khoa học rõ ràng: Biện luận xuất sắc được hiện tượng Lead-Bias của dữ liệu báo chí và nguyên nhân tụt điểm BLEU của phương pháp chắp vá Map-Reduce.

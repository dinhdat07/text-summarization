export default function Footer() {
  return (
    <footer className="footer">
      <p>
        <strong>Vietnamese Text Summarization</strong> — Built with React & FastAPI.
      </p>
      <p className="mt-1" style={{ opacity: 0.7 }}>
        So sánh hiệu năng các mô hình BARTpho và Qwen2.5 trên tập dữ liệu VietNews.
      </p>
    </footer>
  );
}

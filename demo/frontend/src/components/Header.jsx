import { Brain } from 'lucide-react';

export default function Header({ backendStatus, modelCount }) {
  return (
    <header className="header">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Brain size={28} color="var(--accent)" />
            <h1 className="header-title">Vietnamese Text Summarization</h1>
          </div>
          <p className="header-subtitle mt-1">
            So sánh BARTpho (Full Fine-Tuning / LoRA) và Qwen2.5 (LoRA) trên bài báo tiếng Việt
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`badge ${backendStatus === 'connected' ? 'badge-fft' : backendStatus === 'checking' ? 'badge-lora' : 'badge-qwen'}`}>
            {backendStatus === 'connected' ? '● Backend Online' : backendStatus === 'checking' ? '↻ Đang kiểm tra backend...' : '○ Backend Offline'}
          </span>
          {modelCount > 0 && (
            <span className="badge badge-lora">{modelCount} models loaded</span>
          )}
        </div>
      </div>
    </header>
  );
}

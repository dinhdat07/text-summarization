import { Brain, Sun, Moon } from 'lucide-react';

export default function Header({ backendStatus, modelCount, theme, onThemeToggle }) {
  return (
    <>
      <div className="chevron-decoration" style={{ left: 0, top: '20px' }}></div>
      <div className="chevron-decoration" style={{ right: 0, top: '20px' }}></div>
      <header className="header">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Brain size={32} color="var(--primary-bright)" />
              <h1 className="header-title">Vietnamese Text Summarization</h1>
            </div>
            <p className="header-subtitle mt-2">
              So sánh <strong>BARTpho (FFT / LoRA)</strong> và <strong>Qwen2.5 (LoRA)</strong> trên bài báo tiếng Việt
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <span className={`badge ${backendStatus === 'connected' ? 'badge-fft' : backendStatus === 'checking' ? 'badge-lora' : 'badge-qwen'}`}>
                {backendStatus === 'connected' ? '● Backend Online' : backendStatus === 'checking' ? '↻ Checking...' : '○ Offline'}
              </span>
              {modelCount > 0 && (
                <span className="badge badge-lora">{modelCount} models loaded</span>
              )}
            </div>
            <button
              onClick={onThemeToggle}
              className="btn-icon"
              title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            >
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>
        </div>
      </header>
    </>
  );
}

import { useState } from 'react';
import { Link, FileText, Send, Loader, Shuffle, Search, Database, RotateCcw } from 'lucide-react';

export default function InputSection({
  inputText,
  setInputText,
  reference,
  setReference,
  onScrape,
  onSummarize,
  onComputeMetrics,
  onSampleLoaded,
  onReset,
  loading,
  hasModels,
  hasPredictions,
}) {
  const [activeTab, setActiveTab] = useState('url');
  const [url, setUrl] = useState('');
  const [sampleIndex, setSampleIndex] = useState('');

  const handleScrape = async () => {
    if (url.trim()) {
      await onScrape(url.trim());
    }
  };

  return (
    <div className="card mb-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Database size={20} color="var(--primary-bright)" />
          <h2 className="text-primary" style={{ fontSize: '18px', fontWeight: 600 }}>Cấu hình đầu vào</h2>
        </div>
        {hasPredictions && (
          <button className="btn btn-secondary" onClick={onReset} disabled={loading} style={{ height: '36px', padding: '0 16px', fontSize: '13px' }}>
            <RotateCcw size={14} />
            Làm mới
          </button>
        )}
      </div>

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'url' ? 'active' : ''}`}
          onClick={() => setActiveTab('url')}
          disabled={loading}
        >
          <Link size={16} />
          <span>Trích xuất từ URL</span>
        </button>
        <button
          className={`tab ${activeTab === 'text' ? 'active' : ''}`}
          onClick={() => setActiveTab('text')}
          disabled={loading}
        >
          <FileText size={16} />
          <span>Nhập văn bản</span>
        </button>
        <button
          className={`tab ${activeTab === 'sample' ? 'active' : ''}`}
          onClick={() => setActiveTab('sample')}
          disabled={loading}
        >
          <Database size={16} />
          <span>Dữ liệu mẫu</span>
        </button>
      </div>

      <div className="mt-4">
        {/* URL Input */}
        {activeTab === 'url' && (
          <div className="flex gap-4">
            <input
              className="input"
              type="url"
              placeholder="Nhập link bài báo (VNExpress, Tuổi Trẻ, Dân Trí...)"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={loading}
              style={{ flex: 1 }}
            />
            <button className="btn" onClick={handleScrape} disabled={loading || !url.trim()}>
              {loading ? <Loader size={16} className="spin" /> : 'Trích Xuất'}
            </button>
          </div>
        )}

        {/* Text Input */}
        {activeTab === 'text' && (
          <textarea
            className="textarea"
            placeholder="Dán nội dung bài báo tiếng Việt vào đây..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={loading}
          />
        )}

        {/* Sample Input */}
        {activeTab === 'sample' && (
          <div className="flex items-center gap-4 p-4" style={{ background: 'var(--card-header-bg)', borderRadius: '8px', border: '1px solid var(--border-strong)' }}>
            <button
              className="btn btn-secondary"
              onClick={() => onSampleLoaded('random')}
              disabled={loading}
            >
              <Shuffle size={16} />
              <span>Lấy mẫu ngẫu nhiên</span>
            </button>
            <span className="text-muted text-sm">hoặc theo chỉ mục:</span>
            <div className="flex items-center gap-2">
              <input
                className="input"
                type="number"
                min="0"
                max="999"
                placeholder="ID (0-999)"
                value={sampleIndex}
                onChange={(e) => setSampleIndex(e.target.value)}
                disabled={loading}
                style={{ width: '120px' }}
              />
              <button
                className="btn btn-secondary"
                onClick={() => onSampleLoaded('index', parseInt(sampleIndex))}
                disabled={loading || sampleIndex === ''}
                style={{ padding: '0 16px' }}
              >
                <Search size={16} />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Reference Input */}
      <details className="mt-4">
        <summary className="text-sm text-secondary" style={{ cursor: 'pointer', padding: '8px 0' }}>
          Tóm tắt tham chiếu (tuỳ chọn — dành cho đánh giá Metrics)
        </summary>
        <textarea
          className="textarea mt-2"
          placeholder="Dán bản tóm tắt chuẩn (reference summary) để tính điểm ROUGE, BLEU, BERTScore..."
          value={reference}
          onChange={(e) => setReference(e.target.value)}
          disabled={loading}
          style={{ minHeight: '80px' }}
        />
      </details>

      {/* Action Buttons */}
      <div className="flex justify-between items-center mt-6 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
        <div className="flex gap-3">
          {hasModels && (
            <button
              className="btn"
              onClick={onSummarize}
              disabled={loading || !inputText.trim()}
            >
              {loading ? <Loader size={16} className="spin" /> : <Send size={16} />}
              <span>Tóm tắt & Đánh giá</span>
            </button>
          )}
          {hasPredictions && reference.trim() && (
            <button
              className="btn btn-secondary"
              onClick={onComputeMetrics}
              disabled={loading}
            >
              Tính Metrics
            </button>
          )}
        </div>
        {loading && <span className="text-sm text-primary flex items-center gap-2"><Loader size={14} className="spin"/> Đang xử lý...</span>}
      </div>
    </div>
  );
}

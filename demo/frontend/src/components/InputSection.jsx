import { useState } from 'react';
import { Link, FileText, Send, Loader } from 'lucide-react';

export default function InputSection({
  inputText,
  setInputText,
  reference,
  setReference,
  onScrape,
  onSummarize,
  onComputeMetrics,
  loading,
  hasModels,
  hasPredictions,
}) {
  const [activeTab, setActiveTab] = useState('url');
  const [url, setUrl] = useState('');

  const handleScrape = async () => {
    if (url.trim()) {
      await onScrape(url.trim());
    }
  };

  return (
    <div className="input-section card">
      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'url' ? 'active' : ''}`}
          onClick={() => setActiveTab('url')}
        >
          <Link size={16} />
          <span>Paste URL</span>
        </button>
        <button
          className={`tab ${activeTab === 'text' ? 'active' : ''}`}
          onClick={() => setActiveTab('text')}
        >
          <FileText size={16} />
          <span>Nhập nội dung</span>
        </button>
      </div>

      {/* URL Input */}
      {activeTab === 'url' && (
        <div className="flex gap-2 mt-2">
          <input
            className="input"
            type="url"
            placeholder="https://vnexpress.net/..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            style={{ flex: 1 }}
          />
          <button className="btn" onClick={handleScrape} disabled={loading || !url.trim()}>
            {loading ? <Loader size={16} className="spin" /> : 'Crawl'}
          </button>
        </div>
      )}

      {/* Text Input */}
      {activeTab === 'text' && (
        <textarea
          className="textarea mt-2"
          placeholder="Dán nội dung bài báo tiếng Việt vào đây..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          style={{ width: '100%' }}
        />
      )}

      {/* Reference Input */}
      <details className="mt-2">
        <summary className="text-sm text-muted" style={{ cursor: 'pointer' }}>
          Tóm tắt tham chiếu (tuỳ chọn — cần để tính metrics)
        </summary>
        <textarea
          className="textarea mt-1"
          placeholder="Dán bản tóm tắt tham chiếu (reference summary) để tính ROUGE, BLEU, BERTScore..."
          value={reference}
          onChange={(e) => setReference(e.target.value)}
          style={{ width: '100%', minHeight: '80px' }}
        />
      </details>

      {/* Action Buttons */}
      <div className="flex gap-2 mt-3">
        {hasModels && (
          <button
            className="btn"
            onClick={onSummarize}
            disabled={loading || !inputText.trim()}
          >
            {loading ? <Loader size={16} className="spin" /> : <Send size={16} />}
            <span>Tóm tắt (Live Inference)</span>
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
    </div>
  );
}

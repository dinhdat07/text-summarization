import { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import SampleBrowser from './components/SampleBrowser';
import InputSection from './components/InputSection';
import ResultsPanel from './components/ResultsPanel';
import {
  healthCheck,
  getAvailableModels,
  scrapeArticle,
  getRandomSample,
  getSample,
  summarize,
  computeMetrics,
} from './api';

export default function App() {
  // State
  const [backendStatus, setBackendStatus] = useState('checking');
  const [availableModels, setAvailableModels] = useState([]);
  const [inputText, setInputText] = useState('');
  const [reference, setReference] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Check backend on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        await healthCheck();
        setBackendStatus('connected');
        const { models } = await getAvailableModels();
        setAvailableModels(models || []);
      } catch {
        setBackendStatus('disconnected');
      }
    };
    checkBackend();
  }, []);

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Handle sample loading
  const handleSampleLoaded = useCallback(async (type, index) => {
    setLoading(true);
    setError(null);
    try {
      const sample = type === 'random' ? await getRandomSample() : await getSample(index);
      setInputText(sample.article);
      setReference(sample.reference);
      setPredictions(sample.predictions);
      setMetrics(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Handle URL scraping
  const handleScrape = useCallback(async (url) => {
    setLoading(true);
    setError(null);
    try {
      const result = await scrapeArticle(url);
      setInputText(result.text);
      setPredictions(null);
      setMetrics(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Handle live inference
  const handleSummarize = useCallback(async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = await summarize(inputText, reference);
      setPredictions(result.predictions);
      setMetrics(result.metrics || null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [inputText, reference]);

  // Handle metrics computation
  const handleComputeMetrics = useCallback(async () => {
    if (!predictions || !reference.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = await computeMetrics(predictions, reference);
      setMetrics(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [predictions, reference]);

  return (
    <div className="container">
      <Header
        backendStatus={backendStatus}
        modelCount={availableModels.length}
      />

      <div className="mt-3">
        {backendStatus === 'disconnected' && (
          <div className="card mb-4" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', borderColor: 'var(--error)' }}>
            <p style={{ color: '#fca5a5' }}>
              <strong>⚠️ Backend Unreachable:</strong> Demo đang chạy ở chế độ Frontend-only.
              Hãy đảm bảo đã khởi động backend tại <code>localhost:8000</code>.
            </p>
          </div>
        )}
        <SampleBrowser onSampleLoaded={handleSampleLoaded} loading={loading} />
      </div>

      <div className="mt-3">
        <InputSection
          inputText={inputText}
          setInputText={setInputText}
          reference={reference}
          setReference={setReference}
          onScrape={handleScrape}
          onSummarize={handleSummarize}
          onComputeMetrics={handleComputeMetrics}
          loading={loading}
          hasModels={availableModels.length > 0}
          hasPredictions={!!predictions}
        />
      </div>

      {/* Article Preview */}
      {inputText && (
        <div className="card mt-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm" style={{ textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>Nội dung bài báo</h3>
            <span className="badge">{inputText.split(/\s+/).length} từ</span>
          </div>
          <p className="summary-text mt-2" style={{ color: 'var(--text-secondary)', maxHeight: '200px', overflow: 'auto' }}>
            {inputText}
          </p>
        </div>
      )}

      {/* Reference Preview */}
      {reference && (
        <div className="card mt-2">
          <h3 className="text-sm" style={{ textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>Tóm tắt tham chiếu</h3>
          <p className="summary-text mt-1" style={{ color: 'var(--success)' }}>{reference}</p>
        </div>
      )}

      <ResultsPanel
        predictions={predictions}
        metrics={metrics}
        loading={loading}
      />

      {/* Error Toast */}
      {error && (
        <div className="error-toast">{error}</div>
      )}
    </div>
  );
}

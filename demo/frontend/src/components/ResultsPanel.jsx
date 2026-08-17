import ModelCard from './ModelCard';

const MODEL_ORDER = ['bartpho_fft', 'bartpho_lora', 'qwen_lora'];

export default function ResultsPanel({ predictions, metrics, loading }) {
  if (!predictions && !loading) return null;

  return (
    <div style={{ animation: 'fadeIn 0.3s ease' }}>
      <h2 className="text-sm text-muted mt-4" style={{ marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Kết quả tóm tắt
      </h2>
      <div className="results-grid">
        {MODEL_ORDER.map((key) => (
          <ModelCard
            key={key}
            modelKey={key}
            prediction={predictions?.[key] || ''}
            metrics={metrics?.[key] || null}
            loading={loading}
          />
        ))}
      </div>
    </div>
  );
}

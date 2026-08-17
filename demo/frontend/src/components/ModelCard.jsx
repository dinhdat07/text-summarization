import MetricsDisplay from './MetricsDisplay';

const MODEL_INFO = {
  bartpho_fft: {
    name: 'BARTpho',
    subtitle: 'Full Fine-Tuning',
    badgeClass: 'badge-fft',
    badgeText: 'FFT',
  },
  bartpho_lora: {
    name: 'BARTpho',
    subtitle: 'LoRA (r=16)',
    badgeClass: 'badge-lora',
    badgeText: 'LoRA',
  },
  qwen_lora: {
    name: 'Qwen2.5-0.5B',
    subtitle: 'LoRA (r=16)',
    badgeClass: 'badge-qwen',
    badgeText: 'LoRA',
  },
};

export default function ModelCard({ modelKey, prediction, metrics, loading }) {
  const info = MODEL_INFO[modelKey] || { name: modelKey, subtitle: '', badgeClass: '', badgeText: '' };

  return (
    <div className="model-card">
      <div className="model-card-header">
        <div>
          <span style={{ fontWeight: 600, fontSize: '1.05rem' }}>{info.name}</span>
          <span className="text-sm text-muted" style={{ marginLeft: '0.5rem' }}>{info.subtitle}</span>
        </div>
        <span className={`badge ${info.badgeClass}`}>{info.badgeText}</span>
      </div>

      <div className="model-card-body">
        {loading ? (
          <div>
            <div className="skeleton" style={{ height: '1rem', width: '100%', marginBottom: '0.5rem' }} />
            <div className="skeleton" style={{ height: '1rem', width: '90%', marginBottom: '0.5rem' }} />
            <div className="skeleton" style={{ height: '1rem', width: '75%' }} />
          </div>
        ) : prediction ? (
          <p className="summary-text">{prediction}</p>
        ) : (
          <div className="empty-state">
            <p>Chưa có kết quả</p>
          </div>
        )}
      </div>

      {metrics && (
        <div className="model-card-footer">
          <MetricsDisplay metrics={metrics} />
        </div>
      )}
    </div>
  );
}

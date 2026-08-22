import { Sparkles } from 'lucide-react';
import ModelCard from './ModelCard';

const MODEL_ORDER = ['bartpho_fft', 'bartpho_lora', 'qwen_lora', 'custom_transformer'];

export default function ResultsPanel({ predictions, metrics, loading }) {
  if (!predictions && !loading) return null;

  return (
    <div style={{ animation: 'slideIn 0.4s ease-out' }}>
      <div className="flex items-center gap-2 mb-4 mt-6">
        <Sparkles size={20} color="var(--primary-bright)" />
        <h2 className="text-primary" style={{ fontSize: '18px', fontWeight: 600 }}>Kết quả tóm tắt</h2>
      </div>
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

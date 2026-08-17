import { useState } from 'react';
import { Shuffle, Search } from 'lucide-react';

export default function SampleBrowser({ onSampleLoaded, loading }) {
  const [sampleIndex, setSampleIndex] = useState('');

  return (
    <div className="sample-browser">
      <button
        className="btn"
        onClick={() => onSampleLoaded('random')}
        disabled={loading}
      >
        <Shuffle size={16} />
        <span>Mẫu ngẫu nhiên</span>
      </button>
      <span className="text-muted text-sm">hoặc</span>
      <div className="flex items-center gap-1">
        <input
          className="input"
          type="number"
          min="0"
          max="999"
          placeholder="Số thứ tự (0-999)"
          value={sampleIndex}
          onChange={(e) => setSampleIndex(e.target.value)}
          style={{ width: '160px' }}
        />
        <button
          className="btn-secondary btn"
          onClick={() => onSampleLoaded('index', parseInt(sampleIndex))}
          disabled={loading || sampleIndex === ''}
        >
          <Search size={16} />
        </button>
      </div>
    </div>
  );
}

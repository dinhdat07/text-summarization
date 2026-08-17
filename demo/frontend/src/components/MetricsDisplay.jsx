export default function MetricsDisplay({ metrics }) {
  if (!metrics) return null;

  const metricItems = [
    { key: 'rouge1', label: 'ROUGE-1' },
    { key: 'rouge2', label: 'ROUGE-2' },
    { key: 'rougeL', label: 'ROUGE-L' },
    { key: 'bleu', label: 'BLEU' },
    { key: 'bertscore', label: 'BERTScore' },
  ];

  const getColor = (value) => {
    if (value >= 60) return 'var(--success)';
    if (value >= 30) return 'var(--warning)';
    return 'var(--error)';
  };

  return (
    <div className="flex flex-col gap-2">
      {metricItems.map(({ key, label }) => {
        const value = metrics[key] || 0;
        return (
          <div key={key}>
            <div className="flex items-center justify-between" style={{ marginBottom: '4px' }}>
              <span className="text-sm text-muted">{label}</span>
              <span className="metric-value">{value.toFixed(2)}</span>
            </div>
            <div className="metric-bar">
              <div
                className="metric-bar-fill"
                style={{
                  width: `${Math.min(value, 100)}%`,
                  backgroundColor: getColor(value),
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

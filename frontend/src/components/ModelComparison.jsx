import { useState } from 'react'

const FIELDS = [
  { key: 'sharpe',       label: 'Sharpe',    best: 'max', fmt: v => v.toFixed(3) },
  { key: 'sortino',      label: 'Sortino',   best: 'max', fmt: v => v.toFixed(3) },
  { key: 'cagr',         label: 'CAGR',      best: 'max', fmt: v => `${(v * 100).toFixed(1)}%` },
  { key: 'max_drawdown', label: 'Max DD',    best: 'max', fmt: v => `${(v * 100).toFixed(1)}%` },
  { key: 'win_rate',     label: 'Win Rate',  best: 'max', fmt: v => `${(v * 100).toFixed(1)}%` },
  { key: 'turnover',     label: 'Turnover',  best: 'min', fmt: v => `${(v * 100).toFixed(1)}%` },
  { key: 'roc_auc',      label: 'ROC-AUC',   best: 'max', fmt: v => v.toFixed(3) },
]

const MODEL_LABELS = {
  logistic_regression: 'Logistic Regression',
  random_forest:       'Random Forest',
  xgboost:             'XGBoost',
  lstm:                'LSTM',
}

function bestValue(models, key, direction) {
  const vals = models.map(m => m[key])
  return direction === 'max' ? Math.max(...vals) : Math.min(...vals)
}

export default function ModelComparison({ performance, activeModel }) {
  const [year, setYear]       = useState('test')
  const [costBps, setCostBps] = useState(10)

  if (!performance) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 animate-pulse">
        <div className="h-4 bg-slate-800 rounded w-48 mb-4" />
        <div className="space-y-2">
          {[...Array(4)].map((_, i) => <div key={i} className="h-8 bg-slate-800 rounded" />)}
        </div>
      </div>
    )
  }

  const costDecimal  = costBps / 10000
  const sourceModels = year === 'test' ? performance.models : performance.val_models

  // Enrich each model with net_cagr so we can swap the CAGR column seamlessly
  const activeModels = sourceModels.map(m => ({
    ...m,
    net_cagr: m.cagr - m.turnover * 252 * costDecimal,
  }))

  // When costBps > 0, display Net CAGR instead of gross CAGR
  const displayFields = FIELDS.map(f =>
    f.key === 'cagr' && costBps > 0
      ? { ...f, key: 'net_cagr', label: 'Net CAGR*' }
      : f
  )

  const bests = Object.fromEntries(
    displayFields.map(f => [f.key, bestValue(activeModels, f.key, f.best)])
  )

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 gap-4 flex-wrap">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-medium text-slate-300">Model Performance</h2>
          {/* Year toggle */}
          <div className="flex gap-0.5 bg-slate-800/60 rounded-md p-0.5">
            {[{ key: 'test', label: '2024 Test' }, { key: 'val', label: '2023 Val' }].map(y => (
              <button
                key={y.key}
                onClick={() => setYear(y.key)}
                className={`text-xs px-2.5 py-1 rounded transition-colors ${
                  year === y.key
                    ? 'bg-slate-700 text-slate-100'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {y.label}
              </button>
            ))}
          </div>
        </div>
        {/* Cost input */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Cost:</span>
          <input
            type="number"
            min="0"
            max="100"
            step="1"
            value={costBps}
            onChange={e => setCostBps(Math.max(0, parseInt(e.target.value) || 0))}
            className="w-14 bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded px-2 py-1 text-right focus:outline-none focus:border-slate-500"
          />
          <span className="text-xs text-slate-500">bps/trade</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-800">
              <th className="text-left text-slate-500 font-normal py-2 pr-4">Model</th>
              {displayFields.map(f => (
                <th key={f.key} className="text-right text-slate-500 font-normal py-2 px-2 whitespace-nowrap">
                  {f.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {activeModels.map(m => {
              const isActive = m.model === activeModel
              return (
                <tr
                  key={m.model}
                  className={`border-b border-slate-800/50 ${isActive ? 'bg-slate-800/40' : ''}`}
                >
                  <td className="py-2.5 pr-4 whitespace-nowrap">
                    <span className={isActive ? 'text-slate-100 font-medium' : 'text-slate-400'}>
                      {MODEL_LABELS[m.model] ?? m.model}
                    </span>
                    {isActive && <span className="ml-2 text-xs text-slate-500">active</span>}
                  </td>
                  {displayFields.map(f => {
                    const isBest = m[f.key] === bests[f.key]
                    return (
                      <td key={f.key} className="text-right py-2.5 px-2 tabular-nums whitespace-nowrap">
                        <span className={isBest ? 'text-emerald-400 font-medium' : 'text-slate-300'}>
                          {f.fmt(m[f.key])}
                        </span>
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-slate-600 mt-3">
        {costBps > 0
          ? `* Net CAGR after ${costBps} bps/trade · annualized cost = turnover × 252 × ${costBps / 100}%.`
          : 'Highlighted values are best in column. Sortino and XGBoost high Sortino reflects low downside volatility when active.'}
        {' '}No transaction costs on other metrics.
      </p>
    </div>
  )
}

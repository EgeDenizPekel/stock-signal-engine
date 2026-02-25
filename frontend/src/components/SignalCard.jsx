export default function SignalCard({ signal, loading, modelMetrics }) {
  if (loading || !signal) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 animate-pulse">
        <div className="h-4 bg-slate-800 rounded w-12 mb-3" />
        <div className="h-12 bg-slate-800 rounded w-20 mb-3" />
        <div className="h-3 bg-slate-800 rounded w-32" />
      </div>
    )
  }

  const isBuy = signal.signal === 'BUY'
  const pct = (signal.probability * 100).toFixed(1)
  const generatedAt = new Date(signal.generated_at).toLocaleString()

  const gain    = modelMetrics ? (modelMetrics.avg_gain  * 100).toFixed(2) : null
  const loss    = modelMetrics ? (modelMetrics.avg_loss  * 100).toFixed(2) : null
  const payoff  = modelMetrics ? modelMetrics.payoff_ratio.toFixed(2)      : null

  return (
    <div className={`border rounded-xl p-6 ${isBuy ? 'bg-emerald-950 border-emerald-800' : 'bg-slate-900 border-slate-800'}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-slate-400 text-sm mb-1">{signal.ticker}</p>
          <p className={`text-5xl font-bold tracking-tight ${isBuy ? 'text-emerald-400' : 'text-slate-300'}`}>
            {signal.signal}
          </p>
          <p className="text-slate-400 text-sm mt-2">
            {pct}% est. probability of &gt;2% gain in 5 days
          </p>
          {modelMetrics && (
            <div className="flex items-center gap-4 mt-3 pt-3 border-t border-slate-800/60 text-xs">
              <span className="text-slate-500">
                Avg gain <span className="text-emerald-400 font-medium">+{gain}%</span>
              </span>
              <span className="text-slate-500">
                Avg loss <span className="text-red-400 font-medium">{loss}%</span>
              </span>
              <span className="text-slate-500">
                Payoff <span className="text-slate-200 font-medium">{payoff}×</span>
              </span>
            </div>
          )}
        </div>
        <div className="text-right">
          <p className="text-slate-500 text-xs">model</p>
          <p className="text-slate-300 text-sm">{signal.model.replace(/_/g, ' ')}</p>
          <p className="text-slate-500 text-xs mt-3">generated at</p>
          <p className="text-slate-400 text-xs">{generatedAt}</p>
        </div>
      </div>
    </div>
  )
}

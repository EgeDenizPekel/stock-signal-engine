import { useState } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts'

// ---------------------------------------------------------------------------
// Price chart helpers
// ---------------------------------------------------------------------------

const PriceDot = (props) => {
  const { cx, cy, payload } = props
  if (payload.buy == null) return null
  return <circle cx={cx} cy={cy} r={4} fill="#34d399" stroke="#064e3b" strokeWidth={1} />
}

const PriceTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-slate-800 border border-slate-700 rounded px-3 py-2 text-xs">
      <p className="text-slate-400 mb-1">{d.date}</p>
      <p className="text-slate-100">Close: <span className="font-medium">${d.close?.toFixed(2)}</span></p>
      <p className={d.signal === 'BUY' ? 'text-emerald-400' : 'text-slate-400'}>Signal: {d.signal}</p>
      <p className="text-slate-400">Prob: {(d.probability * 100).toFixed(1)}%</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Returns chart helpers
// ---------------------------------------------------------------------------

function computeReturns(data) {
  if (!data || data.length < 2) return []
  const result = [{ date: data[0].date, strategy: 0, buyHold: 0 }]
  let cumStrategy = 1
  let cumBuyHold = 1
  for (let i = 1; i < data.length; i++) {
    const dailyReturn = data[i].close / data[i - 1].close - 1
    // shift by 1: yesterday's signal determines today's position (matches backtest)
    const stratReturn = data[i - 1].signal === 'BUY' ? dailyReturn : 0
    cumStrategy *= (1 + stratReturn)
    cumBuyHold *= (1 + dailyReturn)
    result.push({
      date: data[i].date,
      strategy: +((cumStrategy - 1) * 100).toFixed(2),
      buyHold: +((cumBuyHold - 1) * 100).toFixed(2),
    })
  }
  return result
}

const ReturnsTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const fmt = v => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`
  return (
    <div className="bg-slate-800 border border-slate-700 rounded px-3 py-2 text-xs">
      <p className="text-slate-400 mb-1">{d.date}</p>
      <p className="text-emerald-400">Strategy: {fmt(d.strategy)}</p>
      <p className="text-slate-400">Buy & Hold: {fmt(d.buyHold)}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Drawdown chart helpers
// ---------------------------------------------------------------------------

function computeDrawdown(returnsData) {
  if (!returnsData.length) return []
  let peak = 1
  return returnsData.map(d => {
    const cumFactor = d.strategy / 100 + 1
    if (cumFactor > peak) peak = cumFactor
    const dd = peak <= 0 ? 0 : ((cumFactor - peak) / peak) * 100
    return { date: d.date, drawdown: parseFloat(dd.toFixed(2)) }
  })
}

const DrawdownTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-slate-800 border border-slate-700 rounded px-3 py-2 text-xs">
      <p className="text-slate-400 mb-1">{d.date}</p>
      <p className="text-red-400">Drawdown: {d.drawdown.toFixed(2)}%</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function PriceChart({ history, loading, days, onDaysChange, daysOptions }) {
  const [view, setView] = useState('price')

  const priceData = history?.data?.map(p => ({
    ...p,
    buy: p.signal === 'BUY' ? p.close : null,
  })) ?? []

  const returnsData = computeReturns(history?.data ?? [])
  const drawdownData = computeDrawdown(returnsData)

  const tabs = [
    {
      key: 'price',
      label: 'Price',
      tip: 'Close price over the selected period. Green dots mark days when the model generated a BUY signal.',
    },
    {
      key: 'returns',
      label: 'Strategy vs Buy & Hold',
      tip: 'Cumulative return of the signal-based strategy (green) vs. passive buy-and-hold (grey). The strategy holds only on BUY days; otherwise stays in cash. Signal is shifted 1 day to avoid lookahead bias.',
    },
    {
      key: 'drawdown',
      label: 'Drawdown',
      tip: 'Rolling peak-to-trough loss of the signal strategy. Shows how far below its previous high the portfolio sits at any point. Deeper troughs mean larger unrealised losses.',
    },
  ]

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
      {/* Header row */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex gap-1 bg-slate-800/60 rounded-lg p-0.5">
          {tabs.map(t => (
            <div key={t.key} className="relative group">
              <button
                onClick={() => setView(t.key)}
                className={`text-xs px-3 py-1.5 rounded-md transition-colors ${
                  view === t.key
                    ? 'bg-slate-700 text-slate-100'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {t.label}
              </button>
              {/* Hover tooltip */}
              <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-300 leading-relaxed opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-50 shadow-xl">
                {t.tip}
                {/* Arrow */}
                <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-700" />
              </div>
            </div>
          ))}
        </div>
        <div className="flex gap-1">
          {daysOptions.map(d => (
            <button
              key={d}
              onClick={() => onDaysChange(d)}
              className={`text-xs px-3 py-1 rounded transition-colors ${
                days === d ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {d}d
            </button>
          ))}
        </div>
      </div>

      {/* Chart area */}
      {loading || !history ? (
        <div className="h-64 flex items-center justify-center">
          <div className="w-6 h-6 border-2 border-slate-600 border-t-slate-300 rounded-full animate-spin" />
        </div>
      ) : view === 'price' ? (
        <>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={priceData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="date"
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickLine={false}
                tickFormatter={d => d.slice(5)}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={v => `$${v.toFixed(0)}`}
                domain={['auto', 'auto']}
                width={55}
              />
              <Tooltip content={<PriceTooltip />} />
              <Line
                type="monotone"
                dataKey="close"
                stroke="#94a3b8"
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 3, fill: '#94a3b8' }}
              />
              <Line
                type="monotone"
                dataKey="buy"
                stroke="transparent"
                dot={<PriceDot />}
                activeDot={false}
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-2 mt-3">
            <span className="inline-block w-3 h-3 rounded-full bg-emerald-400" />
            <span className="text-xs text-slate-500">BUY signal day</span>
          </div>
        </>
      ) : view === 'returns' ? (
        <>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={returnsData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="date"
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickLine={false}
                tickFormatter={d => d.slice(5)}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={v => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`}
                domain={['auto', 'auto']}
                width={58}
              />
              <Tooltip content={<ReturnsTooltip />} />
              <ReferenceLine y={0} stroke="#334155" strokeDasharray="3 3" />
              <Line
                type="monotone"
                dataKey="strategy"
                stroke="#34d399"
                strokeWidth={1.5}
                dot={false}
                name="Strategy"
              />
              <Line
                type="monotone"
                dataKey="buyHold"
                stroke="#64748b"
                strokeWidth={1.5}
                dot={false}
                strokeDasharray="4 2"
                name="Buy & Hold"
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-4 mt-3">
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-5 h-0.5 bg-emerald-400 rounded" />
              <span className="text-xs text-slate-500">Strategy (signal-based)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-5 h-0.5 bg-slate-500 rounded" style={{ backgroundImage: 'repeating-linear-gradient(to right, #64748b 0, #64748b 4px, transparent 4px, transparent 6px)' }} />
              <span className="text-xs text-slate-500">Buy & Hold</span>
            </div>
          </div>
        </>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={drawdownData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="date"
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickLine={false}
                tickFormatter={d => d.slice(5)}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={v => `${v.toFixed(1)}%`}
                domain={['auto', 0]}
                width={58}
              />
              <Tooltip content={<DrawdownTooltip />} />
              <ReferenceLine y={0} stroke="#334155" />
              <Area
                type="monotone"
                dataKey="drawdown"
                stroke="#f87171"
                fill="#450a0a"
                strokeWidth={1.5}
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-2 mt-3">
            <span className="inline-block w-5 h-0.5 rounded" style={{ background: '#f87171' }} />
            <span className="text-xs text-slate-500">Strategy drawdown from peak</span>
          </div>
        </>
      )}
    </div>
  )
}

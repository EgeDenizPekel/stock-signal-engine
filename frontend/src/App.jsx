import { useState, useEffect } from 'react'
import { fetchSignal, fetchHistory, fetchPerformance } from './lib/api'
import SignalCard from './components/SignalCard'
import PriceChart from './components/PriceChart'
import ModelComparison from './components/ModelComparison'
import Dropdown from './components/Dropdown'
import Limitations from './components/Limitations'

const TICKER_OPTIONS = [
  { value: 'AAPL',  label: 'Apple' },
  { value: 'MSFT',  label: 'Microsoft' },
  { value: 'GOOGL', label: 'Alphabet' },
  { value: 'AMZN',  label: 'Amazon' },
  { value: 'TSLA',  label: 'Tesla' },
  { value: 'NVDA',  label: 'NVIDIA' },
  { value: 'META',  label: 'Meta' },
  { value: 'JPM',   label: 'JPMorgan Chase' },
  { value: 'SPY',   label: 'S&P 500 ETF' },
]

const MODEL_OPTIONS = [
  { value: 'logistic_regression', label: 'Logistic Regression' },
  { value: 'random_forest',       label: 'Random Forest' },
  { value: 'xgboost',             label: 'XGBoost' },
  { value: 'lstm',                label: 'LSTM' },
]

const DAYS_OPTIONS = [30, 60, 90]

export default function App() {
  const [ticker, setTicker] = useState('AAPL')
  const [model, setModel] = useState('logistic_regression')
  const [days, setDays] = useState(90)

  const [signal, setSignal] = useState(null)
  const [history, setHistory] = useState(null)
  const [performance, setPerformance] = useState(null)

  const [signalLoading, setSignalLoading] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    setSignalLoading(true)
    setSignal(null)
    setError(null)
    fetchSignal(ticker, model)
      .then(setSignal)
      .catch(e => setError(e.message))
      .finally(() => setSignalLoading(false))
  }, [ticker, model])

  useEffect(() => {
    setHistoryLoading(true)
    setHistory(null)
    setError(null)
    fetchHistory(ticker, model, days)
      .then(setHistory)
      .catch(e => setError(e.message))
      .finally(() => setHistoryLoading(false))
  }, [ticker, model, days])

  useEffect(() => {
    fetchPerformance()
      .then(setPerformance)
      .catch(e => setError(e.message))
  }, [])

  return (
    <div className="h-screen bg-slate-950 text-slate-100 flex overflow-hidden">

      {/* Sidebar */}
      <aside className="w-56 shrink-0 sticky top-0 h-screen flex flex-col border-r border-slate-800/80 p-5">
        <div className="mb-8">
          <h1 className="text-base font-semibold text-slate-100 tracking-tight">
            Stock Signal Engine
          </h1>
        </div>

        <div className="space-y-5">
          <Dropdown
            label="Stock"
            options={TICKER_OPTIONS}
            value={ticker}
            onChange={(v) => { setTicker(v); setError(null) }}
          />
          <Dropdown
            label="Model"
            options={MODEL_OPTIONS}
            value={model}
            onChange={(v) => { setModel(v); setError(null) }}
          />
        </div>

        <div className="mt-auto pt-5 border-t border-slate-800/60 space-y-4">
          <div>
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2">
              Model Context
            </p>
            <div className="space-y-1.5">
              {[
                ['Horizon',    '5 trading days'],
                ['Rebalance',  'Daily'],
                ['Train',      '2018 – 2022'],
                ['Val',        '2023'],
                ['Test',       '2024'],
                ['Costs',      'None assumed'],
                ['Universe',   '9 stocks + SPY'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between text-xs">
                  <span className="text-slate-600">{k}</span>
                  <span className="text-slate-400">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 min-w-0 overflow-y-auto relative z-10">
        <main className="px-6 py-8 space-y-6">
          {error && (
            <div className="bg-red-950 border border-red-800 text-red-300 text-sm rounded-lg px-4 py-3">
              {error}
            </div>
          )}

          <SignalCard
            signal={signal}
            loading={signalLoading}
            modelMetrics={performance?.models?.find(m => m.model === model) ?? null}
          />

          <PriceChart
            history={history}
            loading={historyLoading}
            days={days}
            onDaysChange={setDays}
            daysOptions={DAYS_OPTIONS}
          />

          <ModelComparison performance={performance} activeModel={model} />
        </main>
      </div>

      {/* Right sidebar */}
      <aside className="w-56 shrink-0 sticky top-0 h-screen flex flex-col border-l border-slate-800/80 p-5">
        <div className="mt-auto pt-5 border-t border-slate-800/60">
          <Limitations />
        </div>
      </aside>

    </div>
  )
}

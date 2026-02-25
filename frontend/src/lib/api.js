const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export async function fetchSignal(ticker, model) {
  const res = await fetch(`${BASE}/signal/${ticker}?model=${model}`)
  if (!res.ok) throw new Error(`Signal fetch failed: ${res.status}`)
  return res.json()
}

export async function fetchHistory(ticker, model, days) {
  const res = await fetch(`${BASE}/history/${ticker}?model=${model}&days=${days}`)
  if (!res.ok) throw new Error(`History fetch failed: ${res.status}`)
  return res.json()
}

export async function fetchPerformance() {
  const res = await fetch(`${BASE}/models/performance`)
  if (!res.ok) throw new Error(`Performance fetch failed: ${res.status}`)
  return res.json()
}

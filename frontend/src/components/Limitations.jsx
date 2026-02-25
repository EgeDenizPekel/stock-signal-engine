const LIMITATIONS = [
  {
    title: 'Survivorship bias',
    desc: '9 tickers chosen because they survived to 2024. Delisted and failed companies are absent.',
  },
  {
    title: 'Single test year',
    desc: '2024 alone cannot confirm robustness across regimes — recessions, crashes, or rate cycles.',
  },
  {
    title: 'Val metrics are skewed',
    desc: '2023 was a bull recovery year. LR +62%, LSTM +69% CAGR are not long-run expectations.',
  },
  {
    title: 'No slippage or impact',
    desc: 'All fills assumed at close price. Real execution adds bid-ask spread and market impact.',
  },
  {
    title: 'Long-only, no sizing',
    desc: 'Ignores bear-market signals. Position is always 100% in or 100% cash — no risk weighting.',
  },
  {
    title: 'Technical features only',
    desc: 'Earnings surprises, macro data, and news sentiment are not in the feature set.',
  },
  {
    title: 'Static after training',
    desc: 'No online learning. Drift from the 2018–2022 training distribution is not corrected.',
  },
]

export default function Limitations() {
  return (
    <div>
      <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">
        Where This Fails
      </p>
      <div className="space-y-3">
        {LIMITATIONS.map(({ title, desc }) => (
          <div key={title}>
            <p className="text-xs font-medium text-rose-400/80">{title}</p>
            <p className="text-xs text-slate-600 mt-0.5 leading-relaxed">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

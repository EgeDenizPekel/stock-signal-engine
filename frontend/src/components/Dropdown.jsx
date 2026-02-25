import { useState, useRef, useEffect } from 'react'

export default function Dropdown({ label, options, value, onChange }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  const selected = options.find(o => o.value === value)

  useEffect(() => {
    const close = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  return (
    <div className="relative" ref={ref}>
      {label && (
        <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2">
          {label}
        </p>
      )}

      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between bg-slate-800/60 border border-slate-700/80 rounded-lg px-3.5 py-2.5 text-sm text-slate-200 hover:bg-slate-800 hover:border-slate-600 transition-all duration-150 focus:outline-none focus:ring-1 focus:ring-slate-500"
      >
        <span className="font-medium">{selected?.label ?? value}</span>
        <svg
          className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="absolute top-full left-0 right-0 mt-1.5 bg-slate-800 border border-slate-700 rounded-lg shadow-2xl shadow-black/50 z-20 overflow-hidden">
          {options.map(opt => {
            const isActive = opt.value === value
            return (
              <button
                key={opt.value}
                onClick={() => { onChange(opt.value); setOpen(false) }}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 text-sm text-left transition-colors duration-100 ${
                  isActive
                    ? 'bg-slate-700 text-slate-100'
                    : 'text-slate-300 hover:bg-slate-700/60 hover:text-slate-100'
                }`}
              >
                <span>{opt.label}</span>
                {isActive && (
                  <svg className="w-3.5 h-3.5 text-emerald-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

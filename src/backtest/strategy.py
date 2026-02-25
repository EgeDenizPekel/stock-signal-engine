"""
Long-only backtesting strategy.

Given binary model predictions and raw close prices, simulate a simple strategy:
  - Signal = 1 → hold the stock (capture that day's return)
  - Signal = 0 → stay in cash (0% return)

Computes the annualized Sharpe ratio of this strategy.

Usage:
    from src.backtest.strategy import sharpe_from_signals
"""

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def daily_strategy_returns(
    predictions: np.ndarray,
    close_prices: pd.Series,
) -> pd.Series:
    """
    Compute daily strategy returns for a long-only signal strategy.

    Parameters
    ----------
    predictions : np.ndarray of shape (n,)
        Binary predictions aligned to the same dates as close_prices.
        1 = long, 0 = cash.
    close_prices : pd.Series
        Raw close prices with a DatetimeIndex, aligned to predictions.

    Returns
    -------
    pd.Series of daily strategy returns (same index as close_prices).
    """
    close = close_prices.copy()
    daily_returns = close.pct_change().fillna(0)

    # Shift predictions by 1: if today's model says buy (signal at close of day t),
    # we enter at the next day's open and capture day t+1's return.
    shifted_signals = pd.Series(predictions, index=close.index).shift(1).fillna(0)

    strategy_returns = shifted_signals * daily_returns
    return strategy_returns


def sharpe_ratio(returns: pd.Series) -> float:
    """
    Annualized Sharpe ratio assuming zero risk-free rate.

    Parameters
    ----------
    returns : pd.Series of daily strategy returns.

    Returns
    -------
    float — annualized Sharpe ratio.
    """
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_from_signals(
    predictions: np.ndarray,
    close_prices: pd.Series,
) -> float:
    """
    Convenience function: compute annualized Sharpe directly from predictions
    and close prices.
    """
    returns = daily_strategy_returns(predictions, close_prices)
    return sharpe_ratio(returns)


def buy_and_hold_sharpe(close_prices: pd.Series) -> float:
    """
    Sharpe ratio of a simple buy-and-hold strategy on the same prices.
    Useful as a baseline comparison.
    """
    daily_returns = close_prices.pct_change().fillna(0)
    return sharpe_ratio(daily_returns)


def cagr(returns: pd.Series) -> float:
    """Compound Annual Growth Rate over the full return series."""
    n = len(returns)
    if n == 0:
        return 0.0
    total = (1 + returns).prod()
    years = n / TRADING_DAYS_PER_YEAR
    return float(total ** (1 / years) - 1)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative value, e.g. -0.15 = -15%)."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def sortino_ratio(returns: pd.Series) -> float:
    """Annualized Sortino ratio (risk-free rate = 0, penalises downside only)."""
    mean_return = returns.mean()
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std()  # ddof=1; needs ≥2 observations
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(mean_return / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def win_rate(strategy_returns: pd.Series) -> float:
    """Fraction of active (BUY) trading days that closed with a positive return."""
    active = strategy_returns[strategy_returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def turnover(predictions: np.ndarray) -> float:
    """Average daily signal changes — 0 = never changes, 1 = flips every day."""
    if len(predictions) < 2:
        return 0.0
    return float((np.diff(predictions) != 0).mean())


def avg_gain(strategy_returns: pd.Series) -> float:
    """Average daily return on active (BUY) days that closed positive."""
    active = strategy_returns[strategy_returns != 0]
    winners = active[active > 0]
    return float(winners.mean()) if len(winners) > 0 else 0.0


def avg_loss(strategy_returns: pd.Series) -> float:
    """Average daily return on active (BUY) days that closed negative (negative value)."""
    active = strategy_returns[strategy_returns != 0]
    losers = active[active < 0]
    return float(losers.mean()) if len(losers) > 0 else 0.0


def payoff_ratio(strategy_returns: pd.Series) -> float:
    """avg_gain / abs(avg_loss) on active days. >1 means wins outsize losses."""
    g = avg_gain(strategy_returns)
    l = avg_loss(strategy_returns)
    if l == 0 or np.isnan(l):
        return 0.0
    return float(g / abs(l))

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

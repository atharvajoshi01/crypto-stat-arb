"""Walk-forward pair re-discovery.

Most stat-arb backtests discover a pair set on a training window and hold it
fixed for the rest of the backtest. In real markets cointegration is fragile:
a pair that was stably cointegrated in 2020-2021 may have decoupled by 2023.
Holding the original set forever overstates Sharpe and understates turnover.

This module closes the loop. It re-tests the current pair set on a rolling
window, drops pairs that have broken their cointegration property, and
discovers fresh candidates from the recent window. It also reports pair
turnover (added, dropped, retained) as a first-class metric so the user can
see how stable the pair universe actually is.

Usage::

    from cryptoarb.rediscovery import WalkForwardRediscovery

    rediscovery = WalkForwardRediscovery(
        refresh_every_days=90,
        lookback_days=365,
        adf_pvalue=0.05,
        min_half_life=3.0,
        max_half_life=30.0,
        max_pairs=20,
        min_correlation=0.70,
    )
    history = rediscovery.run(log_price_matrix, initial_pairs=initial_pairs)

The returned ``RediscoveryHistory`` lists every refresh event with the pairs
held before, after, plus the dropped and added sets and timing metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from cryptoarb.pairs import PairResult, discover_pairs, test_cointegration

logger = logging.getLogger(__name__)


@dataclass
class RediscoveryEvent:
    """A single refresh point in the walk-forward schedule."""

    refresh_date: pd.Timestamp
    pairs_before: List[Tuple[str, str]]
    pairs_after: List[Tuple[str, str]]
    dropped: List[Tuple[str, str]] = field(default_factory=list)
    added: List[Tuple[str, str]] = field(default_factory=list)
    retained: List[Tuple[str, str]] = field(default_factory=list)
    n_tested: int = 0

    @property
    def churn_rate(self) -> float:
        """Fraction of the pre-refresh set that did not survive."""
        if not self.pairs_before:
            return 0.0
        return len(self.dropped) / len(self.pairs_before)

    def to_dict(self) -> Dict:
        def _serialize(pairs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
            return [{"asset_a": a, "asset_b": b} for a, b in pairs]

        return {
            "refresh_date": self.refresh_date.strftime("%Y-%m-%d"),
            "n_before": len(self.pairs_before),
            "n_after": len(self.pairs_after),
            "n_dropped": len(self.dropped),
            "n_added": len(self.added),
            "n_retained": len(self.retained),
            "n_tested": self.n_tested,
            "churn_rate": round(self.churn_rate, 4),
            "pairs_before": _serialize(self.pairs_before),
            "pairs_after": _serialize(self.pairs_after),
            "dropped": _serialize(self.dropped),
            "added": _serialize(self.added),
            "retained": _serialize(self.retained),
        }


@dataclass
class RediscoveryHistory:
    """Sequence of refresh events plus the final pair set."""

    events: List[RediscoveryEvent] = field(default_factory=list)
    final_pairs: List[PairResult] = field(default_factory=list)

    @property
    def n_refreshes(self) -> int:
        return len(self.events)

    @property
    def avg_churn(self) -> float:
        if not self.events:
            return 0.0
        return sum(e.churn_rate for e in self.events) / len(self.events)

    @property
    def total_dropped(self) -> int:
        return sum(len(e.dropped) for e in self.events)

    @property
    def total_added(self) -> int:
        return sum(len(e.added) for e in self.events)

    def to_dict(self) -> Dict:
        return {
            "n_refreshes": self.n_refreshes,
            "avg_churn": round(self.avg_churn, 4),
            "total_dropped": self.total_dropped,
            "total_added": self.total_added,
            "n_final_pairs": len(self.final_pairs),
            "events": [e.to_dict() for e in self.events],
        }


def _pair_key(pair: PairResult) -> Tuple[str, str]:
    """Canonical (asset_a, asset_b) key, alphabetized so order doesn't matter."""
    return tuple(sorted((pair.asset_a, pair.asset_b)))  # type: ignore[return-value]


def _retest_existing(
    log_prices_window: pd.DataFrame,
    pairs: List[PairResult],
    adf_pvalue: float,
    min_half_life: float,
    max_half_life: float,
) -> Tuple[List[PairResult], List[Tuple[str, str]]]:
    """Re-test cointegration of an existing pair set on a fresh window.

    Returns (survived, dropped_keys).
    """
    survived: List[PairResult] = []
    dropped: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair.asset_a not in log_prices_window.columns or pair.asset_b not in log_prices_window.columns:
            dropped.append(_pair_key(pair))
            continue
        retested = test_cointegration(
            log_prices_window[pair.asset_a],
            log_prices_window[pair.asset_b],
            adf_pvalue_threshold=adf_pvalue,
        )
        if retested.cointegrated and min_half_life <= retested.half_life <= max_half_life:
            survived.append(retested)
        else:
            dropped.append(_pair_key(pair))
    return survived, dropped


@dataclass
class WalkForwardRediscovery:
    """Periodic pair-set refresh on a rolling window.

    Parameters mirror ``discover_pairs`` plus a refresh cadence and lookback.

    Attributes:
        refresh_every_days: Cadence of the refresh schedule. The first refresh
            fires ``refresh_every_days`` after the start of the test window.
        lookback_days: Window used for re-testing and discovery at each event.
        min_correlation: Pre-filter threshold for new candidates.
        adf_pvalue: ADF p-value threshold for cointegration.
        min_half_life: Lower bound on half-life in days.
        max_half_life: Upper bound on half-life in days.
        max_pairs: Cap on the total pair set after each refresh.
    """

    refresh_every_days: int = 90
    lookback_days: int = 365
    min_correlation: float = 0.70
    adf_pvalue: float = 0.05
    min_half_life: float = 3.0
    max_half_life: float = 30.0
    max_pairs: int = 20

    def run(
        self,
        log_prices: pd.DataFrame,
        initial_pairs: List[PairResult],
        start_date: Optional[pd.Timestamp] = None,
    ) -> RediscoveryHistory:
        """Execute the rolling rediscovery schedule.

        Args:
            log_prices: Full log-price matrix (dates × symbols).
            initial_pairs: Pair set known at ``start_date``.
            start_date: First date in the test window. Defaults to the
                earliest index in ``log_prices``.

        Returns:
            RediscoveryHistory with one event per refresh.
        """
        if log_prices.empty:
            raise ValueError("log_prices is empty.")

        index = pd.DatetimeIndex(log_prices.index)
        if start_date is None:
            start_date = index[0]
        else:
            start_date = pd.Timestamp(start_date)

        last_date = index[-1]
        history = RediscoveryHistory(final_pairs=list(initial_pairs))
        current_pairs = list(initial_pairs)
        refresh_date = start_date + pd.Timedelta(days=self.refresh_every_days)

        while refresh_date <= last_date:
            window_start = refresh_date - pd.Timedelta(days=self.lookback_days)
            window = log_prices.loc[window_start:refresh_date]
            if window.empty or len(window) < 30:
                logger.warning(
                    f"Skipping refresh at {refresh_date.date()}: window too small "
                    f"({len(window)} rows)."
                )
                refresh_date += pd.Timedelta(days=self.refresh_every_days)
                continue

            pairs_before = [_pair_key(p) for p in current_pairs]
            survived, dropped = _retest_existing(
                window, current_pairs,
                adf_pvalue=self.adf_pvalue,
                min_half_life=self.min_half_life,
                max_half_life=self.max_half_life,
            )

            # Discover fresh candidates on the same window.
            fresh = discover_pairs(
                window,
                min_correlation=self.min_correlation,
                adf_pvalue=self.adf_pvalue,
                min_half_life=self.min_half_life,
                max_half_life=self.max_half_life,
                max_pairs=self.max_pairs,
            )

            survived_keys = {_pair_key(p) for p in survived}
            fresh_new = [p for p in fresh if _pair_key(p) not in survived_keys]

            combined = survived + fresh_new
            # Re-rank by ADF strength so the cap drops the weakest.
            combined.sort(key=lambda p: p.adf_stat)
            current_pairs = combined[: self.max_pairs]

            pairs_after = [_pair_key(p) for p in current_pairs]
            added = [k for k in pairs_after if k not in pairs_before]
            retained = [k for k in pairs_after if k in pairs_before]

            event = RediscoveryEvent(
                refresh_date=refresh_date,
                pairs_before=pairs_before,
                pairs_after=pairs_after,
                dropped=dropped,
                added=added,
                retained=retained,
                n_tested=len(current_pairs) + len(dropped),
            )
            history.events.append(event)
            logger.info(
                f"Refresh {refresh_date.date()}: dropped={len(dropped)}, "
                f"added={len(added)}, retained={len(retained)}, "
                f"churn={event.churn_rate:.1%}"
            )

            refresh_date += pd.Timedelta(days=self.refresh_every_days)

        history.final_pairs = current_pairs
        return history

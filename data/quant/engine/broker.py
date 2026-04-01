"""
Simulated broker — handles order execution with:
- Round-lot (100 shares) constraint
- One-way commission on both buy and sell
- Slippage support
- Insufficient-funds guard (buy fails gracefully)
- Per-trade record keeping
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BacktestConfig


@dataclass
class TradeRecord:
    """Single trade record."""
    date: str
    ts_code: str
    direction: str          # "BUY" / "SELL"
    price: float            # fill price (after slippage)
    shares: int             # shares traded
    amount: float           # price * shares
    commission: float       # commission paid
    status: str             # "FILLED" / "REJECTED"
    reason: str = ""        # rejection reason


class SimBroker:
    """
    Simulated broker for A-share equities.

    Maintains:
      - cash balance
      - holdings: {ts_code: shares}
      - trade log: list of TradeRecord
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.cash: float = cfg.initial_capital
        self.holdings: Dict[str, int] = {}      # ts_code → shares
        self.cost_basis: Dict[str, dict] = {}   # ts_code → {"price": avg_cost, "date": first_buy_date}
        self.trade_log: List[TradeRecord] = []

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_slippage(price: float, slippage: float,
                        direction: str) -> float:
        """Apply slippage: buy higher, sell lower."""
        if direction == "BUY":
            return price * (1.0 + slippage)
        else:
            return price * (1.0 - slippage)

    # ------------------------------------------------------------------
    # Core order methods
    # ------------------------------------------------------------------

    def execute_buy(self, date: str, ts_code: str,
                    target_shares: int, price: float) -> TradeRecord:
        """
        Attempt to buy *target_shares* (already rounded to lot_size)
        at *price*.  Returns TradeRecord with status.
        """
        lot = self.cfg.lot_size
        # Ensure round-lot
        shares = (target_shares // lot) * lot
        if shares <= 0:
            rec = TradeRecord(date, ts_code, "BUY", price, 0, 0.0,
                              0.0, "REJECTED", "目标股数不足一手")
            self.trade_log.append(rec)
            return rec

        fill_price = self._apply_slippage(price, self.cfg.slippage, "BUY")
        amount = fill_price * shares
        commission = amount * self.cfg.commission_rate

        total_cost = amount + commission
        if total_cost > self.cash:
            # Try to reduce to affordable lots
            affordable = int(self.cash / (fill_price * (1 + self.cfg.commission_rate)))
            shares = (affordable // lot) * lot
            if shares <= 0:
                rec = TradeRecord(date, ts_code, "BUY", fill_price, 0, 0.0,
                                  0.0, "REJECTED", "资金不足")
                self.trade_log.append(rec)
                return rec
            amount = fill_price * shares
            commission = amount * self.cfg.commission_rate
            total_cost = amount + commission

        # Execute
        self.cash -= total_cost
        old_shares = self.holdings.get(ts_code, 0)
        new_shares = old_shares + shares
        self.holdings[ts_code] = new_shares

        # Update cost basis (weighted average price)
        if ts_code in self.cost_basis and old_shares > 0:
            old_cost = self.cost_basis[ts_code]["price"]
            avg_price = (old_cost * old_shares + fill_price * shares) / new_shares
            self.cost_basis[ts_code]["price"] = avg_price
            # Keep the original first buy date
        else:
            self.cost_basis[ts_code] = {"price": fill_price, "date": date}

        rec = TradeRecord(date, ts_code, "BUY", fill_price, shares,
                          amount, commission, "FILLED")
        self.trade_log.append(rec)
        return rec

    def execute_sell(self, date: str, ts_code: str,
                     target_shares: int, price: float) -> TradeRecord:
        """
        Sell *target_shares* of ts_code.  Capped at current holding.
        """
        lot = self.cfg.lot_size
        held = self.holdings.get(ts_code, 0)
        shares = min((target_shares // lot) * lot, held)

        if shares <= 0:
            rec = TradeRecord(date, ts_code, "SELL", price, 0, 0.0,
                              0.0, "REJECTED", "无持仓可卖")
            self.trade_log.append(rec)
            return rec

        fill_price = self._apply_slippage(price, self.cfg.slippage, "SELL")
        amount = fill_price * shares
        commission = amount * self.cfg.commission_rate

        proceeds = amount - commission
        self.cash += proceeds
        self.holdings[ts_code] -= shares
        if self.holdings[ts_code] == 0:
            del self.holdings[ts_code]
            if ts_code in self.cost_basis:
                del self.cost_basis[ts_code]

        rec = TradeRecord(date, ts_code, "SELL", fill_price, shares,
                          amount, commission, "FILLED")
        self.trade_log.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Rebalance to target weights
    # ------------------------------------------------------------------

    def rebalance(self, date: str,
                  target_weights: Dict[str, float],
                  prices: Dict[str, float]) -> List[TradeRecord]:
        """
        Rebalance portfolio to target_weights.

        Parameters
        ----------
        date : str
            Trade date (for logging).
        target_weights : dict
            {ts_code: weight} where weight in [0,1], sum ≈ 1.
        prices : dict
            {ts_code: close_price} for all relevant stocks on *date*.

        Returns
        -------
        list of TradeRecord
        """
        records: List[TradeRecord] = []

        # Normalise weights
        w_sum = sum(target_weights.values())
        if w_sum <= 0:
            target_weights = {}
        elif abs(w_sum - 1.0) > 1e-8:
            target_weights = {k: v / w_sum for k, v in target_weights.items()}

        # Current portfolio value (at today's prices)
        portfolio_value = self.cash
        for code, shares in self.holdings.items():
            p = prices.get(code)
            if p is not None:
                portfolio_value += p * shares

        # Target shares per stock
        target_shares_map: Dict[str, int] = {}
        lot = self.cfg.lot_size
        for code, w in target_weights.items():
            p = prices.get(code)
            if p is None or p <= 0:
                continue
            target_value = portfolio_value * w
            raw_shares = target_value / p
            target_shares_map[code] = int(raw_shares // lot) * lot

        # ---------- Phase 1: SELL orders (free up cash first) ----------
        # Sell stocks no longer in target, or reduce over-weight positions
        all_codes = set(list(self.holdings.keys()) + list(target_shares_map.keys()))
        for code in list(self.holdings.keys()):
            current = self.holdings.get(code, 0)
            target = target_shares_map.get(code, 0)
            if current > target:
                sell_shares = current - target
                p = prices.get(code)
                if p is not None and sell_shares > 0:
                    rec = self.execute_sell(date, code, sell_shares, p)
                    records.append(rec)

        # ---------- Phase 2: BUY orders ----------
        for code, target in target_shares_map.items():
            current = self.holdings.get(code, 0)
            if target > current:
                buy_shares = target - current
                p = prices.get(code)
                if p is not None and buy_shares > 0:
                    rec = self.execute_buy(date, code, buy_shares, p)
                    records.append(rec)

        return records

    # ------------------------------------------------------------------
    # Portfolio valuation
    # ------------------------------------------------------------------

    def portfolio_value(self, prices: Dict[str, float]) -> float:
        """Return total portfolio value = cash + market value of holdings."""
        mv = self.cash
        for code, shares in self.holdings.items():
            p = prices.get(code)
            if p is not None:
                mv += p * shares
        return mv

    def snapshot(self, date: str, prices: Dict[str, float]) -> dict:
        """Return a snapshot dict of the current portfolio state."""
        mv = self.portfolio_value(prices)
        return {
            "date": date,
            "cash": round(self.cash, 2),
            "market_value": round(mv - self.cash, 2),
            "total_value": round(mv, 2),
            "n_stocks": len(self.holdings),
            "holdings": dict(self.holdings),
        }

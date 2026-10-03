# -*- coding: utf-8 -*-
"""Kiểm tra PositionManager: exit rule priority, entry conditions."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy import PositionManager


def test_hard_stop_triggers_before_other_rules():
    pm = PositionManager(stop_loss_pct=0.08, trail_pct=0.12)
    pm.open_position("ABC", price=100.0, date=pd.Timestamp("2024-01-01"))
    # Giá giảm 10% -> vượt hard stop 8%
    action, reason, size = pm.check_exit("ABC", current_price=90.0, composite_rank=0.9, vol_spike=0.5, mom_6=0.1)
    assert action == "HARD_STOP"
    assert size == 1.0


def test_hold_when_all_conditions_ok():
    pm = PositionManager()
    pm.open_position("ABC", price=100.0, date=pd.Timestamp("2024-01-01"))
    action, reason, size = pm.check_exit("ABC", current_price=105.0, composite_rank=0.9, vol_spike=0.5, mom_6=0.1)
    assert action == "HOLD"
    assert size == 0.0


def test_trailing_stop_uses_peak_not_entry():
    pm = PositionManager(stop_loss_pct=0.5, trail_pct=0.10)  # stop_loss cao để không trigger trước
    pm.open_position("ABC", price=100.0, date=pd.Timestamp("2024-01-01"))
    # Giá lên đỉnh 150 rồi giảm xuống 130 (giảm 13.3% từ đỉnh, nhưng vẫn +30% từ entry)
    pm.check_exit("ABC", current_price=150.0, composite_rank=0.9, vol_spike=0.5, mom_6=0.1)  # cập nhật peak
    action, reason, size = pm.check_exit("ABC", current_price=130.0, composite_rank=0.9, vol_spike=0.5, mom_6=0.1)
    assert action == "TRAILING_STOP"


def test_entry_requires_positive_momentum():
    pm = PositionManager(top_entry_rank=0.5)
    ok, reason = pm.check_entry("ABC", price=100.0, composite_rank=0.9, vol_spike=0.5, mom_6=-0.05)
    assert not ok


def test_entry_requires_low_vol_spike():
    pm = PositionManager(top_entry_rank=0.5, vol_reduce_thr=2.0)
    ok, reason = pm.check_entry("ABC", price=100.0, composite_rank=0.9, vol_spike=2.5, mom_6=0.1)
    assert not ok


def test_entry_ok_when_all_conditions_met():
    pm = PositionManager(top_entry_rank=0.5, vol_reduce_thr=2.0)
    ok, reason = pm.check_entry("ABC", price=100.0, composite_rank=0.9, vol_spike=0.5, mom_6=0.1)
    assert ok


def test_close_position_partial_reduces_size():
    pm = PositionManager()
    pm.open_position("ABC", price=100.0, date=pd.Timestamp("2024-01-01"))
    pm.close_position("ABC", size_to_sell=0.5)
    assert pm.positions["ABC"]["size"] == 0.5


def test_close_position_full_removes_from_positions():
    pm = PositionManager()
    pm.open_position("ABC", price=100.0, date=pd.Timestamp("2024-01-01"))
    pm.close_position("ABC")
    assert "ABC" not in pm.positions

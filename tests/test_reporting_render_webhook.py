# -*- coding: utf-8 -*-
"""Test dữ liệu báo cáo (src/reporting.py), renderer PNG/HTML (src/report_render.py) và webhook Discord (src/discord_webhook.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import backtest_with_exits
from src.discord_webhook import WebhookError, build_embed, send_report, send_text
from src.pipeline import ENGINE_PARAMS, build_strategy_inputs
from src.report_render import render_html, render_png
from src.reporting import build_report, check_ledger, diff_orders, extract_orders, positions_asof, positions_signature

URL = "https://discord.com/api/webhooks/42/TOPSECRET"


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(3)
    n, m = 900, 24
    syms = [f"S{i:02d}" for i in range(m)]
    idx = pd.bdate_range("2022-01-03", periods=n)
    drift = np.zeros((n, m))
    for t in range(1, n):
        drift[t] = 0.985 * drift[t - 1] + rng.normal(0, 0.0005, m)
    price = pd.DataFrame(25 * np.cumprod(1 + drift + rng.normal(0, 0.017, (n, m)), 0), idx, syms)
    vol = pd.DataFrame(rng.lognormal(13, .4, (n, m)), idx, syms)
    inp = build_strategy_inputs(price, vol)
    _, trades = backtest_with_exits(price, inp["comp_score"], inp["vol_spike"], inp["mom_6"], cost_rate=0.0023, **ENGINE_PARAMS)
    cnt = trades.groupby("date").size()
    asof = cnt[cnt.index > idx[-120]].idxmax()                              # ngày gần cuối có nhiều lệnh nhất
    cut = lambda df: df.loc[:asof]
    port, trades = backtest_with_exits(cut(price), cut(inp["comp_score"]), cut(inp["vol_spike"]), cut(inp["mom_6"]), cost_rate=0.0023, **ENGINE_PARAMS)
    return dict(price=cut(price), inp={k: (cut(v) if isinstance(v, pd.DataFrame) else v) for k, v in inp.items()}, port=port, trades=trades, asof=asof)


def make_report(w, mode="final", **kw):
    i = w["inp"]
    return build_report(price_df=w["price"], comp_score=i["comp_score"], vol_spike=i["vol_spike"], mom_6=i["mom_6"], port_df=w["port"],
                        trade_df=w["trades"], asof=w["asof"], mode=mode, params=ENGINE_PARAMS, cost_rate=0.0023, generated_at="15:02 06/10/2026", **kw)


# ------------------------------------------------------------------ reporting
def test_report_orders_and_positions_match_engine(world):
    rep = make_report(world)
    t = world["trades"][world["trades"]["date"] == world["asof"]]
    assert len(rep["orders"]) == len(t) > 0
    assert {(o["side"] == "BUY", o["symbol"]) for o in rep["orders"]} == {(a == "BUY", s) for a, s in zip(t["action"], t["symbol"])}
    open_pos = world["port"].attrs["open_positions"]
    assert {p["symbol"] for p in rep["positions"]} == set(open_pos)
    for p in rep["positions"]:
        e = open_pos[p["symbol"]]
        assert p["stop_price"] == pytest.approx(e["entry_price"] * (1 - 0.08), abs=0.01)
        assert p["trail_price"] == pytest.approx(max(e["peak_price"], p["last_price"]) * (1 - 0.12), abs=0.01)
    assert sum(p["weight"] for p in rep["positions"]) <= 1 + 1e-9
    assert 0 <= rep["meta"]["sessions_to_rebalance"] <= ENGINE_PARAMS["rebal_freq"]
    json.dumps(rep)                                                         # phải JSON-serializable


def test_day_return_in_report_equals_engine_row(world):
    rep = make_report(world)
    row = world["port"].loc[world["asof"]]
    assert rep["portfolio"]["day_net"] == pytest.approx(row["port_return"], abs=1e-5)
    assert rep["portfolio"]["day_cost"] == pytest.approx(row["cost"], abs=1e-5)


def test_diff_orders_and_flip_warning(world):
    rep = make_report(world)
    o = rep["orders"]
    d = diff_orders(o[:-1], o + [{"side": "SELL", "symbol": "ZZZ"}])
    assert [(x["side"], x["symbol"]) for x in d["added"]] == [(o[-1]["side"], o[-1]["symbol"]), ("SELL", "ZZZ")]
    assert diff_orders(o, o)["added"] == [] and diff_orders(o, o)["removed"] == []
    assert diff_orders(o, [])["removed"] == o
    both = {x["symbol"] for x in o if x["side"] != "BUY"} & {x["symbol"] for x in o if x["side"] == "BUY"}
    assert bool(both) == any("cùng phiên" in w for w in rep["meta"]["warnings"])


def test_ledger_roundtrip_and_mismatch(world):
    i = world["inp"]
    sig = positions_asof(world["price"], i["comp_score"], i["vol_spike"], i["mom_6"], ENGINE_PARAMS, 0.0023, world["asof"])
    assert sig == positions_signature(world["port"].attrs["open_positions"])
    ledger = {"asof": str(world["asof"].date()), "positions": sig}
    assert check_ledger(ledger, sig, world["asof"]) == []
    assert check_ledger(None, sig, world["asof"]) == [] and check_ledger({"asof": "1999-01-01", "positions": {}}, sig, world["asof"]) == []
    bad = json.loads(json.dumps(sig)); s0 = next(iter(bad)); bad[s0]["entry_price"] *= 1.1; bad["XXX"] = {"entry_date": "2020-01-01", "entry_price": 1, "size": 1}
    msgs = check_ledger({"asof": ledger["asof"], "positions": bad}, sig, world["asof"])
    assert any("XXX" in m for m in msgs) and any(s0 in m for m in msgs)


# ------------------------------------------------------------------ render
def test_png_and_html_render(world, tmp_path):
    rep = make_report(world, preview_orders=[], warnings=["thử cảnh báo <b>"])
    png = render_png(rep, tmp_path / "r.png")
    head = png.read_bytes()[:8]
    assert head == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 50_000
    from PIL import Image
    w, h = Image.open(png).size
    assert w == 1430 and h > 1200
    html = render_html(rep)
    assert "ĐÃ CHỐT" in html and "<svg" in html and "http://" not in html and "https://" not in html and "<script" not in html   # offline, không JS
    assert "&lt;b&gt;" in html                                              # nội dung được escape
    for o in rep["orders"]:
        assert o["symbol"] in html


def test_preview_badge_and_empty_orders_message(world, tmp_path):
    rep = make_report(world, mode="preview")
    rep["orders"] = []
    html = render_html(rep)
    assert "TẠM TÍNH" in html and "Không có lệnh nào tính đến thời điểm này" in html and "KHỐI LƯỢNG chưa đủ phiên" in html
    assert render_png(rep, tmp_path / "p.png").stat().st_size > 50_000


# ------------------------------------------------------------------ webhook
class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._b, self.text = code, body or {}, "x"

    def json(self):
        return self._b


def test_embed_content(world):
    e = build_embed(make_report(world), "a.png")
    assert e["image"]["url"] == "attachment://a.png" and e["title"].startswith("📊") and len(json.dumps(e)) < 6000
    assert all(len(f["value"]) <= 1024 for f in e["fields"])
    names = " ".join(f["name"] for f in e["fields"])
    assert "MUA" in names or "BÁN" in names


def test_send_report_payload_and_files(world, tmp_path):
    rep = make_report(world)
    png, html = render_png(rep, tmp_path / "x.png"), tmp_path / "x.html"
    html.write_text(render_html(rep), encoding="utf-8")
    seen = {}

    def post(url, params=None, data=None, files=None, timeout=None, headers=None):
        seen.update(url=url, params=params, payload=json.loads(data["payload_json"]), files=files)
        return Resp(200)

    send_report(URL, rep, png, html, mention="<@1>", post=post)
    assert seen["params"] == {"wait": "true"} and seen["url"] == URL
    assert [a["filename"] for a in seen["payload"]["attachments"]] == ["x.png", "x.html"]
    assert seen["payload"]["content"] == "<@1>" and seen["payload"]["allowed_mentions"]["parse"] == ["users", "roles"]
    assert seen["files"]["files[0]"][2] == "image/png" and seen["files"]["files[1]"][2] == "text/html"


def test_retry_on_429_then_success_and_5xx_exhaustion():
    sleeps, responses = [], [Resp(429, {"retry_after": 1.5}), Resp(502), Resp(200)]
    send_text(URL, "hi", post=lambda *a, **k: responses.pop(0), sleep=sleeps.append)
    assert sleeps[0] == 1.5 and len(sleeps) == 2 and responses == []
    with pytest.raises(WebhookError, match="sau 3 lần thử"):
        send_text(URL, "hi", post=lambda *a, **k: Resp(500), sleep=lambda s: None)


def test_4xx_and_network_errors_never_leak_the_webhook_url():
    with pytest.raises(WebhookError) as e:
        send_text(URL, "hi", post=lambda *a, **k: Resp(401), sleep=lambda s: None)
    assert "TOPSECRET" not in str(e.value)

    def boom(*a, **k):
        raise requests.ConnectionError(f"failed to connect {URL}")
    with pytest.raises(WebhookError) as e2:
        send_text(URL, "hi", post=boom, sleep=lambda s: None)
    assert "TOPSECRET" not in str(e2.value)


def test_invalid_webhook_url_rejected():
    with pytest.raises(ValueError, match="không hợp lệ"):
        send_text("https://example.com/hook", "hi", post=lambda *a, **k: Resp(200))
    with pytest.raises(ValueError):
        send_text("", "hi")

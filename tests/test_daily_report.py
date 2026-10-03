# -*- coding: utf-8 -*-
"""Kiểm thử end-to-end `daily_report.run` bằng dữ liệu GIẢ LẬP + Discord GIẢ (không cần mạng, không cần webhook thật)."""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import daily_report as dr
from src.data_loader import VN30_UNIVERSE

FAKE_URL = "https://discord.com/api/webhooks/123456/SECRET_TOKEN_abc"
TODAY = pd.Timestamp("2026-10-06")                       # thứ Ba
NOW = datetime(2026, 10, 6, 13, 55, tzinfo=dr.VN_TZ)


class FakeResp:
    def __init__(self, code=200, body=None):
        self.status_code, self._b, self.text = code, body or {}, json.dumps(body or {})

    def json(self):
        return self._b


class Recorder:
    """Thay requests.post: ghi lại mọi lời gọi."""
    def __init__(self, fail_channel_with=None):
        self.calls, self.fail_channel_with = [], fail_channel_with

    def __call__(self, url, params=None, data=None, files=None, timeout=None, headers=None):
        self.calls.append({"url": url, "payload": json.loads(data["payload_json"]), "files": files or {}, "headers": headers or {}})
        if self.fail_channel_with and "/channels/" in url:
            return FakeResp(self.fail_channel_with, {"message": "Missing Access"})
        return FakeResp(200, {"id": "1"})


def make_loader(end=TODAY, n=900, drop_last=False, missing=(), calls=None):
    def loader(symbols, start, end_str, fields=("close", "volume")):
        if calls is not None:
            calls.append(list(symbols))
        idx = pd.bdate_range(end=end - (pd.offsets.BDay(1) if drop_last else pd.Timedelta(0)), periods=n)
        close, vol = {}, {}
        for s in symbols:
            if s in missing:
                continue
            rng = np.random.default_rng(abs(hash(s)) % (2 ** 32) if False else sum(map(ord, s)))
            drift = np.zeros(n)
            for t in range(1, n):
                drift[t] = 0.985 * drift[t - 1] + rng.normal(0, 0.0005)
            px = 25 * np.cumprod(1 + drift + rng.normal(0, 0.017, n))
            if s in ("TCX", "VPL"):
                px[:500] = np.nan                           # coverage thấp → bị loại như dữ liệu thật
            close[s] = pd.Series(px, idx)
            vol[s] = pd.Series(rng.lognormal(13, 0.4, n), idx)
        return {"close": pd.DataFrame(close), "volume": pd.DataFrame(vol)}
    return loader


def args_for(tmp, mode, *extra):
    return dr.parse_args(["--mode", mode, "--state-dir", str(tmp / "state"), "--out-dir", str(tmp / "out"), *extra])


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", FAKE_URL)


def test_wait_until_sleeps_exactly_until_target_and_skips_when_too_early():
    slept = []
    assert dr.wait_until("14:00", 45, now_fn=lambda: NOW, sleep_fn=slept.append) == pytest.approx(300)
    assert slept == [pytest.approx(300)]
    slept.clear()
    early = datetime(2026, 10, 6, 9, 0, tzinfo=dr.VN_TZ)                     # sớm 5 giờ > 45 phút → không chờ
    assert dr.wait_until("14:00", 45, now_fn=lambda: early, sleep_fn=slept.append) == 0 and slept == []
    late = datetime(2026, 10, 6, 14, 20, tzinfo=dr.VN_TZ)                    # trễ giờ hẹn → chạy ngay
    assert dr.wait_until("14:00", 45, now_fn=lambda: late, sleep_fn=slept.append) == 0 and slept == []


def test_preview_then_final_full_flow(tmp_path):
    rec, slept = Recorder(), []
    assert dr.run(args_for(tmp_path, "preview"), loader=make_loader(), now_fn=lambda: NOW, sleep_fn=slept.append, post=rec) == 0
    assert slept and slept[0] == pytest.approx(300)                            # đã chờ tới 14:00
    assert len(rec.calls) == 1
    c = rec.calls[0]
    assert set(c["files"]) == {"files[0]", "files[1]"}
    assert c["files"]["files[0]"][0].endswith("_preview.png") and c["files"]["files[0]"][1][:8] == b"\x89PNG\r\n\x1a\n"
    assert c["files"]["files[1]"][0].endswith("_preview.html") and b"<!DOCTYPE html>" in c["files"]["files[1]"][1]
    assert c["payload"]["embeds"][0]["title"].startswith("⏱ Lệnh dự kiến 14:00")
    assert (tmp_path / "state" / "preview.json").exists() and (tmp_path / "state" / "ic_weights.json").exists()
    assert not (tmp_path / "state" / "ledger.json").exists()                   # preview KHÔNG ghi sổ lệnh

    # --- 15:00: báo cáo cuối phiên, có so sánh với 14:00
    now2 = datetime(2026, 10, 6, 15, 1, tzinfo=dr.VN_TZ)
    assert dr.run(args_for(tmp_path, "final"), loader=make_loader(), now_fn=lambda: now2, sleep_fn=slept.append, post=rec) == 0
    assert len(rec.calls) == 2 and rec.calls[1]["payload"]["embeds"][0]["title"].startswith("📊 Báo cáo cuối phiên 15:00")
    rep = json.loads(next((tmp_path / "out").glob("*_final.json")).read_text(encoding="utf-8"))
    assert rep["vs_preview"] is not None and rep["meta"]["is_partial"] is False
    ledger = json.loads((tmp_path / "state" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["asof"] == "2026-10-06" and set(ledger["positions"]) == {p["symbol"] for p in rep["positions"]}
    rows = list(csv.DictReader((tmp_path / "state" / "daily_log.csv").open(encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["date"] == "2026-10-06"

    # --- chạy lại cùng ngày: không nhân đôi dòng nhật ký, không có cảnh báo sổ lệnh (lịch sử tính lại khớp sổ)
    assert dr.run(args_for(tmp_path, "final"), loader=make_loader(), now_fn=lambda: now2, sleep_fn=slept.append, post=rec) == 0
    assert len(list(csv.DictReader((tmp_path / "state" / "daily_log.csv").open(encoding="utf-8")))) == 1
    rep2 = json.loads(next((tmp_path / "out").glob("*_final.json")).read_text(encoding="utf-8"))
    assert not any("Sổ lệnh lệch" in w for w in rep2["meta"]["warnings"])
    # URL webhook không bao giờ nằm trong payload
    assert all("SECRET_TOKEN" not in json.dumps(c["payload"]) for c in rec.calls)


def test_weekend_is_skipped_silently(tmp_path):
    rec = Recorder()
    sat = datetime(2026, 10, 3, 14, 0, tzinfo=dr.VN_TZ)
    assert dr.run(args_for(tmp_path, "final"), loader=make_loader(), now_fn=lambda: sat, sleep_fn=lambda s: None, post=rec) == 0
    assert rec.calls == []


def test_missing_today_bar_sends_notice_only(tmp_path):
    rec = Recorder()
    assert dr.run(args_for(tmp_path, "final"), loader=make_loader(drop_last=True), now_fn=lambda: NOW, sleep_fn=lambda s: None, post=rec) == 0
    assert len(rec.calls) == 1 and rec.calls[0]["files"] == {} and "Chưa có nến" in rec.calls[0]["payload"]["content"]
    rec2 = Recorder()
    dr.run(args_for(tmp_path, "final", "--no-notice"), loader=make_loader(drop_last=True), now_fn=lambda: NOW, sleep_fn=lambda s: None, post=rec2)
    assert rec2.calls == []


def test_fetch_retries_missing_symbols_then_aborts(tmp_path):
    calls = []
    with pytest.raises(RuntimeError, match="Không tải được dữ liệu của: ACB"):
        dr.run(args_for(tmp_path, "final", "--fetch-retries", "2"), loader=make_loader(missing=("ACB",), calls=calls),
               now_fn=lambda: NOW, sleep_fn=lambda s: None, post=Recorder())
    assert calls[0] == VN30_UNIVERSE and calls[1:] == [["ACB"], ["ACB"]]       # lần đầu cả universe, sau đó chỉ tải lại mã thiếu


def test_universe_change_aborts_unless_accepted(tmp_path):
    kw = dict(now_fn=lambda: NOW, sleep_fn=lambda s: None, post=Recorder())
    dr.run(args_for(tmp_path, "preview"), loader=make_loader(), **kw)
    shrunk = [s for s in VN30_UNIVERSE if s != "FPT"]
    with pytest.raises(RuntimeError, match="Universe thay đổi"):
        dr.run(args_for(tmp_path, "preview", "--universe", *shrunk), loader=make_loader(), **kw)
    assert dr.run(args_for(tmp_path, "preview", "--universe", *shrunk, "--accept-universe-change"), loader=make_loader(), **kw) == 0


def test_ledger_mismatch_is_reported_as_warning(tmp_path):
    kw = dict(now_fn=lambda: NOW, sleep_fn=lambda s: None, post=Recorder())
    dr.run(args_for(tmp_path, "final"), loader=make_loader(), **kw)
    led_path = tmp_path / "state" / "ledger.json"
    led = json.loads(led_path.read_text(encoding="utf-8"))
    led["asof"] = "2026-10-05"                                                  # giả vờ sổ lệnh là của phiên liền trước…
    sym = next(iter(led["positions"]))
    led["positions"][sym]["entry_price"] *= 1.2                                  # …và có 1 vị thế sai giá vốn
    led_path.write_text(json.dumps(led), encoding="utf-8")
    dr.run(args_for(tmp_path, "final"), loader=make_loader(), **kw)
    rep = json.loads(next((tmp_path / "out").glob("*_final.json")).read_text(encoding="utf-8"))
    assert any("Sổ lệnh lệch" in w and sym in w for w in rep["meta"]["warnings"])


def test_no_webhook_env_skips_sending_but_writes_files(tmp_path, monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL")
    rec = Recorder()
    assert dr.run(args_for(tmp_path, "final"), loader=make_loader(), now_fn=lambda: NOW, sleep_fn=lambda s: None, post=rec) == 0
    assert rec.calls == [] and list((tmp_path / "out").glob("*.png"))


def test_main_reports_failure_to_discord_without_leaking_url(monkeypatch):
    sent = []
    monkeypatch.setattr(dr, "run", lambda args, **kw: (_ for _ in ()).throw(RuntimeError("Không tải được dữ liệu của: ACB")))
    monkeypatch.setattr(dr, "send_text", lambda url, text, **kw: sent.append((url, text)))
    with pytest.raises(RuntimeError):
        dr.main(["--mode", "final"])
    assert len(sent) == 1 and "❌ Báo cáo final lỗi" in sent[0][1] and "ACB" in sent[0][1] and "SECRET_TOKEN" not in sent[0][1]


# ------------------------------------------------------------------ theo yêu cầu (slash command /report)
CH, USER = "123456789012345678", "987654321098765432"


def od_args(tmp, *extra):
    return dr.parse_args(["--mode", "auto", "--on-demand", "--channel-id", CH, "--requester-id", USER,
                          "--state-dir", str(tmp / "state"), "--out-dir", str(tmp / "out"), *extra])


@pytest.fixture
def bot_env(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "BOT_SECRET_TOKEN")


def read_report(tmp):
    return json.loads(next((tmp / "out").glob("*.json")).read_text(encoding="utf-8"))


def test_on_demand_in_session_is_preview_posts_to_channel_and_never_touches_state(tmp_path, bot_env):
    rec, slept = Recorder(), []
    now = datetime(2026, 10, 6, 11, 23, tzinfo=dr.VN_TZ)
    assert dr.run(od_args(tmp_path), loader=make_loader(), now_fn=lambda: now, sleep_fn=slept.append, post=rec) == 0
    assert slept == []                                                          # không chờ giờ hẹn
    c = rec.calls[0]
    assert c["url"] == f"https://discord.com/api/v10/channels/{CH}/messages"
    assert c["headers"]["Authorization"] == "Bot BOT_SECRET_TOKEN"
    assert c["payload"]["content"].startswith(f"<@{USER}>") and c["payload"]["allowed_mentions"] == {"parse": [], "users": [USER]}
    title = c["payload"]["embeds"][0]["title"]
    assert "Lệnh tạm tính lúc 11:23" in title and "(theo yêu cầu)" in title
    assert set(c["files"]) == {"files[0]", "files[1]"} and "_ondemand" in c["files"]["files[0]"][0]
    assert read_report(tmp_path)["meta"]["trigger"] == "on_demand" and read_report(tmp_path)["meta"]["is_partial"] is True
    assert not (tmp_path / "state").exists()                                    # KHÔNG ghi state/ (kể cả trọng số IC)
    assert any("Chưa có trọng số IC đóng băng" in w for w in read_report(tmp_path)["meta"]["warnings"])


def test_on_demand_after_close_is_final_and_has_no_preview_diff(tmp_path, bot_env):
    rec = Recorder()
    now = datetime(2026, 10, 6, 16, 10, tzinfo=dr.VN_TZ)
    dr.run(od_args(tmp_path), loader=make_loader(), now_fn=lambda: now, sleep_fn=lambda s: None, post=rec)
    rep = read_report(tmp_path)
    assert rep["meta"]["mode"] == "final" and rep["meta"]["is_partial"] is False and rep["vs_preview"] is None
    assert rec.calls[0]["payload"]["embeds"][0]["title"].startswith("📊 Báo cáo cuối phiên (theo yêu cầu)")


def test_on_demand_on_weekend_reports_last_session_with_warning(tmp_path, bot_env):
    rec = Recorder()
    sat = datetime(2026, 10, 3, 10, 0, tzinfo=dr.VN_TZ)
    assert dr.run(od_args(tmp_path), loader=make_loader(end=pd.Timestamp("2026-10-02")), now_fn=lambda: sat, sleep_fn=lambda s: None, post=rec) == 0
    rep = read_report(tmp_path)
    assert rep["meta"]["asof"] == "2026-10-02" and rep["meta"]["mode"] == "final"
    assert any("phiên gần nhất 02/10/2026" in w for w in rep["meta"]["warnings"])


def test_on_demand_past_date_and_non_session_date(tmp_path, bot_env):
    rec = Recorder()
    now = datetime(2026, 10, 6, 10, 0, tzinfo=dr.VN_TZ)
    dr.run(od_args(tmp_path, "--asof", "2026-09-25"), loader=make_loader(), now_fn=lambda: now, sleep_fn=lambda s: None, post=rec)
    rep = read_report(tmp_path)
    assert rep["meta"]["asof"] == "2026-09-25" and rep["meta"]["mode"] == "final" and rep["vs_preview"] is None
    assert "2026-09-25_final_ondemand" in rec.calls[0]["files"]["files[0]"][0]
    # ngày thứ Bảy → không có phiên → chỉ trả lời 1 dòng cho người dùng (luôn gửi, kể cả --no-notice)
    rec2 = Recorder()
    dr.run(od_args(tmp_path, "--asof", "2026-09-26", "--no-notice"), loader=make_loader(), now_fn=lambda: now, sleep_fn=lambda s: None, post=rec2)
    assert len(rec2.calls) == 1 and rec2.calls[0]["files"] == {} and "không có phiên giao dịch" in rec2.calls[0]["payload"]["content"]
    assert rec2.calls[0]["payload"]["content"].startswith(f"<@{USER}>")


def test_channel_403_falls_back_to_default_webhook(tmp_path, bot_env):
    rec = Recorder(fail_channel_with=403)
    now = datetime(2026, 10, 6, 16, 10, tzinfo=dr.VN_TZ)
    assert dr.run(od_args(tmp_path), loader=make_loader(), now_fn=lambda: now, sleep_fn=lambda s: None, post=rec) == 0
    assert "/channels/" in rec.calls[0]["url"] and rec.calls[1]["url"] == FAKE_URL                  # thử kênh trước, rồi webhook
    assert set(rec.calls[1]["files"]) == {"files[0]", "files[1]"} and rec.calls[1]["payload"]["content"] == f"<@{USER}>"


def test_cli_rejects_bad_ids_dates_and_auto_without_on_demand():
    base = ["--mode", "auto", "--on-demand"]
    for bad in (["--channel-id", "abc"], ["--requester-id", "12;rm -rf"], ["--asof", "2026/10/02"], ["--asof", "02-10-2026"]):
        with pytest.raises(SystemExit):
            dr.parse_args(base + bad)
    with pytest.raises(SystemExit):
        dr.parse_args(["--mode", "auto"])


def test_on_demand_failure_is_reported_to_the_requesting_channel(monkeypatch, bot_env):
    sent = []
    monkeypatch.setattr(dr, "run", lambda args, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(dr, "send_text_channel", lambda token, ch, text, **kw: sent.append((token, ch, text, kw.get("requester_id"))))
    with pytest.raises(RuntimeError):
        dr.main(["--mode", "auto", "--on-demand", "--channel-id", CH, "--requester-id", USER])
    assert sent and sent[0][1] == CH and sent[0][3] == USER and "boom" in sent[0][2] and "BOT_SECRET" not in sent[0][2]


def test_on_demand_uses_frozen_weights_from_scheduled_run_without_warning(tmp_path, bot_env):
    kw = dict(loader=make_loader(), sleep_fn=lambda s: None, post=Recorder())
    dr.run(args_for(tmp_path, "final"), now_fn=lambda: NOW, **kw)                       # lần chạy theo lịch tạo state/
    before = {p.name: p.read_text(encoding="utf-8") for p in (tmp_path / "state").iterdir()}
    now = datetime(2026, 10, 6, 11, 0, tzinfo=dr.VN_TZ)
    dr.run(od_args(tmp_path), now_fn=lambda: now, **kw)
    rep = json.loads(next((tmp_path / "out").glob("*_ondemand.json")).read_text(encoding="utf-8"))
    assert not any("Chưa có trọng số IC" in w for w in rep["meta"]["warnings"])
    assert {p.name: p.read_text(encoding="utf-8") for p in (tmp_path / "state").iterdir()} == before   # theo yêu cầu KHÔNG đổi state/

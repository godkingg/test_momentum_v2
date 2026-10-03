# -*- coding: utf-8 -*-
"""Hạ tầng cho slash command: định nghĩa lệnh hợp lệ với Discord, workflow YAML, gửi vào kênh bằng bot, và test JS của Worker."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import register_discord_commands as reg
from src.discord_webhook import WebhookError, send_report_channel, send_text_channel


def test_command_definition_satisfies_discord_limits():
    assert len(reg.COMMANDS) == 1
    c = reg.COMMANDS[0]
    assert re.fullmatch(r"[a-z0-9_-]{1,32}", c["name"]) and c["name"] == "report" and 1 <= len(c["description"]) <= 100
    assert c["dm_permission"] is False
    for o in c["options"]:
        assert re.fullmatch(r"[a-z0-9_-]{1,32}", o["name"]) and 1 <= len(o["description"]) <= 100 and o["type"] == 3


def test_register_script_calls_right_endpoint_and_never_prints_token(monkeypatch, capsys):
    seen = {}

    class R:
        status_code = 200
        text = "ok"
        def json(self): return reg.COMMANDS

    monkeypatch.setenv("DISCORD_BOT_TOKEN", "BOT_SECRET")
    monkeypatch.setattr(reg.requests, "put", lambda url, json=None, headers=None, timeout=None: (seen.update(url=url, headers=headers, body=json) or R()))
    assert reg.main(["--app-id", "111", "--guild-id", "222"]) == 0
    assert seen["url"] == "https://discord.com/api/v10/applications/111/guilds/222/commands" and seen["headers"]["Authorization"] == "Bot BOT_SECRET"
    assert "BOT_SECRET" not in capsys.readouterr().out
    assert reg.endpoint("111", None) == "https://discord.com/api/v10/applications/111/commands"


def test_workflows_are_valid_and_pass_inputs_via_env_not_inline():
    for name, expect_cron in (("daily_report.yml", True), ("report_on_demand.yml", False)):
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        w = yaml.safe_load(text)
        on = w.get(True, w.get("on"))
        assert ("schedule" in on) == expect_cron and "workflow_dispatch" in on
        # chống command injection: ${{ inputs.* }} chỉ được xuất hiện ở khối env:, không nằm trong lệnh run:
        for step in w["jobs"]["report"]["steps"]:
            if "run" in step:
                assert "${{ inputs." not in step["run"] and "${{ github.event.inputs" not in step["run"], f"{name}: inputs nằm trong run:"
    od = yaml.safe_load((ROOT / ".github" / "workflows" / "report_on_demand.yml").read_text(encoding="utf-8"))
    assert od["permissions"] == {"contents": "read"}                      # báo cáo theo yêu cầu không được ghi repo
    assert set(od[True]["workflow_dispatch"]["inputs"]) == {"channel_id", "requester_id", "asof"}


class _Resp:
    def __init__(self, code=200):
        self.status_code, self.text = code, "x"
    def json(self): return {}


def test_send_to_channel_uses_bot_auth_and_never_leaks_token(tmp_path):
    png = tmp_path / "a.png"; png.write_bytes(b"\x89PNG")
    seen = {}
    report = {"meta": {"mode": "final", "trigger": "on_demand", "asof": "2026-10-02", "generated_at": "10:00 03/10/2026", "is_partial": False, "warnings": []},
              "portfolio": {"day_net": 0.01, "day_bench": 0.0, "cum_net": 0.1, "cum_bench": 0.05, "n_positions": 0, "exposure": 0.0}, "orders": [], "positions": []}
    send_report_channel("BOT_SECRET", "123456789012345678", report, png, requester_id="987654321098765432",
                        post=lambda url, **kw: (seen.update(url=url, **kw) or _Resp()))
    assert seen["url"].endswith("/channels/123456789012345678/messages") and seen["headers"]["Authorization"] == "Bot BOT_SECRET"
    assert "params" not in seen and "BOT_SECRET" not in seen["data"]["payload_json"]
    with pytest.raises(WebhookError) as e:
        send_text_channel("BOT_SECRET", "123456789012345678", "x", post=lambda url, **kw: _Resp(403), sleep=lambda s: None)
    assert e.value.status == 403 and "BOT_SECRET" not in str(e.value)
    for bad in ("12", "abc", "1234567890123456; DROP"):
        with pytest.raises(ValueError):
            send_text_channel("t", bad, "x", post=lambda *a, **k: _Resp())
    with pytest.raises(ValueError, match="DISCORD_BOT_TOKEN"):
        send_text_channel("", "123456789012345678", "x", post=lambda *a, **k: _Resp())


@pytest.mark.skipif(shutil.which("node") is None, reason="cần Node ≥ 20 để chạy test của Cloudflare Worker")
def test_cloudflare_worker_js_suite():
    r = subprocess.run(["node", "--test", "test/worker.test.mjs"], cwd=ROOT / "worker", capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-1500:]

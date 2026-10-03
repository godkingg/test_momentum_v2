# -*- coding: utf-8 -*-
"""report_render.py — Xuất báo cáo hằng ngày ra PNG (matplotlib, không cần trình duyệt) và HTML độc lập (CSS + SVG inline).

Phong cách dark giống dashboard /cocau của bot cũ (nền #0d1117, xanh #26a69a = tăng, đỏ #ef5350 = giảm).
PNG chạy được trên mọi môi trường (GitHub Actions, Colab, máy local) vì chỉ dùng matplotlib + font DejaVu (có đủ dấu tiếng Việt).
"""
from __future__ import annotations

import html
from pathlib import Path

C = {"bg": "#0d1117", "panel": "#161b22", "panel2": "#1c2330", "border": "#30363d", "text": "#e6edf3", "muted": "#8b949e",
     "green": "#26a69a", "red": "#ef5350", "blue": "#58a6ff", "pink": "#fea7e9", "amber": "#d29922"}
SIDE_COLOR = {"BUY": C["green"], "SELL": C["red"], "REDUCE": C["amber"]}
SIDE_TEXT = {"BUY": "MUA", "SELL": "BÁN", "REDUCE": "GIẢM"}
MODE_TITLE = {"preview": "LỆNH DỰ KIẾN — 14:00 (nến chưa chốt)", "final": "BÁO CÁO CUỐI PHIÊN — 15:00"}
DISCLAIMER = ("Danh mục MÔ HÌNH (paper) · giá khớp giả định = close · tỷ trọng equal-weight · cost 0.23%/lệnh · "
              "KHÔNG phải khuyến nghị đầu tư.")


# ----------------------------------------------------------------------------- định dạng
def mode_title(report: dict) -> str:
    """Tiêu đề: lịch cố định (14:00/15:00) hay theo yêu cầu (giờ bất kỳ)."""
    m = report["meta"]
    if m.get("trigger") == "on_demand":
        return f"LỆNH TẠM TÍNH — {m['generated_at'][:5]}" if m["mode"] == "preview" else "BÁO CÁO CUỐI PHIÊN"
    return MODE_TITLE[m["mode"]]


def pct(x, nd: int = 2, sign: bool = True) -> str:
    return "—" if x is None else f"{x * 100:{'+' if sign else ''}.{nd}f}%"


def num(x, nd: int = 2) -> str:
    return "—" if x is None else f"{x:,.{nd}f}"


def tone(x) -> str:
    return C["muted"] if x is None or x == 0 else (C["green"] if x > 0 else C["red"])


def vn_date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{d}/{m}/{y}"


def no_order_reason(report: dict) -> str:
    m = report["meta"]
    base = "Không có lệnh nào" + (" tính đến thời điểm này" if m["is_partial"] else " trong phiên")
    why = ("không phải ngày rebalance (lần tới sau {n} phiên) và không có exit rule nào kích hoạt"
           if not m["is_rebalance_day"] else "hôm nay là ngày rebalance nhưng không mã nào đủ điều kiện vào lệnh / không có exit")
    return f"{base}: {why.format(n=m['sessions_to_rebalance'])}."


def kpis(report: dict) -> list[dict]:
    p, m = report["portfolio"], report["meta"]
    return [
        {"label": "Return phiên (net, tạm tính)" if m["is_partial"] else "Return phiên (net)", "value": pct(p["day_net"]), "color": tone(p["day_net"]),
         "sub": f"1/N: {pct(p['day_bench'])} · chênh {pct(p['day_net'] - p['day_bench'])}"},
        {"label": "5 phiên gần nhất", "value": pct(p["ret_5d"]), "color": tone(p["ret_5d"]), "sub": f"1/N: {pct(p['bench_5d'])}"},
        {"label": "Danh mục mô hình", "value": f"{p['n_positions']} vị thế", "color": C["blue"],
         "sub": f"Đầu tư {p['exposure']:.0%} · tiền mặt {1 - p['exposure']:.0%}"},
        {"label": "Lũy kế mô phỏng", "value": pct(p["cum_net"], 1), "color": tone(p["cum_net"]),
         "sub": f"1/N {pct(p['cum_bench'], 1)} · DD hiện tại {pct(p['drawdown'], 1, False)}"},
    ]


def table_specs(report: dict) -> dict:
    """Định nghĩa bảng dùng chung cho PNG và HTML: {tên: (tiêu đề, cột, hàng)}; cột = (header, độ rộng tương đối, căn lề)."""
    orders = [{
        "cells": [(SIDE_TEXT[o["side"]], SIDE_COLOR[o["side"]], True), (o["symbol"], C["text"], True), (num(o["price"]), C["text"], False),
                  (o["size_text"], C["muted"], False), (pct(o["pnl"]) if o["pnl"] is not None else "—", tone(o["pnl"]), False),
                  (f"{o['hold_days']} ngày" if o["hold_days"] is not None else "—", C["muted"], False),
                  (num(o["rank"], 2), C["muted"], False), (f"{o['label']} — {o['reason']}", C["text"], False)],
    } for o in report["orders"]]
    positions = [{
        "cells": [(p["symbol"], C["text"], True), (vn_date(p["entry_date"])[:5], C["muted"], False), (num(p["entry_price"]), C["text"], False),
                  (num(p["last_price"]), C["text"], False), (pct(p["pnl"]), tone(p["pnl"]), True), (pct(p["day_ret"]), tone(p["day_ret"]), False),
                  (pct(p["weight"], 1, False), C["blue"], False), (num(p["stop_price"]), C["red"], False),
                  (num(p["trail_price"]), C["amber"], False), (num(p["rank"], 2), C["muted"], False)],
    } for p in report["positions"]]
    watch = [{
        "cells": [(w["symbol"], C["text"], True), (num(w["rank"], 2), C["text"], False), (pct(w["mom_6"], 1), tone(w["mom_6"]), False),
                  (f"{w['vol_spike']:.2f}x", C["amber"] if w["vol_spike"] >= 2 else C["muted"], False), (num(w["price"]), C["text"], False),
                  (w["status"], C["green"] if w["status"].startswith(("Đã mua", "Đang giữ")) else C["muted"], False)],
    } for w in report["watchlist"]]
    return {
        "orders": (f"LỆNH {'TẠM TÍNH ' if report['meta']['is_partial'] else ''}PHIÊN {vn_date(report['meta']['asof'])[:5]} ({len(orders)})",
                   [("Hành động", 0.8, "l"), ("Mã", 0.7, "l"), ("Giá", 0.8, "r"), ("Khối lượng", 1.5, "l"), ("PnL", 0.8, "r"),
                    ("Đã giữ", 0.8, "r"), ("Rank", 0.6, "r"), ("Lý do", 5.2, "l")], orders),
        "positions": (f"VỊ THẾ MÔ HÌNH ĐANG GIỮ ({len(positions)})",
                      [("Mã", 0.8, "l"), ("Mua", 0.8, "r"), ("Giá mua", 1, "r"), ("Giá nay", 1, "r"), ("PnL", 0.9, "r"), ("Hôm nay", 0.9, "r"),
                       ("Tỷ trọng", 0.9, "r"), ("Cắt lỗ @", 1, "r"), ("Trailing @", 1, "r"), ("Rank", 0.7, "r")], positions),
        "watchlist": ("WATCHLIST — TOP RANK COMPOSITE",
                      [("Mã", 0.9, "l"), ("Rank", 0.8, "r"), ("Mom-6", 1, "r"), ("Vol spike", 1, "r"), ("Giá", 1, "r"), ("Trạng thái", 5, "l")], watch),
    }


def preview_diff_lines(report: dict) -> list[tuple[str, str]]:
    d = report.get("vs_preview")
    if d is None:
        return []
    fmt = lambda os_: ", ".join(f"{SIDE_TEXT[o['side']]} {o['symbol']}" for o in os_) or "—"
    return [(f"Giữ nguyên so với 14:00: {fmt(d['kept'])}", C["muted"]),
            (f"Mới khi chốt phiên: {fmt(d['added'])}", C["green"] if d["added"] else C["muted"]),
            (f"Có lúc 14:00 nhưng KHÔNG còn: {fmt(d['removed'])}", C["red"] if d["removed"] else C["muted"])]


def footer_lines(report: dict) -> list[tuple[str, str]]:
    m = report["meta"]
    lines = [(f"⚠ {w}", C["amber"]) for w in m["warnings"]]
    if m["is_partial"]:
        lines.append(("⚠ Nến hôm nay chưa chốt: giá là giá gần nhất, KHỐI LƯỢNG chưa đủ phiên → thứ hạng/lệnh có thể thay đổi lúc 15:00.", C["amber"]))
    lines.append((DISCLAIMER, C["muted"]))
    lines.append((f"Engine {m['engine']} · universe {m['universe_size']} mã" + (f" (loại: {', '.join(m['dropped'])})" if m["dropped"] else "")
                  + f" · {m['data_note']}".rstrip(" ·") + (f" · tạo lúc {m['generated_at']}" if m["generated_at"] else ""), C["muted"]))
    return lines


# ----------------------------------------------------------------------------- PNG
class _Canvas:
    """Ghi lại các lệnh vẽ (toạ độ inch, gốc ở góc trên-trái) rồi phát lại lên figure có chiều cao vừa đủ."""

    def __init__(self, width: float):
        self.W, self.ops = width, []

    def rect(self, x, y, w, h, fc, ec=None, r=0.08):
        self.ops.append(("rect", x, y, w, h, fc, ec, r))

    def text(self, x, y, s, color, size=10, weight="normal", ha="left", va="center"):
        self.ops.append(("text", x, y, s, color, size, weight, ha, va))

    def line(self, xs, ys, color, lw=1.2, ls="-"):
        self.ops.append(("line", xs, ys, color, lw, ls))

    def save(self, path: Path, height: float, dpi: int = 110):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        fig = plt.figure(figsize=(self.W, height), dpi=dpi, facecolor=C["bg"])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.W); ax.set_ylim(height, 0); ax.axis("off")
        for op in self.ops:
            if op[0] == "rect":
                _, x, y, w, h, fc, ec, r = op
                ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or fc, lw=0.8))
            elif op[0] == "text":
                _, x, y, s, color, size, weight, ha, va = op
                ax.text(x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha, va=va, family="DejaVu Sans")
            else:
                _, xs, ys, color, lw, ls = op
                ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round")
        fig.savefig(path, dpi=dpi, facecolor=C["bg"])
        plt.close(fig)


def _clip(s: str, width_in: float, size: float) -> str:
    max_chars = max(4, int(width_in / (size * 0.0083)))
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def render_png(report: dict, path: str | Path, width: float = 13.0, dpi: int = 110) -> Path:
    """Vẽ báo cáo ra PNG. Chiều cao tự tính theo số dòng của từng bảng."""
    path = Path(path)
    cv, m = _Canvas(width), 0.35
    inner = width - 2 * m
    y = 0.3

    # ---- header
    meta = report["meta"]
    cv.rect(m, y, inner, 0.95, C["panel"], C["border"])
    cv.text(m + 0.25, y + 0.33, f"MOMENTUM VN30 · {mode_title(report)}", C["text"], 17, "bold")
    cv.text(m + 0.25, y + 0.68, f"Phiên {vn_date(meta['asof'])} · Composite IC-weighted + exit rules · cập nhật {meta['generated_at'] or '—'}"
            + (" · theo yêu cầu" if meta.get("trigger") == "on_demand" else ""), C["muted"], 9.5)
    badge = "TẠM TÍNH" if meta["is_partial"] else "ĐÃ CHỐT"
    cv.rect(m + inner - 1.45, y + 0.27, 1.2, 0.4, C["amber"] if meta["is_partial"] else C["green"], r=0.1)
    cv.text(m + inner - 0.85, y + 0.47, badge, C["bg"], 10.5, "bold", "center")
    y += 1.15

    # ---- KPI
    gap = 0.18
    bw = (inner - 3 * gap) / 4
    for i, k in enumerate(kpis(report)):
        x = m + i * (bw + gap)
        cv.rect(x, y, bw, 1.1, C["panel"], C["border"])
        cv.text(x + 0.2, y + 0.25, k["label"], C["muted"], 9)
        cv.text(x + 0.2, y + 0.62, k["value"], k["color"], 19, "bold")
        cv.text(x + 0.2, y + 0.93, _clip(k["sub"], bw - 0.3, 8.5), C["muted"], 8.5)
    y += 1.35

    # ---- bảng
    def table(title, cols, rows, empty_msg=None):
        nonlocal y
        cv.text(m, y + 0.15, title, C["blue"], 11, "bold")
        y += 0.38
        total = sum(c[1] for c in cols)
        widths = [c[1] / total * (inner - 0.3) for c in cols]
        cv.rect(m, y, inner, 0.34, C["panel2"], r=0.05)
        x = m + 0.15
        for (h, _, al), w in zip(cols, widths):
            cv.text(x + (w - 0.08 if al == "r" else 0.1), y + 0.17, h, C["muted"], 8.8, "bold", "right" if al == "r" else "left")
            x += w
        y += 0.38
        if not rows:
            cv.rect(m, y, inner, 0.42, C["panel"], r=0.05)
            cv.text(m + 0.2, y + 0.21, _clip(empty_msg or "—", inner - 0.4, 9.5), C["muted"], 9.5)
            y += 0.55
            return
        for i, r in enumerate(rows):
            if i % 2 == 0:
                cv.rect(m, y, inner, 0.34, C["panel"], r=0.04)
            x = m + 0.15
            for (txt, color, bold), (_, _, al), w in zip(r["cells"], cols, widths):
                cv.text(x + (w - 0.08 if al == "r" else 0.1), y + 0.17, _clip(str(txt), w - 0.2, 9.3), color, 9.3, "bold" if bold else "normal",
                        "right" if al == "r" else "left")
                x += w
            y += 0.34
        y += 0.2

    specs = table_specs(report)
    for key in ("orders", "positions", "watchlist"):
        title, cols, rows = specs[key]
        table(title, cols, rows, no_order_reason(report) if key == "orders" else "Chưa có vị thế mô hình nào." if key == "positions" else None)

    # ---- so với 14:00
    diff = preview_diff_lines(report)
    if diff:
        cv.text(m, y + 0.15, "SO VỚI LỆNH DỰ KIẾN LÚC 14:00", C["blue"], 11, "bold")
        y += 0.42
        for txt, color in diff:
            cv.text(m + 0.15, y + 0.12, _clip(txt, inner - 0.3, 9.5), color, 9.5)
            y += 0.28
        y += 0.2

    # ---- biểu đồ
    ch = report["portfolio"]["chart"]
    cv.text(m, y + 0.15, f"LŨY KẾ {len(ch['dates'])} PHIÊN GẦN NHẤT (rebase 0%) — Danh mục mô hình vs 1/N", C["blue"], 11, "bold")
    y += 0.4
    ph = 2.1
    cv.rect(m, y, inner, ph, C["panel"], C["border"])
    pl, pr, pt, pb = m + 0.7, m + inner - 1.0, y + 0.2, y + ph - 0.45
    s, b = ch["strategy"], ch["benchmark"]
    lo, hi = min(min(s), min(b), 0), max(max(s), max(b), 0)
    span = (hi - lo) or 1.0
    X = lambda i: pl + (pr - pl) * i / max(1, len(s) - 1)
    Y = lambda v: pb - (pb - pt) * (v - lo) / span
    cv.line([pl, pr], [Y(0), Y(0)], C["border"], 1, "--")
    cv.line([X(i) for i in range(len(b))], [Y(v) for v in b], C["muted"], 1.5)
    cv.line([X(i) for i in range(len(s))], [Y(v) for v in s], C["pink"], 2.2)
    cv.text(pr + 0.08, Y(s[-1]) - 0.1, pct(s[-1], 1), C["pink"], 9, "bold")
    cv.text(pr + 0.08, Y(b[-1]) + 0.12, pct(b[-1], 1), C["muted"], 9, "bold")
    cv.text(pl - 0.08, Y(lo), pct(lo, 0), C["muted"], 8, ha="right")
    cv.text(pl - 0.08, Y(hi), pct(hi, 0), C["muted"], 8, ha="right")
    cv.text(pl, y + ph - 0.2, vn_date(ch["dates"][0]), C["muted"], 8)
    cv.text(pr, y + ph - 0.2, vn_date(ch["dates"][-1]), C["muted"], 8, ha="right")
    cv.text(pl + 1.3, y + ph - 0.2, "— Danh mục mô hình", C["pink"], 8.5, "bold")
    cv.text(pl + 3.1, y + ph - 0.2, "— 1/N Equal-Weight", C["muted"], 8.5, "bold")
    y += ph + 0.3

    # ---- footer
    for txt, color in footer_lines(report):
        cv.text(m, y + 0.1, _clip(txt, inner, 8.6), color, 8.6)
        y += 0.27
    cv.save(path, y + 0.25, dpi)
    return path


# ----------------------------------------------------------------------------- HTML
def _svg_chart(ch: dict, w: int = 1000, h: int = 230) -> str:
    s, b = ch["strategy"], ch["benchmark"]
    lo, hi = min(min(s), min(b), 0), max(max(s), max(b), 0)
    span = (hi - lo) or 1.0
    pl, pr, pt, pb = 60, w - 80, 16, h - 34
    X = lambda i: pl + (pr - pl) * i / max(1, len(s) - 1)
    Y = lambda v: pb - (pb - pt) * (v - lo) / span
    poly = lambda arr, col, sw: f'<polyline fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round" points="' + " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(arr)) + '"/>'
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Lũy kế danh mục vs 1/N">'
            f'<line x1="{pl}" x2="{pr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="{C["border"]}" stroke-dasharray="5 4"/>'
            + poly(b, C["muted"], 1.6) + poly(s, C["pink"], 2.4)
            + f'<text x="{pr + 6}" y="{Y(s[-1]):.1f}" fill="{C["pink"]}" font-size="13" font-weight="700">{pct(s[-1], 1)}</text>'
            f'<text x="{pr + 6}" y="{Y(b[-1]) + 14:.1f}" fill="{C["muted"]}" font-size="13" font-weight="700">{pct(b[-1], 1)}</text>'
            f'<text x="{pl - 6}" y="{Y(hi) + 4:.1f}" fill="{C["muted"]}" font-size="11" text-anchor="end">{pct(hi, 0)}</text>'
            f'<text x="{pl - 6}" y="{Y(lo) + 4:.1f}" fill="{C["muted"]}" font-size="11" text-anchor="end">{pct(lo, 0)}</text>'
            f'<text x="{pl}" y="{h - 8}" fill="{C["muted"]}" font-size="11">{vn_date(ch["dates"][0])}</text>'
            f'<text x="{pr}" y="{h - 8}" fill="{C["muted"]}" font-size="11" text-anchor="end">{vn_date(ch["dates"][-1])}</text></svg>')


def render_html(report: dict) -> str:
    """HTML độc lập (không CDN/JS): mở offline được, in/lưu PDF được."""
    meta, esc = report["meta"], html.escape
    badge_col = C["amber"] if meta["is_partial"] else C["green"]
    kpi_html = "".join(
        f'<div class="kpi"><div class="lbl">{esc(k["label"])}</div><div class="val" style="color:{k["color"]}">{esc(k["value"])}</div><div class="sub">{esc(k["sub"])}</div></div>'
        for k in kpis(report))

    def tbl(title, cols, rows, empty):
        head = "".join(f'<th class="{a}">{esc(h)}</th>' for h, _, a in cols)
        if rows:
            body = "".join("<tr>" + "".join(f'<td class="{a}" style="color:{c};{"font-weight:700" if b else ""}">{esc(str(t))}</td>'
                                           for (t, c, b), (_, _, a) in zip(r["cells"], cols)) + "</tr>" for r in rows)
        else:
            body = f'<tr><td colspan="{len(cols)}" class="empty">{esc(empty)}</td></tr>'
        return f'<section><h2>{esc(title)}</h2><div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>'

    specs = table_specs(report)
    empties = {"orders": no_order_reason(report), "positions": "Chưa có vị thế mô hình nào.", "watchlist": "—"}
    tables = "".join(tbl(*specs[k][:3], empties[k]) for k in ("orders", "positions", "watchlist"))
    diff = preview_diff_lines(report)
    diff_html = ("<section><h2>SO VỚI LỆNH DỰ KIẾN LÚC 14:00</h2>" + "".join(f'<p style="color:{c}">{esc(t)}</p>' for t, c in diff) + "</section>") if diff else ""
    foot = "".join(f'<p class="foot" style="color:{c}">{esc(t)}</p>' for t, c in footer_lines(report))
    css = f"""
    :root{{color-scheme:dark}} *{{box-sizing:border-box}}
    body{{margin:0;background:{C['bg']};color:{C['text']};font:14px/1.5 -apple-system,'Segoe UI',Roboto,'DejaVu Sans',sans-serif}}
    .wrap{{max-width:1100px;margin:0 auto;padding:22px 16px 40px}}
    header{{background:{C['panel']};border:1px solid {C['border']};border-radius:12px;padding:18px 22px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}}
    h1{{margin:0;font-size:22px}} .meta{{color:{C['muted']};font-size:13px;margin-top:4px}}
    .badge{{background:{badge_col};color:{C['bg']};font-weight:800;border-radius:8px;padding:6px 14px;font-size:13px}}
    .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:14px 0}}
    .kpi{{background:{C['panel']};border:1px solid {C['border']};border-radius:12px;padding:14px 18px}}
    .lbl{{color:{C['muted']};font-size:12px}} .val{{font-size:27px;font-weight:800;margin:2px 0}} .sub{{color:{C['muted']};font-size:12px}}
    section{{margin-top:22px}} h2{{font-size:14px;color:{C['blue']};letter-spacing:.4px;margin:0 0 8px}}
    .scroll{{overflow-x:auto;border:1px solid {C['border']};border-radius:10px}}
    table{{border-collapse:collapse;width:100%;min-width:640px;background:{C['panel']}}}
    th{{background:{C['panel2']};color:{C['muted']};font-size:12px;padding:8px 12px;white-space:nowrap}}
    td{{padding:8px 12px;border-top:1px solid {C['border']};font-size:13px}} .r{{text-align:right}} .l{{text-align:left}}
    tbody tr:nth-child(even) td{{background:rgba(255,255,255,.015)}} .empty{{color:{C['muted']};padding:14px 12px}}
    .chart{{background:{C['panel']};border:1px solid {C['border']};border-radius:12px;padding:10px}}
    .legend{{color:{C['muted']};font-size:12px;margin:4px 6px}} .foot{{font-size:12px;margin:6px 0}}
    """
    return f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Momentum VN30 — {esc(vn_date(meta['asof']))} — {esc(mode_title(report))}</title><style>{css}</style></head><body><div class="wrap">
<header><div><h1>MOMENTUM VN30 · {esc(mode_title(report))}</h1>
<div class="meta">Phiên {esc(vn_date(meta['asof']))} · Composite IC-weighted + exit rules · cập nhật {esc(meta['generated_at'] or '—')}{' · theo yêu cầu' if meta.get('trigger') == 'on_demand' else ''}</div></div>
<span class="badge">{'TẠM TÍNH' if meta['is_partial'] else 'ĐÃ CHỐT'}</span></header>
<div class="kpis">{kpi_html}</div>{tables}{diff_html}
<section><h2>LŨY KẾ {len(report['portfolio']['chart']['dates'])} PHIÊN GẦN NHẤT (rebase 0%)</h2><div class="chart">{_svg_chart(report['portfolio']['chart'])}
<div class="legend"><span style="color:{C['pink']}">━ Danh mục mô hình</span> &nbsp; <span style="color:{C['muted']}">━ 1/N Equal-Weight</span></div></div></section>
<section>{foot}</section></div></body></html>"""

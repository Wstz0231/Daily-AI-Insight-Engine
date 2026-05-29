import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date
from collections import Counter, defaultdict
from util import load_json, parse_yyyy_mm_dd, sentiment_label, get_openai_client

def impact_to_weight(level: Optional[int]) -> float:
    if level == 3:
        return 1.0
    if level == 2:
        return 0.6
    if level == 1:
        return 0.2
    return 0.4


def load_structured(path: Path = Path("data/structured_data.json")) -> List[Dict[str, Any]]:
    data = load_json(path)
    return data

def compute_recency_map(records: List[Dict[str, Any]]) -> Dict[int, float]:
    dated: List[Tuple[int, date]] = []
    for r in records:
        d = parse_yyyy_mm_dd(r.get("published_at"))
        if d is not None:
            rid = r.get("id")
            if isinstance(rid, int):
                dated.append((rid, d))
    if not dated:
        return {}
    all_dates = [d for _, d in dated]
    d_min, d_max = min(all_dates), max(all_dates)
    span = max(1, (d_max - d_min).days)
    out: Dict[int, float] = {}
    for rid, d in dated:
        out[rid] = 1.0 - ((d_max - d).days / span)
    return out


def sentiment_boost(val: Optional[int]) -> float:
    if val == 1:
        return 1.0
    if val == -1:
        return 1.0
    return 0.0


def compute_scores(records: List[Dict[str, Any]]) -> Dict[int, float]:
    """加权计算分数"""
    recency = compute_recency_map(records)
    out: Dict[int, float] = {}
    for r in records:
        rid = r.get("id")
        if not isinstance(rid, int):
            continue
        s_rec = recency.get(rid, 0.5)
        s_imp = impact_to_weight(r.get("impact_level"))
        s_sen = sentiment_boost(r.get("sentiment"))
        score = 0.3 * s_rec + 0.4 * s_imp + 0.3 * s_sen
        out[rid] = round(float(score), 6)
    return out

# 影响力可视化
def impact_badge(level: Optional[int]) -> str:
    try:
        lvl = int(level)
    except Exception:
        lvl = 2
    lvl = max(1, min(3, lvl))
    return "★" * lvl


# 趋势列表
def build_trends(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    topics = Counter()
    senti = Counter()
    sources = Counter()
    for r in records:
        for t in r.get("topic_tags", []) or []:
            if isinstance(t, str) and t:
                topics[t] += 1
        sv = r.get("sentiment")
        senti[sentiment_label(sv)] += 1
        src = r.get("source")
        if isinstance(src, str) and src:
            sources[src] += 1
    return {
        "topics": topics,
        "sentiment": senti,
        "sources": sources,
    }


def select_top_events(records: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k records by score; tie-break by newer published_at, then id."""
    if not records:
        return []
    scores = compute_scores(records)

    def sort_key(r: Dict[str, Any]):
        rid = r.get("id") if isinstance(r.get("id"), int) else -1
        sc = scores.get(rid, 0.0)
        d = parse_yyyy_mm_dd(r.get("published_at")) or date.min
        ord_val = d.toordinal()
        return (-sc, -ord_val, rid)

    sorted_items = sorted(records, key=sort_key)
    return sorted_items[: max(0, k)]


def pick_report_date(records: List[Dict[str, Any]]) -> str:
    dates = [parse_yyyy_mm_dd(r.get("published_at")) for r in records]
    dates = [d for d in dates if d is not None]
    if dates:
        return max(dates).isoformat()
    return date.today().isoformat()


def fmt_list(values: List[str], limit: int = 3) -> str:
    vals = [v for v in values if isinstance(v, str) and v]
    if not vals:
        return "—"
    if len(vals) > limit:
        vals = vals[:limit]
    return "、".join(vals)


def render_top_events_section(top: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("## 今日热点 Top 5")
    lines.append("")
    for idx, r in enumerate(top, start=1):
        title = r.get("title", "")
        url = r.get("url", "")
        src = r.get("source", "")
        d = r.get("published_at", "")
        sen = sentiment_label(r.get("sentiment"))
        head = f"- [{idx}] [{title}]({url}) — {src} | {d} | {sen}"
        lines.append(head)
    return "\n".join(lines)


def render_deep_dives_section(top: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("## 重要事件概述")
    lines.append("")
    for r in top:
        title = r.get("title", "")
        url = r.get("url", "")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"  - 背景/概述：{r.get('summary','')}")
        lines.append(f"  - 影响（{impact_badge(r.get('impact_level'))}）：{r.get('impact','影响待评估。')}")
        lines.append(f"  - 相关方：{fmt_list(r.get('key_entities', []) or [])}")
        lines.append(f"  - 主题：{fmt_list(r.get('topic_tags', []) or [])}")
    return "\n".join(lines)


def render_trends_section(tr: Dict[str, Any], top_n: int = 5) -> str:
    lines: List[str] = []
    lines.append("## 趋势与分布")
    lines.append("")
    topics: Counter = tr["topics"]
    senti: Counter = tr["sentiment"]
    sources: Counter = tr["sources"]
    top_topics = topics.most_common(top_n)
    if top_topics:
        tline = ", ".join([f"{k}:{v}" for k, v in top_topics])
    else:
        tline = "—"
    lines.append(f"- 主题Top{top_n}：{tline}")
    total = sum(senti.values())
    if total > 0:
        order = ["正面", "中性", "负面"]
        parts = []
        for label in order:
            pct = 100.0 * senti.get(label, 0) / total
            parts.append(f"{label}:{pct:.0f}%")
        sline = ", ".join(parts)
    else:
        sline = "—"
    lines.append(f"- 舆情分布：{sline}")
    return "\n".join(lines)


def render_markdown(records: List[Dict[str, Any]], top_k: int = 5) -> str:
    report_date = pick_report_date(records)
    scores = compute_scores(records)
    top = select_top_events(records, k=top_k)
    trends = build_trends(records)

    lines: List[str] = []
    lines.append(f"# AI 舆情分析日报（{report_date}）")
    lines.append("")
    lines.append(render_top_events_section(top))
    lines.append("")
    lines.append(render_deep_dives_section(top))
    lines.append("")
    lines.append(render_trends_section(trends))
    lines.append("")
    return "\n".join(lines)


def write_report(markdown_text: str, out_dir: Path = Path("output"), report_date: Optional[str] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = report_date or date.today().isoformat()
    path = out_dir / f"report_{date_str}.md"
    path.write_text(markdown_text, encoding="utf-8")
    return path


def generate_trend_insight_lm(top_events: List[Dict[str, Any]], client) -> str:
    bullets = []
    for i, r in enumerate(top_events[:5], start=1):
        bullets.append(
            f"[{i}] 标题：{r.get('title','')} | 主题：{','.join(r.get('topic_tags',[]) or [])}\n摘要：{r.get('summary','')}"
        )
    joined = "\n\n".join(bullets)

    messages = [
        {
            "role": "system",
            "content": (
                "你是科技行业分析师。请基于提供的重点事件摘要，提炼今日在技术/应用/政策/资本四个方向上的趋势洞察。"
                "用简洁的中文写一段分析，3-5句为宜。不要罗列清单，不要使用标题或小结，只输出一段文字。"
            ),
        },
        {
            "role": "user",
            "content": (
                "以下是Top 5事件的关键信息，请综合判断整体趋势：\n\n" + joined
            ),
        },
    ]

    from openai import OpenAI
    try:
        resp = get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=messages,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text
    except Exception:
        return "今日趋势：无"


def render_trend_judgement_section(text: str) -> str:
    lines = ["## 趋势判断", "", text, ""]
    return "\n".join(lines)


def generate_risk_opportunity_lm(trend_text: str, client) -> str:
    """基于趋势判断生成风险或机会提示，输出两句话。"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是科技行业风险与机会分析师。"
                "根据提供的趋势判断，分别给出一句风险提示和一句机会提示。"
                "格式严格为两行，第一行以'【风险】'开头，第二行以'【机会】'开头，每行只有一句话，不超过40字。"
            ),
        },
        {
            "role": "user",
            "content": f"以下是今日趋势判断，请基于此给出风险与机会提示：\n\n{trend_text}",
        },
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=messages,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return "【风险】暂无风险提示。\n【机会】暂无机会提示。"


def render_risk_opportunity_section(text: str) -> str:
    lines = ["## 风险或机会提示", "", text, ""]
    return "\n".join(lines)


def render_report(records: List[Dict[str, Any]], top: List[Dict[str, Any]], trends: Dict[str, Any], trend_text: str, risk_opportunity_text: str) -> str:
    report_date = pick_report_date(records)
    lines: List[str] = []
    lines.append(f"# AI 舆情分析日报（{report_date}）")
    lines.append("")
    lines.append(render_top_events_section(top))
    lines.append("")
    lines.append(render_deep_dives_section(top))
    lines.append("")
    lines.append(render_trend_judgement_section(trend_text))
    lines.append("")
    lines.append(render_trends_section(trends))
    lines.append("")
    lines.append(render_risk_opportunity_section(risk_opportunity_text))
    return "\n".join(lines)


def run():
    structured_path = Path("data/structured_data.json")
    items = load_structured(structured_path)
    top = select_top_events(items, k=5)
    trends = build_trends(items)
    client = get_openai_client()
    trend_text = generate_trend_insight_lm(top, client)
    risk_opportunity_text = generate_risk_opportunity_lm(trend_text, client)
    md = render_report(items, top, trends, trend_text, risk_opportunity_text)
    d = pick_report_date(items)
    out_path = write_report(md, Path("output"), d)
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    run()


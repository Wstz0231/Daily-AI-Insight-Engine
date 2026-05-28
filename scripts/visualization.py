from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
from datetime import date
from util import load_json, parse_yyyy_mm_dd, sentiment_label


def load_structured(path: Path = Path("data/structured_data.json")) -> List[Dict[str, Any]]:
    return load_json(path)


def derive_report_date(records: List[Dict[str, Any]]) -> str:
    dates = [parse_yyyy_mm_dd(r.get("published_at")) for r in records]
    dates = [d for d in dates if d is not None]
    if dates:
        return max(dates).isoformat()
    return date.today().isoformat()


def count_topics(records: List[Dict[str, Any]], top_n: int = 10) -> List[Tuple[str, int]]:
    c = Counter()
    for r in records:
        for t in r.get("topic_tags", []) or []:
            if isinstance(t, str) and t:
                c[t] += 1
    return c.most_common(top_n)


def count_sentiment(records: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    c = Counter()
    for r in records:
        c[sentiment_label(r.get("sentiment"))] += 1
    return c.most_common()


def ensure_chinese_font():
    # Try set a common Chinese font; if missing, silently continue
    import matplotlib
    try:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Arial Unicode MS']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


def plot_pie(items: List[Tuple[str, int]], title: str, out_path: Path, figsize=(7, 7)) -> Optional[Path]:
    if not items:
        return None
    import matplotlib.pyplot as plt

    labels = [k for k, _ in items]
    values = [v for _, v in items]

    total = sum(values)
    def _autopct(pct):
        count = int(round(pct / 100.0 * total))
        return f"{pct:.1f}%\n({count})"

    ensure_chinese_font()
    plt.figure(figsize=figsize)
    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        autopct=_autopct,
        startangle=140,
        counterclock=False,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def generate_topic_chart(records: List[Dict[str, Any]], out_dir: Path = Path("output/figures")) -> Optional[Path]:
    items = count_topics(records, top_n=10)
    if not items:
        return None
    d = derive_report_date(records)
    title = f"主题分布– {d}"
    path = out_dir / f"topics_{d}.png"
    return plot_pie(items, title, path, figsize=(7, 7))


def generate_sentiment_chart(records: List[Dict[str, Any]], out_dir: Path = Path("output/figures")) -> Optional[Path]:
    items = count_sentiment(records)
    if not items:
        return None
    d = derive_report_date(records)
    title = f"舆情分布 – {d}"
    path = out_dir / f"sentiment_{d}.png"
    return plot_pie(items, title, path, figsize=(6, 6))


def main():
    structured_path = Path("data/structured_data.json")
    records = load_structured(structured_path)
    out_dir = Path("output/figures")
    out_paths = []
    p = generate_topic_chart(records, out_dir)
    if p:
        out_paths.append(p)
    p = generate_sentiment_chart(records, out_dir)
    if p:
        out_paths.append(p)
    if out_paths:
        print("Saved figures:")
        for x in out_paths:
            print(" -", x)
    else:
        print("No figures generated (no data)")


if __name__ == "__main__":
    main()

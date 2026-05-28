import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
from datetime import datetime, date
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


def count_categories(records: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    c = Counter()
    for r in records:
        cat = r.get("category")
        if isinstance(cat, str) and cat:
            c[cat] += 1
    return c.most_common()


def sentiment_label(v: Optional[int]) -> str:
    if v == 1:
        return "正面"
    if v == -1:
        return "负面"
    return "中性"


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


def plot_barh(items: List[Tuple[str, int]], title: str, xlabel: str, out_path: Path, figsize=(8, 5)) -> Optional[Path]:
    if not items:
        return None
    import matplotlib.pyplot as plt

    labels = [k for k, _ in items][::-1]  # reverse for horizontal bar from top to bottom
    values = [v for _, v in items][::-1]

    ensure_chinese_font()
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(labels)), values, color="#4C78A8")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel(xlabel)
    plt.title(title)

    # annotate counts
    for i, b in enumerate(bars):
        w = b.get_width()
        plt.text(w + max(values) * 0.01, b.get_y() + b.get_height()/2, str(values[i]), va='center')

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
    title = f"主题分布（Top 10）– {d}"
    path = out_dir / f"topics_{d}.png"
    return plot_barh(items, title, "条目数", path, figsize=(9, 6))


def generate_category_chart(records: List[Dict[str, Any]], out_dir: Path = Path("output/figures")) -> Optional[Path]:
    items = count_categories(records)
    if not items:
        return None
    d = derive_report_date(records)
    title = f"类别分布 – {d}"
    path = out_dir / f"categories_{d}.png"
    return plot_barh(items, title, "条目数", path, figsize=(7, 5))


def generate_sentiment_chart(records: List[Dict[str, Any]], out_dir: Path = Path("output/figures")) -> Optional[Path]:
    items = count_sentiment(records)
    if not items:
        return None
    d = derive_report_date(records)
    title = f"舆情分布 – {d}"
    path = out_dir / f"sentiment_{d}.png"
    return plot_barh(items, title, "条目数", path, figsize=(6, 4))


def main():
    structured_path = Path("data/structured_data.json")
    records = load_structured(structured_path)
    out_dir = Path("output/figures")
    out_paths = []
    p = generate_topic_chart(records, out_dir)
    if p:
        out_paths.append(p)
    # Optional extras
    p = generate_category_chart(records, out_dir)
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

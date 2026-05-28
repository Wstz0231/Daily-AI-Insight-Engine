import re
from typing import Optional
from datetime import datetime
from pathlib import Path
from util import load_json, save_json, get_openai_client


def clean_text(text: Optional[str]) -> str:
    """清理源语句"""
    if not text:
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def has_cjk(text: Optional[str]) -> bool:
    """判断语言是否为中文"""
    if not text:
        return False
    return re.search(r"[\u4e00-\u9fff]", str(text)) is not None


def parse_date(date_str: str) -> str:
    """时间标准化"""
    if not date_str:
        return ""
    s = date_str.strip()
    # YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    patterns = [
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
    ]
    for p in patterns:
        try:
            dt = datetime.strptime(s, p)
            return dt.date().isoformat()
        except Exception:
            continue
    return ""


def clean_published_at(date_str: Optional[str]) -> str:
    """清理发布日期项"""
    if not date_str:
        return ""
    s = clean_text(date_str)
    return parse_date(s)


def openai_translate_to_cn(text: str, client, model: str = "gpt-4o-mini") -> str:
    """翻译语料为中文."""
    if not text:
        return ""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "你是一个中文翻译助手。只输出中文翻译结果。"},
            {"role": "user", "content": f"用中文翻译下面内容，只输出中文结果：\n{text}"},
        ],
    )
    out = resp.choices[0].message.content or ""
    return out.strip()


def openai_generate_title_cn(text: str, client, model: str = "gpt-4o-mini") -> str:
    """通过概述生成标题"""
    if not text:
        return ""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "你是新闻编辑。只输出中文标题，不加引号或多余说明。"},
            {"role": "user", "content": f"基于以下内容生成中文新闻标题，不超过20个汉字，只输出标题：\n{text}"},
        ],
    )
    out = (resp.choices[0].message.content or "").strip()
    return out[:20]


def openai_summarize_cn(text: str, client, model: str = "gpt-4o-mini", max_sentences: int = 3) -> str:
    """生成概述，主要包括：背景+影响分析."""
    if not text:
        return ""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名中文新闻编辑。请围绕‘关键事件的背景+影响分析’撰写摘要。"
                    "仅输出摘要正文，不要任何标签、标题或多余说明。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请用中文写一段1-{max_sentences}句的摘要，按‘背景+影响/意义/后续’展开，"
                    "优先保留事实、机构与数值信息，避免营销语与主观夸张。只输出正文：\n"
                    f"{text}"
                ),
            },
        ],
    )
    out = resp.choices[0].message.content or ""
    return out.strip()


def clean_summary(item: dict, client) -> str:
    """清理概述项"""
    src = item.get("content") or item.get("summary") or ""
    src = clean_text(src)
    if not src:
        return ""
    out = openai_summarize_cn(src, client)
    return clean_text(out)


def clean_title(item: dict, client) -> Optional[str]:
    """清理标题项，如果为其他语言则翻译，如果无标题则生成"""
    title = clean_text(item.get("title", ""))
    if title:
        if not has_cjk(title):
            title = openai_translate_to_cn(title, client)
        title = clean_text(title)[:20]
        return title if title else None
    # No title; try to generate from available text
    base = clean_text(item.get("content", "") or item.get("summary", ""))
    if not base:
        return None
    gen = openai_generate_title_cn(base, client)
    gen = clean_text(gen)[:20]
    return gen if gen else None


def should_discard(item: dict) -> bool:
    """清理无内容也无标题的条目"""
    title = clean_text(item.get("title", ""))
    if title:
        return False
    src = clean_text(item.get("summary", "") or item.get("content", ""))
    return not bool(src)


def clean_record(item: dict, client) -> Optional[dict]:
    """清理整个条目"""
    if should_discard(item):
        return None
    title = clean_title(item, client)
    if not title:
        return None
    summary = clean_summary(item, client)
    published_at = clean_published_at(item.get("published_at"))
    url = item.get("url", "")
    source = item.get("source", "")
    content = item.get("content", "")
    return {
        "title": title,
        "summary": summary,
        "published_at": published_at,
        "url": url,
        "source": source,
        "content": content,
    }


def assign_sequential_ids(records: list[dict]) -> list[dict]:
    """给予每个条目数字编号"""
    for i, rec in enumerate(records, start=1):
        rec["id"] = i
    return records


def run_pipeline(input_path: Path, output_path: Path, client) -> None:
    """加载 -> 清理 -> ID -> 存储"""
    records = load_json(input_path)
    cleaned = []
    for item in records:
        rec = clean_record(item, client)
        if rec is not None:
            cleaned.append(rec)
    assign_sequential_ids(cleaned)
    save_json(cleaned, output_path)


if __name__ == "__main__":
    client = get_openai_client()
    in_path = Path("data/raw_data.json")
    out_path = Path("data/cleaned_data.json")
    run_pipeline(in_path, out_path, client)
    print(f"Cleaned data written to: {out_path}")
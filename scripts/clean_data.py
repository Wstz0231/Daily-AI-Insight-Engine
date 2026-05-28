import re
from typing import Optional
from datetime import datetime
from pathlib import Path
from util import load_json, save_json, get_openai_client


def clean_text(text: Optional[str]) -> str:
	"""Trim text and collapse whitespace/newlines into single spaces."""
	if not text:
		return ""
	cleaned = str(text).strip()
	cleaned = re.sub(r"\s+", " ", cleaned)
	return cleaned


def has_cjk(text: Optional[str]) -> bool:
	"""Return True if any CJK character is present (simple heuristic)."""
	if not text:
		return False
	return re.search(r"[\u4e00-\u9fff]", str(text)) is not None


def parse_date(date_str: str) -> str:
	"""Parse common date formats and return YYYY-MM-DD, else empty string."""
	if not date_str:
		return ""
	s = date_str.strip()
	# Fast path: already YYYY-MM-DD
	if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
		return s
	patterns = [
		"%Y/%m/%d",
		"%Y-%m-%d %H:%M",
		"%Y-%m-%d %H:%M:%S",
		"%b %d, %Y",   # May 29, 2026
		"%B %d, %Y",   # May 29, 2026 (full month)
		"%d %b %Y",    # 29 May 2026
		"%d %B %Y",    # 29 May 2026 (full month)
	]
	for p in patterns:
		try:
			dt = datetime.strptime(s, p)
			return dt.date().isoformat()
		except Exception:
			continue
	return ""


def clean_published_at(date_str: Optional[str]) -> str:
	"""Basic wrapper: trim and normalize to YYYY-MM-DD or empty string."""
	if not date_str:
		return ""
	s = clean_text(date_str)
	return parse_date(s)



def openai_translate_to_cn(text: str, client, model: str = "gpt-4o-mini") -> str:
	"""Translate arbitrary text to Chinese using OpenAI Chat Completions."""
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
	"""Generate a concise Chinese news title (≤20 chars)."""
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
	# Hard cut to 20 chars to keep it brief
	return out[:20]


def openai_summarize_cn(text: str, client, model: str = "gpt-4o-mini", max_sentences: int = 3) -> str:
	"""Summarize input into 1-3 Chinese sentences focusing on 背景+影响分析."""
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


# --- Field cleaners ---

def clean_summary(item: dict, client) -> str:
	"""Summarize content or summary to 1-3 Chinese sentences."""
	src = item.get("content") or item.get("summary") or ""
	src = clean_text(src)
	if not src:
		return ""
	out = openai_summarize_cn(src, client)
	return clean_text(out)


def clean_title(item: dict, client) -> Optional[str]:
	"""Produce a Chinese title (≤20 chars). Translate if needed or generate from summary/content."""
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
	"""Discard if no title and no usable text (summary or content) to derive a title."""
	title = clean_text(item.get("title", ""))
	if title:
		return False
	src = clean_text(item.get("summary", "") or item.get("content", ""))
	return not bool(src)


def clean_record(item: dict, client) -> Optional[dict]:
	"""Clean a single record. Returns cleaned dict or None if discarded."""
	if should_discard(item):
		return None
	title = clean_title(item, client)
	if not title:
		return None
	summary = clean_summary(item, client)
	published_at = clean_published_at(item.get("published_at"))
	url = item.get("url", "")  # keep as-is (trim not required by spec)
	source = item.get("source", "")  # keep as-is
	content = item.get("content", "")  # keep original text as-is
	return {
		"title": title,
		"summary": summary,
		"published_at": published_at,
		"url": url,
		"source": source,
		"content": content,
	}


def assign_sequential_ids(records: list[dict]) -> list[dict]:
	"""Assign id = 1..N in order and return the list."""
	for i, rec in enumerate(records, start=1):
		rec["id"] = i
	return records


# --- I/O and pipeline ---

def run_pipeline(input_path: Path, output_path: Path, client) -> None:
	"""Load raw -> clean records -> assign ids -> save cleaned JSON."""
	records = load_json(input_path)
	cleaned = []
	for item in records:
		rec = clean_record(item, client)
		if rec is not None:
			cleaned.append(rec)
	assign_sequential_ids(cleaned)
	save_json(cleaned, output_path)


# ---  CLI ---


if __name__ == "__main__":
	client = get_openai_client()
	in_path = Path("data/raw_data.json")
	out_path = Path("data/cleaned_data.json")
	run_pipeline(in_path, out_path, client)
	print(f"Cleaned data written to: {out_path}")


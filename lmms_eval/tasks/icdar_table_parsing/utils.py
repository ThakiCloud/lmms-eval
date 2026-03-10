from __future__ import annotations

import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger

from lmms_eval.tasks.icdar_table_parsing.teds import (
    compute_ocr,
    compute_teds_full,
    compute_teds_struct,
)

TABLE_PARSING_PROMPT = "Convert the table in the image to HTML format. Output only the HTML table code, starting with <table> and ending with </table>. Do not include any explanation."

_NUM_WORKERS = int(os.environ.get("TEDS_NUM_WORKERS", min(os.cpu_count() or 1, 64)))


def _parallel_compute(items: list[tuple[str, str, str]], fn) -> list[float]:
    """Run *fn* on every item using a process pool and return ordered results.

    Uses spawn context to avoid CUDA fork issues from vLLM parent process.
    Worker functions must live in a properly-importable module (teds.py).
    """
    n = len(items)
    if n == 0:
        return []
    workers = min(_NUM_WORKERS, n)
    if workers <= 1:
        return [fn(item) for item in items]

    results: list[float] = [0.0] * n
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        future_to_idx = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning("Worker failed for item {}: {}", idx, e)
                results[idx] = 0.0
    return results


# ---------------------------------------------------------------------------
# doc_to_* helpers
# ---------------------------------------------------------------------------


def table_parsing_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def table_parsing_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    prompt = lmms_eval_specific_kwargs.get("prompt", TABLE_PARSING_PROMPT)
    return f"{pre_prompt}{prompt}{post_prompt}"


def table_parsing_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    prompt = lmms_eval_specific_kwargs.get("prompt", TABLE_PARSING_PROMPT)

    text = f"{pre_prompt}{prompt}{post_prompt}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": doc["image"].convert("RGB")},
                {"type": "text", "text": text},
            ],
        }
    ]
    return messages


def table_parsing_doc_to_target(doc):
    return doc["html"]


def _extract_html_table(text: str) -> str:
    """Extract clean HTML table from model output.

    Uses BeautifulSoup to properly parse the HTML, strip Qwen3-VL positional
    attributes (data-bbox, data-polygon), and extract <table> elements.
    Also handles markdown code blocks wrapping the output.
    """
    code_block = re.search(r"```(?:html)?\s*(.*?)```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "html.parser")

    for attr in ("data-bbox", "data-polygon"):
        for tag in soup.find_all(attrs={attr: True}):
            del tag[attr]

    tables = soup.find_all("table")
    if tables:
        return "\n".join(str(t) for t in tables)

    return text.strip()


# ---------------------------------------------------------------------------
# process_results – lightweight; defers expensive TEDS to aggregation phase
# ---------------------------------------------------------------------------


def table_parsing_process_results(doc, results):
    """Return (pred_html, gt_html, table_type) tuples for each metric.

    The actual TEDS / OCR scores are computed in parallel during aggregation
    (see teds_aggregate, teds_struct_aggregate, ocr_aggregate below).
    """
    pred_html = _extract_html_table(results[0])
    gt_html = doc["html"]
    table_type = doc.get("type", "unknown")

    data = (pred_html, gt_html, table_type)
    return {
        "teds": data,
        "teds_struct": data,
        "ocr": data,
    }


# ---------------------------------------------------------------------------
# Aggregation – parallel TEDS / OCR computation across all samples
# ---------------------------------------------------------------------------


def teds_aggregate(items: list[tuple[str, str, str]]) -> float:
    """Compute TEDS (structure + content) for all samples in parallel."""
    workers = min(_NUM_WORKERS, len(items))
    logger.info("Computing TEDS for {} samples with {} workers", len(items), workers)
    scores = _parallel_compute(items, compute_teds_full)
    return sum(scores) / len(scores) if scores else 0.0


def teds_struct_aggregate(items: list[tuple[str, str, str]]) -> float:
    """Compute TEDS-struct (structure only) for all samples in parallel."""
    workers = min(_NUM_WORKERS, len(items))
    logger.info("Computing TEDS-struct for {} samples with {} workers", len(items), workers)
    scores = _parallel_compute(items, compute_teds_struct)
    return sum(scores) / len(scores) if scores else 0.0


def ocr_aggregate(items: list[tuple[str, str, str]]) -> float:
    """Compute OCR similarity for all samples in parallel."""
    workers = min(_NUM_WORKERS, len(items))
    logger.info("Computing OCR metric for {} samples with {} workers", len(items), workers)
    scores = _parallel_compute(items, compute_ocr)
    return sum(scores) / len(scores) if scores else 0.0

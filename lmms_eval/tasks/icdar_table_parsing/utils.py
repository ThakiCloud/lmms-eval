from __future__ import annotations

from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.icdar_table_parsing.teds import TEDS, OCRMetric

_teds_full = TEDS(structure_only=False)
_teds_struct = TEDS(structure_only=True)
_ocr_metric = OCRMetric()

TABLE_PARSING_PROMPT = "Convert the table in the image to HTML format. Output only the HTML table code, starting with <table> and ending with </table>. Do not include any explanation."


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
    import re

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


def table_parsing_process_results(doc, results):
    pred_html = _extract_html_table(results[0])
    gt_html = doc["html"]
    table_type = doc.get("type", "unknown")

    try:
        teds_score = _teds_full.evaluate(pred_html, gt_html)
    except Exception:
        logger.warning("TEDS computation failed, defaulting to 0.0")
        teds_score = 0.0

    try:
        teds_struct_score = _teds_struct.evaluate(pred_html, gt_html)
    except Exception:
        logger.warning("TEDS-struct computation failed, defaulting to 0.0")
        teds_struct_score = 0.0

    try:
        ocr_score = _ocr_metric.evaluate(pred_html, gt_html)
    except Exception:
        logger.warning("OCR metric computation failed, defaulting to 0.0")
        ocr_score = 0.0

    result = {
        "teds": teds_score,
        "teds_struct": teds_struct_score,
        "ocr": ocr_score,
    }

    if table_type == "simple":
        result["teds_simple"] = teds_score
        result["teds_struct_simple"] = teds_struct_score
    elif table_type == "complex":
        result["teds_complex"] = teds_score
        result["teds_struct_complex"] = teds_struct_score

    return result


def table_parsing_aggregate_results(results, args):
    """Aggregate per-sample results and save detailed report."""
    if not results:
        return 0.0
    mean_score = sum(results) / len(results)

    try:
        path = generate_submission_file("table_parsing_results.txt", args, subpath="results")
        with open(path, "w") as f:
            print(f"Table Parsing Results ({len(results)} samples)", file=f)
            print(f"Mean TEDS: {mean_score:.4f}", file=f)
        logger.info(f"Table parsing results saved to {path}")
    except Exception:
        pass

    return mean_score

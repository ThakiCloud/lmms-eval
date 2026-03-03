"""
Benetech Parsing 태스크 유틸리티

Benetech "Making Graphs Accessible" 공식 메트릭 기반 평가.
  - chart_type 불일치 -> x, y 모두 0점
  - 값 개수 불일치 -> 해당 시리즈 0점
  - 숫자 시리즈 -> sigmoid(sqrt(1 - R2))
  - 문자열 시리즈 -> sigmoid(total_levenshtein / total_len)
  - 최종 점수 -> 모든 시리즈 점수의 평균

Reference:
  https://www.kaggle.com/competitions/benetech-making-graphs-accessible/overview/evaluation
"""

import json
import re
from typing import Optional

import numpy as np
from rapidfuzz.distance.Levenshtein import distance as levenshtein_distance
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Benetech 공식 메트릭 핵심 함수
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 2 - 2 / (1 + np.exp(-x))


def _normalized_rmse(y_true: list, y_pred: list) -> float:
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return 0.0
    y_true, y_pred = y_true[valid], y_pred[valid]
    return float(_sigmoid((1 - r2_score(y_true, y_pred)) ** 0.5))


def _normalized_levenshtein(y_true: list, y_pred: list) -> float:
    total_dist = sum(levenshtein_distance(str(yt), str(yp)) for yt, yp in zip(y_true, y_pred))
    total_len = sum(len(str(yt)) for yt in y_true)
    if total_len == 0:
        return 0.0
    return float(_sigmoid(total_dist / total_len))


def _score_series(y_true: list, y_pred: list) -> float:
    """단일 시리즈 (x 또는 y) 점수. 길이 불일치 -> 0점."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0

    if isinstance(y_true[0], str):
        return _normalized_levenshtein(y_true, [str(x) for x in y_pred])
    else:
        try:
            y_pred_num = [float(x) for x in y_pred]
        except (ValueError, TypeError):
            return 0.0
        return _normalized_rmse(y_true, y_pred_num)


def _is_str_series(series: list) -> bool:
    return len(series) > 0 and isinstance(series[0], str)


# ---------------------------------------------------------------------------
# JSON 파싱 헬퍼
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Optional[dict]:
    """모델 출력에서 JSON 객체를 추출. 코드 블록 안이나 직접 JSON 모두 처리."""
    code_block = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    for start in range(len(text)):
        if text[start] == "{":
            for end in range(len(text), start, -1):
                if text[end - 1] == "}":
                    try:
                        return json.loads(text[start:end])
                    except json.JSONDecodeError:
                        continue
    return None


def _load_pred_series(pred: dict) -> tuple:
    """예측 JSON에서 (x_series, y_series, chart_type) 추출."""
    chart_type = pred.get("chart_type", "unknown")
    ds = pred.get("data_series", [])
    if not isinstance(ds, list):
        return [], [], chart_type

    pred_x, pred_y = [], []
    for dp in ds:
        if isinstance(dp, dict):
            pred_x.append(dp.get("x", ""))
            pred_y.append(dp.get("y", ""))
        else:
            return [], [], chart_type
    return pred_x, pred_y, chart_type


def _load_gt_series(data_series: list) -> tuple:
    """GT data-series에서 (x_values, y_values) 추출. 원래 타입 유지."""
    gt_x = [dp["x"] for dp in data_series]
    gt_y = [dp["y"] for dp in data_series]
    return gt_x, gt_y


# ---------------------------------------------------------------------------
# lmms-eval 태스크 인터페이스
# ---------------------------------------------------------------------------


def benetech_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def benetech_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{post_prompt}"


def benetech_process_results(doc, results):
    pred_text = results[0]
    gt_chart_type = doc["chart_type"]
    gt_data_series = json.loads(doc["data_series_json"])
    gt_x, gt_y = _load_gt_series(gt_data_series)

    zero = {
        "benetech_overall": 0.0,
        "benetech_ocr": 0.0,
        "benetech_value": 0.0,
        "benetech_type_acc": 0.0,
    }

    pred_obj = _extract_json(pred_text)
    if pred_obj is None:
        return zero

    pred_x, pred_y, pred_chart_type = _load_pred_series(pred_obj)

    chart_type_match = gt_chart_type == pred_chart_type
    type_acc = 1.0 if chart_type_match else 0.0

    if chart_type_match:
        x_score = _score_series(gt_x, pred_x)
        y_score = _score_series(gt_y, pred_y)
    else:
        x_score = 0.0
        y_score = 0.0

    overall = (x_score + y_score) / 2

    x_is_str = _is_str_series(gt_x)
    y_is_str = _is_str_series(gt_y)

    ocr_scores = []
    value_scores = []
    for score, is_str in [(x_score, x_is_str), (y_score, y_is_str)]:
        if is_str:
            ocr_scores.append(score)
        else:
            value_scores.append(score)

    ocr_mean = float(np.mean(ocr_scores)) if ocr_scores else overall
    value_mean = float(np.mean(value_scores)) if value_scores else overall

    return {
        "benetech_overall": overall,
        "benetech_ocr": ocr_mean,
        "benetech_value": value_mean,
        "benetech_type_acc": type_acc,
    }

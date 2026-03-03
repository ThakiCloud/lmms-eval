import csv
import io
import re

import numpy as np
from rapidfuzz.distance.Levenshtein import distance as levenshtein_distance
from sklearn.metrics import r2_score


def chartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def chartqa_process_results(doc, results):
    pred = results[0]
    type = doc["type"]
    score = relaxed_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict


def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


# ---------------------------------------------------------------------------
# Chart Parsing: Benetech 확장 메트릭
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Benetech 공식 sigmoid: 점수를 0~1 범위로 매핑."""
    return 2 - 2 / (1 + np.exp(-x))


def _normalized_rmse(y_true: list, y_pred: list) -> float:
    """숫자 시리즈 점수. sigmoid(sqrt(1 - R²))"""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return 0.0
    y_true, y_pred = y_true[valid], y_pred[valid]

    return float(_sigmoid((1 - r2_score(y_true, y_pred)) ** 0.5))


def _normalized_levenshtein(y_true: list, y_pred: list) -> float:
    """문자열 시리즈 점수. sigmoid(total_dist / total_len)"""
    total_dist = sum(levenshtein_distance(str(yt), str(yp)) for yt, yp in zip(y_true, y_pred))
    total_len = sum(len(str(yt)) for yt in y_true)
    if total_len == 0:
        return 0.0
    return float(_sigmoid(total_dist / total_len))


def _strip_unit(val: str) -> str:
    """숫자 값에서 단위/기호 제거: '6.12%' → '6.12', '175.09 g' → '175.09', '$1,234' → '1234'"""
    s = str(val).strip()
    s = s.replace(",", "")
    s = s.strip("$€£¥₩%")
    s = re.sub(r"\s*[a-zA-Z°]+\.?\s*$", "", s)
    return s.strip()


def _is_numeric(val) -> bool:
    try:
        float(_strip_unit(str(val)))
        return True
    except (ValueError, TypeError):
        return False


def _to_float_safe(val) -> float:
    return float(_strip_unit(str(val)))


def _score_series(gt_series: list, pred_series: list) -> float:
    """
    단일 컬럼(시리즈) 점수.
    - 길이 불일치 → 0점
    - GT가 숫자 → 단위 제거 후 RMSE
    - GT가 문자열 → Levenshtein
    """
    if len(gt_series) != len(pred_series) or len(gt_series) == 0:
        return 0.0

    if all(_is_numeric(v) for v in gt_series if str(v).strip()):
        try:
            pred_nums = [_to_float_safe(x) for x in pred_series]
        except (ValueError, TypeError):
            return 0.0
        gt_nums = [_to_float_safe(x) for x in gt_series]
        return _normalized_rmse(gt_nums, pred_nums)
    else:
        return _normalized_levenshtein(
            [str(x) for x in gt_series],
            [str(x) for x in pred_series],
        )


# ---------------------------------------------------------------------------
# 테이블 파서
# ---------------------------------------------------------------------------


def _parse_csv_to_columns(csv_string: str) -> list[list[str]]:
    """CSV 문자열 → 컬럼별 리스트. columns[0]은 첫번째 컬럼 전체(헤더+데이터)."""
    reader = csv.reader(io.StringIO(csv_string.strip()))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        return []

    num_cols = len(rows[0])
    columns = []
    for col_idx in range(num_cols):
        col_data = [rows[row_idx][col_idx].strip() if col_idx < len(rows[row_idx]) else "" for row_idx in range(1, len(rows))]
        columns.append(col_data)
    return columns


def _parse_markdown_table(md_string: str) -> list[list[str]]:
    """마크다운 테이블 → 컬럼별 리스트. 구분선(---) 행은 건너뜀."""
    md_string = md_string.strip()

    # 코드 블록 안의 테이블 추출
    code_block = re.search(r"```(?:markdown)?\s*\n(.*?)\n```", md_string, re.DOTALL)
    if code_block:
        md_string = code_block.group(1).strip()

    lines = md_string.split("\n")
    table_lines = []
    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        # 구분선 행 건너뜀 (| --- | --- | 등)
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.match(r"^[-:]+$", c) for c in cells if c):
            continue
        table_lines.append(cells)

    if len(table_lines) < 2:
        return []

    num_cols = len(table_lines[0])
    columns = []
    for col_idx in range(num_cols):
        col_data = [table_lines[row_idx][col_idx].strip() if col_idx < len(table_lines[row_idx]) else "" for row_idx in range(1, len(table_lines))]
        columns.append(col_data)
    return columns


# ---------------------------------------------------------------------------
# Robust 메트릭용 행렬 파서 및 행 정렬
# ---------------------------------------------------------------------------


def _parse_csv_to_matrix(csv_string: str) -> list[list[str]]:
    """CSV 문자열 → 전체 행렬 (헤더 포함). matrix[0]은 헤더 행."""
    reader = csv.reader(io.StringIO(csv_string.strip()))
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    return rows if len(rows) >= 2 else []


def _parse_markdown_to_matrix(md_string: str) -> list[list[str]]:
    """마크다운 테이블 → 전체 행렬 (헤더 포함). matrix[0]은 헤더 행."""
    md_string = md_string.strip()
    code_block = re.search(r"```(?:markdown)?\s*\n(.*?)\n```", md_string, re.DOTALL)
    if code_block:
        md_string = code_block.group(1).strip()

    lines = md_string.split("\n")
    table_lines = []
    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.match(r"^[-:]+$", c) for c in cells if c):
            continue
        table_lines.append(cells)
    return table_lines if len(table_lines) >= 2 else []


def _matrix_to_columns(matrix: list[list[str]]) -> list[list[str]]:
    """행렬(헤더 포함)에서 데이터 컬럼 추출 (헤더 행 제외)."""
    if len(matrix) < 2:
        return []
    num_cols = len(matrix[0])
    columns = []
    for col_idx in range(num_cols):
        col_data = [matrix[row_idx][col_idx] if col_idx < len(matrix[row_idx]) else "" for row_idx in range(1, len(matrix))]
        columns.append(col_data)
    return columns


def _transpose_full_matrix(matrix: list[list[str]]) -> list[list[str]]:
    """헤더 포함 전체 행렬을 전치. [[H1,H2],[a,b],[c,d]] → [[H1,a,c],[H2,b,d]]"""
    if not matrix:
        return []
    max_cols = max(len(row) for row in matrix)
    transposed = []
    for col_idx in range(max_cols):
        new_row = [matrix[row_idx][col_idx] if col_idx < len(matrix[row_idx]) else "" for row_idx in range(len(matrix))]
        transposed.append(new_row)
    return transposed


def _align_rows_by_label(gt_labels: list[str], pred_labels: list[str]) -> list[int]:
    """
    GT 라벨 기준으로 pred 행의 최적 매칭 인덱스를 반환 (greedy, 유사도 내림차순).
    alignment[i] = pred 행 인덱스 (-1이면 미매칭).
    """
    n_gt = len(gt_labels)
    n_pred = len(pred_labels)
    if n_gt == 0 or n_pred == 0:
        return [-1] * n_gt

    pairs = []
    for gi, gl in enumerate(gt_labels):
        gt_str = str(gl).strip()
        for pi, pl in enumerate(pred_labels):
            pred_str = str(pl).strip()
            max_len = max(len(gt_str), len(pred_str), 1)
            sim = 1.0 - levenshtein_distance(gt_str, pred_str) / max_len
            pairs.append((sim, gi, pi))

    pairs.sort(key=lambda x: x[0], reverse=True)

    alignment = [-1] * n_gt
    used_pred: set[int] = set()
    for sim, gi, pi in pairs:
        if alignment[gi] != -1 or pi in used_pred:
            continue
        if sim >= 0.3:
            alignment[gi] = pi
            used_pred.add(pi)
    return alignment


def _score_table_robust(gt_columns: list[list[str]], pred_columns: list[list[str]]) -> tuple[float, float, float]:
    """
    Robust 스코어링: 라벨 기반 행 정렬 후 매칭된 행만 채점.
    미매칭 행은 coverage 비율로 감점 (0.75 매칭 → 점수 × 0.75).
    """
    if not gt_columns or not pred_columns:
        return 0.0, 0.0, 0.0

    n_gt_rows = len(gt_columns[0])
    n_pred_rows = len(pred_columns[0]) if pred_columns else 0
    if n_gt_rows == 0 or n_pred_rows == 0:
        return 0.0, 0.0, 0.0

    while len(pred_columns) < len(gt_columns):
        pred_columns.append([""] * n_pred_rows)
    pred_columns = pred_columns[: len(gt_columns)]

    alignment = _align_rows_by_label(gt_columns[0], pred_columns[0])
    matched_gt_indices = [i for i, a in enumerate(alignment) if a != -1]

    if not matched_gt_indices:
        return 0.0, 0.0, 0.0

    coverage = len(matched_gt_indices) / n_gt_rows

    aligned_gt_columns = []
    aligned_pred_columns = []
    for col_idx in range(len(gt_columns)):
        gt_col = [gt_columns[col_idx][i] for i in matched_gt_indices]
        pred_col = [pred_columns[col_idx][alignment[i]] for i in matched_gt_indices]
        aligned_gt_columns.append(gt_col)
        aligned_pred_columns.append(pred_col)

    label_score = _score_series(aligned_gt_columns[0], aligned_pred_columns[0])

    value_scores = []
    for i in range(1, len(aligned_gt_columns)):
        score = _score_series(aligned_gt_columns[i], aligned_pred_columns[i])
        value_scores.append(score)

    avg_value = float(np.mean(value_scores)) if value_scores else 0.0
    all_scores = [label_score] + value_scores
    raw_overall = float(np.mean(all_scores))

    return raw_overall * coverage, label_score * coverage, avg_value * coverage


# ---------------------------------------------------------------------------
# lmms-eval 태스크 인터페이스
# ---------------------------------------------------------------------------


def chartqa_parsing_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def chartqa_parsing_doc_to_text(doc, lmms_eval_specific_kwargs):
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "Extract all data from this chart as a markdown table.")
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    return f"{pre_prompt}{post_prompt}"


def _transpose_columns(columns: list[list[str]]) -> list[list[str]]:
    """컬럼 리스트를 전치(transpose). [[a,b],[c,d]] → [[a,c],[b,d]]"""
    if not columns or not columns[0]:
        return columns
    n_rows = len(columns[0])
    n_cols = len(columns)
    transposed = []
    for r in range(n_rows):
        transposed.append([columns[c][r] if r < len(columns[c]) else "" for c in range(n_cols)])
    return transposed


def _score_table(gt_columns, pred_columns):
    """GT와 예측 컬럼을 비교해서 (overall, label, value) 점수 반환."""
    if not gt_columns or not pred_columns:
        return 0.0, 0.0, 0.0

    while len(pred_columns) < len(gt_columns):
        pred_columns.append([])
    pred_columns = pred_columns[: len(gt_columns)]

    label_score = _score_series(gt_columns[0], pred_columns[0])

    value_scores = []
    for i in range(1, len(gt_columns)):
        score = _score_series(gt_columns[i], pred_columns[i])
        value_scores.append(score)

    avg_value = float(np.mean(value_scores)) if value_scores else 0.0
    all_scores = [label_score] + value_scores
    overall = float(np.mean(all_scores))
    return overall, label_score, avg_value


def chartqa_parsing_process_results(doc, results):
    pred = results[0]
    gt_csv = doc["table_csv"]

    gt_columns = _parse_csv_to_columns(gt_csv)
    pred_columns = _parse_markdown_table(pred)

    zero_result = {"parsing_overall": 0.0, "parsing_label": 0.0, "parsing_value": 0.0, "robust_overall": 0.0, "robust_label": 0.0, "robust_value": 0.0}

    if not gt_columns or not pred_columns:
        return zero_result

    # --- Strict 메트릭 (기존): 위치 기반 비교 ---
    overall, label_score, avg_value = _score_table(gt_columns, pred_columns)

    if len(pred_columns) >= 2 and len(pred_columns[0]) >= 2:
        transposed = _transpose_columns(pred_columns)
        t_overall, t_label, t_value = _score_table(gt_columns, transposed)
        if t_overall > overall:
            overall, label_score, avg_value = t_overall, t_label, t_value

    # --- Robust 메트릭: 라벨 기반 행 정렬 + 헤더 포함 full transpose ---
    gt_matrix = _parse_csv_to_matrix(gt_csv)
    pred_matrix = _parse_markdown_to_matrix(pred)

    if not gt_matrix or not pred_matrix:
        r_overall, r_label, r_value = 0.0, 0.0, 0.0
    else:
        gt_cols_r = _matrix_to_columns(gt_matrix)
        pred_cols_orig = _matrix_to_columns(pred_matrix)
        r_overall, r_label, r_value = _score_table_robust(gt_cols_r, pred_cols_orig)

        pred_matrix_t = _transpose_full_matrix(pred_matrix)
        pred_cols_t = _matrix_to_columns(pred_matrix_t)
        if pred_cols_t:
            rt_overall, rt_label, rt_value = _score_table_robust(gt_cols_r, pred_cols_t)
            if rt_overall > r_overall:
                r_overall, r_label, r_value = rt_overall, rt_label, rt_value

    return {
        "parsing_overall": overall,
        "parsing_label": label_score,
        "parsing_value": avg_value,
        "robust_overall": r_overall,
        "robust_label": r_label,
        "robust_value": r_value,
    }

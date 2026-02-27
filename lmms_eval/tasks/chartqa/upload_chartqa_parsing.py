"""
ChartQA Parsing 데이터셋 업로드 스크립트

차트 이미지 + GT 테이블(CSV)을 HuggingFace 데이터셋으로 변환.
기존 ChartQA의 QA 데이터가 아닌, 차트 파싱 평가용 데이터셋.

각 차트 이미지에 대응하는 CSV 테이블을 GT로 포함:
  - image: 차트 이미지
  - image_id: 이미지 파일명 (확장자 제외)
  - table_csv: GT 테이블 원문 (CSV 문자열)
  - num_columns: 컬럼 수
  - num_rows: 데이터 행 수 (헤더 제외)

Usage:
    python upload_chartqa_parsing.py
"""

import csv
import io
import os

import datasets
from PIL import Image

CHARTQA_ROOT = "/data/workspace/hongcheol/ChartQA/ChartQA Dataset"

_CITATION = """\
@inproceedings{masry-etal-2022-chartqa,
    title = "{C}hart{QA}: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning",
    author = "Masry, Ahmed  and Long, Do  and Tan, Jia Qing  and Joty, Shafiq  and Hoque, Enamul",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    year = "2022",
    url = "https://aclanthology.org/2022.findings-acl.177",
    pages = "2263--2279",
}
"""

_DESCRIPTION = "ChartQA Parsing: chart image to underlying data table extraction benchmark."


dataset_features = {
    "image": datasets.Image(),
    "image_id": datasets.Value("string"),
    "table_csv": datasets.Value("string"),
    "num_columns": datasets.Value("int32"),
    "num_rows": datasets.Value("int32"),
}


def read_csv_info(csv_path):
    """CSV 파일을 읽어서 원문 문자열, 컬럼 수, 행 수를 반환."""
    with open(csv_path, "r", encoding="utf-8") as f:
        content = f.read()

    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if len(rows) == 0:
        return content, 0, 0
    num_columns = len(rows[0])
    num_rows = len(rows) - 1  # 헤더 제외
    return content, num_columns, num_rows


def generate_split(image_dir, table_dir):
    """이미지와 테이블을 매칭하여 데이터셋 생성."""
    table_files = {f.replace(".csv", ""): os.path.join(table_dir, f) for f in os.listdir(table_dir) if f.endswith(".csv")}

    index = 0
    for img_file in sorted(os.listdir(image_dir)):
        if not img_file.endswith(".png"):
            continue
        image_id = img_file.replace(".png", "")
        if image_id not in table_files:
            continue

        image_path = os.path.join(image_dir, img_file)
        csv_path = table_files[image_id]
        table_csv, num_columns, num_rows = read_csv_info(csv_path)

        yield index, {
            "image": Image.open(image_path).convert("RGB"),
            "image_id": image_id,
            "table_csv": table_csv,
            "num_columns": num_columns,
            "num_rows": num_rows,
        }
        index += 1


class ChartQAParsing(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="ChartQAParsing",
            version=datasets.Version("1.0.0"),
            description="ChartQA Parsing benchmark",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(dataset_features),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_dir": os.path.join(CHARTQA_ROOT, "test", "png"),
                    "table_dir": os.path.join(CHARTQA_ROOT, "test", "tables"),
                },
            ),
        ]

    def _generate_examples(self, image_dir, table_dir):
        yield from generate_split(image_dir, table_dir)


if __name__ == "__main__":
    from datasets import load_dataset

    script_path = os.path.abspath(__file__)
    data = load_dataset(script_path)
    print(data)
    print(f"\nTest set size: {len(data['test'])}")
    print(f"Sample: {data['test'][0]}")

    # HuggingFace Hub에 업로드하려면 아래 주석 해제
    # data.push_to_hub("your-org/chartqa-parsing", private=True)

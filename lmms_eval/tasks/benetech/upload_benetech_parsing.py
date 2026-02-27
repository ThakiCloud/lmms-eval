"""
Benetech Parsing 데이터셋 준비 스크립트

Benetech "Making Graphs Accessible" 대회 데이터에서 source=extracted만 추출하여
lmms-eval용 parquet 데이터셋으로 변환.

각 차트 이미지에 대응하는 GT annotation을 포함:
  - image: 차트 이미지
  - image_id: 이미지 파일명 (확장자 제외)
  - chart_type: 차트 유형 (line, vertical_bar, scatter, dot, horizontal_bar)
  - data_series_json: GT data-series JSON 문자열 ([{"x": ..., "y": ...}, ...])
  - num_points: 데이터 포인트 수

Usage:
    python upload_benetech_parsing.py
"""

import json
import os

import datasets
from PIL import Image

BENETECH_ROOT = "/data/workspace/hongcheol/thaki-document-understanding/data/benetech/train"

_CITATION = """\
@misc{benetech2023,
    title = {Benetech - Making Graphs Accessible},
    year = {2023},
    url = {https://www.kaggle.com/competitions/benetech-making-graphs-accessible},
}
"""

_DESCRIPTION = "Benetech Parsing: chart image to structured data series extraction benchmark (extracted subset only)."


dataset_features = {
    "image": datasets.Image(),
    "image_id": datasets.Value("string"),
    "chart_type": datasets.Value("string"),
    "data_series_json": datasets.Value("string"),
    "num_points": datasets.Value("int32"),
}


def generate_extracted_split(ann_dir: str, image_dir: str):
    """source=extracted인 annotation만 필터링하여 데이터셋 생성."""
    index = 0
    for ann_file in sorted(os.listdir(ann_dir)):
        if not ann_file.endswith(".json"):
            continue

        ann_path = os.path.join(ann_dir, ann_file)
        with open(ann_path, encoding="utf-8") as f:
            ann = json.load(f)

        if ann.get("source") != "extracted":
            continue

        image_id = ann_file.replace(".json", "")
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            continue

        data_series = ann.get("data-series", [])
        chart_type = ann.get("chart-type", "unknown")

        yield index, {
            "image": Image.open(image_path).convert("RGB"),
            "image_id": image_id,
            "chart_type": chart_type,
            "data_series_json": json.dumps(data_series, ensure_ascii=False),
            "num_points": len(data_series),
        }
        index += 1


class BenetechParsing(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="BenetechParsing",
            version=datasets.Version("1.0.0"),
            description="Benetech Parsing benchmark (extracted only)",
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
                    "ann_dir": os.path.join(BENETECH_ROOT, "annotations"),
                    "image_dir": os.path.join(BENETECH_ROOT, "images"),
                },
            ),
        ]

    def _generate_examples(self, ann_dir, image_dir):
        yield from generate_extracted_split(ann_dir, image_dir)


if __name__ == "__main__":
    import datasets as ds

    ann_dir = os.path.join(BENETECH_ROOT, "annotations")
    image_dir = os.path.join(BENETECH_ROOT, "images")

    rows = []
    for idx, row in generate_extracted_split(ann_dir, image_dir):
        rows.append(row)
        if idx % 100 == 0:
            print(f"  Processed {idx + 1} samples...")

    print(f"\nTotal extracted samples: {len(rows)}")

    data = ds.Dataset.from_dict(
        {k: [r[k] for r in rows] for k in dataset_features.keys()},
        features=ds.Features(dataset_features),
    )
    print(data)
    print(f"Sample chart_type: {data[0]['chart_type']}")
    print(f"Sample num_points: {data[0]['num_points']}")

    script_path = os.path.abspath(__file__)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path)))), "data", "benetech_parsing")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test.parquet")
    data.to_parquet(output_path)
    print(f"\nSaved to: {output_path}")

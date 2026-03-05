# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.
#
# Adapted from PubTabNet (https://github.com/ibm-aur-nlp/PubTabNet)
# for integration with lmms-eval framework.

from __future__ import annotations

from collections import deque

import editdistance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html


class TableTree(Tree):
    def __init__(self, tag: str, colspan: int | None = None, rowspan: int | None = None, content: list | None = None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self) -> str:
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences) -> int:
        return max(map(len, sequences))

    def normalized_distance(self, *sequences) -> float:
        return float(editdistance.eval(*sequences)) / self.maximum(*sequences)

    def rename(self, node1: TableTree, node2: TableTree) -> float:
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Tree Edit Distance based Similarity for HTML table evaluation.

    Computes structural and content similarity between predicted and
    ground-truth HTML tables using the APTED algorithm.
    """

    def __init__(self, structure_only: bool = False, ignore_nodes: list[str] | None = None):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__: list[str] = []

    def tokenize(self, node) -> None:
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent: TableTree | None = None) -> TableTree | None:
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag, int(node.attrib.get("colspan", "1")), int(node.attrib.get("rowspan", "1")), cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    @staticmethod
    def _find_table(node):
        """Find <table> element from parsed HTML, handling various structures."""
        if node.tag == "table":
            return node
        tables = node.xpath("body/table")
        if tables:
            return tables[0]
        tables = node.xpath(".//table")
        if tables:
            return tables[0]
        return None

    @staticmethod
    def _normalize_table(node):
        """Normalize a <table> element for fair comparison.

        Strips thead/tbody wrappers, inline formatting tags, and converts th to td.
        """
        etree.strip_tags(node, "thead", "tbody", "b", "i", "sup", "sub")
        for th in node.xpath(".//th"):
            th.tag = "td"
        return node

    def evaluate(self, pred: str, true: str) -> float:
        """Compute TEDS score between predicted and ground-truth HTML tables.

        Returns a float in [0, 1] where 1.0 is a perfect match.
        """
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred_tree = html.fromstring(pred, parser=parser)
        true_tree = html.fromstring(true, parser=parser)
        pred_table = self._find_table(pred_tree)
        true_table = self._find_table(true_tree)
        if pred_table is not None and true_table is not None:
            pred_table = self._normalize_table(pred_table)
            true_table = self._normalize_table(true_table)
            if self.ignore_nodes:
                etree.strip_tags(pred_table, *self.ignore_nodes)
                etree.strip_tags(true_table, *self.ignore_nodes)
            n_nodes_pred = len(pred_table.xpath(".//*"))
            n_nodes_true = len(true_table.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            if n_nodes == 0:
                return 1.0
            tree_pred = self.load_html_tree(pred_table)
            tree_true = self.load_html_tree(true_table)
            dist = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(dist) / n_nodes)
        else:
            return 0.0


class OCRMetric:
    """OCR-only evaluation: compares cell text content ignoring table structure.

    Extracts all cell texts from prediction and ground truth HTML tables,
    concatenates them in reading order, and computes normalised Levenshtein
    similarity (1 - NED).
    """

    @staticmethod
    def _extract_cell_texts(html_string: str) -> str:
        if not html_string:
            return ""
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        doc = html.fromstring(html_string, parser=parser)

        table = doc if doc.tag == "table" else None
        if table is None:
            tables = doc.xpath("body/table")
            if not tables:
                tables = doc.xpath(".//table")
            table = tables[0] if tables else None
        if table is None:
            return ""

        etree.strip_tags(table, "thead", "tbody", "b", "i", "sup", "sub")
        for th in table.xpath(".//th"):
            th.tag = "td"

        cells = table.xpath(".//td")
        texts = []
        for cell in cells:
            cell_text = etree.tostring(cell, method="text", encoding="unicode") or ""
            cell_text = " ".join(cell_text.split())
            if cell_text:
                texts.append(cell_text)
        return " ".join(texts)

    def evaluate(self, pred: str, true: str) -> float:
        """Compute OCR similarity for a single sample.

        Returns a float in [0, 1] where 1.0 is a perfect match.
        """
        pred_text = self._extract_cell_texts(pred)
        true_text = self._extract_cell_texts(true)
        if not pred_text and not true_text:
            return 1.0
        if not pred_text or not true_text:
            return 0.0
        max_len = max(len(pred_text), len(true_text))
        dist = editdistance.eval(pred_text, true_text)
        return 1.0 - (dist / max_len)


# ---------------------------------------------------------------------------
# Picklable worker functions for ProcessPoolExecutor (spawn mode).
# Defined here because this module is importable by its fully-qualified name
# (lmms_eval.tasks.icdar_table_parsing.teds), unlike the YAML-loaded utils.py.
# ---------------------------------------------------------------------------


def compute_teds_full(args: tuple) -> float:
    pred_html, gt_html, _ = args
    try:
        return TEDS(structure_only=False).evaluate(pred_html, gt_html)
    except Exception:
        import traceback
        traceback.print_exc()
        return 0.0


def compute_teds_struct(args: tuple) -> float:
    pred_html, gt_html, _ = args
    try:
        return TEDS(structure_only=True).evaluate(pred_html, gt_html)
    except Exception:
        import traceback
        traceback.print_exc()
        return 0.0


def compute_ocr(args: tuple) -> float:
    pred_html, gt_html, _ = args
    try:
        return OCRMetric().evaluate(pred_html, gt_html)
    except Exception:
        import traceback
        traceback.print_exc()
        return 0.0

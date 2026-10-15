# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression for competing formula/code proposals and operator-led continuations."""

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster
from docling.datamodel.pipeline_options import LayoutPostprocessorOptions
from docling.utils.layout_postprocessor import LayoutPostprocessor


def cell(index: int, text: str, box: tuple[float, float, float, float]) -> TextCell:
    return TextCell(
        index=index,
        text=text,
        orig=text,
        from_ocr=False,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3])
        ),
    )


class PageStub:
    size = Size(width=600, height=800)
    parsed_page = None

    def __init__(self, cells: list[TextCell]) -> None:
        self.cells = cells


def process(
    *,
    enabled: bool = True,
    formula_confidence: float = 0.446615,
    code_confidence: float = 0.721637,
    fraction: bool = True,
    continuation: str = "\N{MULTIPLICATION SIGN} Annual emissions",
    continuation_left: float = 117,
    continuation_top: float = 377,
    skip_cells: bool = False,
    formula_proposal: bool = True,
    native: bool = True,
    bar_top: float = 356,
) -> list[Cluster]:
    # Geometry and confidences follow the failing two-line exhibit. Text is synthetic.
    bbox = BoundingBox(l=118, t=352, r=363, b=375)
    clusters = [
        Cluster(id=15, label=DocItemLabel.CODE, bbox=bbox, confidence=code_confidence)
    ]
    if formula_proposal:
        clusters.append(
            Cluster(
                id=27,
                label=DocItemLabel.FORMULA,
                bbox=bbox,
                confidence=formula_confidence,
            )
        )
    cells = [
        cell(25, "Attributed total = [ Exposure", (120, 350, 349, 374)),
        cell(27, "Adjusted denominator ]", (220, 350, 361, 374)),
    ]
    if fraction:
        cells.append(cell(26, "________________", (220, bar_top, 356, bar_top + 7)))
    if continuation:
        box = (
            continuation_left,
            continuation_top,
            continuation_left + 166,
            continuation_top + 10,
        )
        cells.append(cell(28, continuation, box))
        clusters.append(
            Cluster(
                id=14,
                label=DocItemLabel.LIST_ITEM,
                bbox=BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3]),
                confidence=0.722584,
            )
        )
    for source_cell in cells:
        source_cell.from_ocr = not native
    options = LayoutPostprocessorOptions(
        recover_formula_regions=enabled,
        skip_cell_assignment=skip_cells,
    )
    return LayoutPostprocessor(PageStub(cells), clusters, options).postprocess()


@pytest.mark.parametrize(
    "formula_confidence,code_confidence", [(0.487694, 0.470074), (0.446615, 0.721637)]
)
def test_fraction_evidence_recovers_formula_and_operator_continuation(
    formula_confidence: float,
    code_confidence: float,
) -> None:
    result = process(
        formula_confidence=formula_confidence, code_confidence=code_confidence
    )
    assert len(result) == 1
    assert result[0].label == DocItemLabel.FORMULA
    assert result[0].confidence == formula_confidence
    assert {c.index for c in result[0].cells} == {25, 26, 27, 28}
    assert result[0].bbox.b == 387


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": False},
        {"fraction": False},
        {"formula_confidence": 0.1},
        {"skip_cells": True},
        {"formula_proposal": False},
        {"native": False},
        {"bar_top": 375},
    ],
)
def test_code_is_not_reclassified_without_opt_in_and_source_evidence(
    kwargs: dict,
) -> None:
    result = process(**kwargs)
    assert any(c.label == DocItemLabel.CODE for c in result)
    assert not any(c.label == DocItemLabel.FORMULA for c in result)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"continuation": "• A genuine list item"},
        {"continuation": "+ A list item with an ambiguous marker"},
        {"continuation_left": 380},
        {"continuation_top": 430},
    ],
)
def test_unrelated_list_item_is_not_absorbed(kwargs: dict) -> None:
    result = process(**kwargs)
    assert {c.label for c in result} == {DocItemLabel.FORMULA, DocItemLabel.LIST_ITEM}
    formula = next(c for c in result if c.label == DocItemLabel.FORMULA)
    assert 28 not in {c.index for c in formula.cells}


def test_arithmetic_code_with_an_underline_separator_remains_code() -> None:
    bbox = BoundingBox(l=10, t=10, r=300, b=45)
    cells = [
        cell(1, "result = numerator / denominator", (10, 10, 290, 20)),
        cell(2, "________________", (10, 23, 290, 26)),
        cell(3, "print(result)", (10, 30, 200, 40)),
    ]
    clusters = [
        Cluster(id=1, label=DocItemLabel.CODE, bbox=bbox, confidence=0.9),
        Cluster(id=2, label=DocItemLabel.FORMULA, bbox=bbox, confidence=0.48),
    ]
    result = LayoutPostprocessor(
        PageStub(cells),
        clusters,
        LayoutPostprocessorOptions(recover_formula_regions=True),
    ).postprocess()
    assert len(result) == 1
    assert result[0].label == DocItemLabel.CODE
    assert [c.text for c in result[0].cells] == [c.text for c in cells]

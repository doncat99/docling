# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ProvenanceItem,
)
from docling_core.types.doc.base import BoundingBox, CoordOrigin

from docling.datamodel.base_models import Cluster, TextElement
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)


def _bounding_box() -> BoundingBox:
    return BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)


def _list_item(text: str):
    document = DoclingDocument(
        name="test",
        origin=DocumentOrigin(
            filename="test.pdf", mimetype="application/pdf", binary_hash="0" * 64
        ),
    )
    list_group = document.add_group(label=GroupLabel.LIST, name="list")
    return document.add_list_item(
        text=text,
        prov=ProvenanceItem(page_no=1, charspan=(0, len(text)), bbox=_bounding_box()),
        parent=list_group,
    )


def _element(cluster_id: int, text: str) -> TextElement:
    return TextElement(
        id=cluster_id,
        page_no=1,
        label=DocItemLabel.LIST_ITEM,
        text=text,
        cluster=Cluster(
            id=cluster_id,
            label=DocItemLabel.LIST_ITEM,
            bbox=_bounding_box(),
        ),
    )


@pytest.mark.parametrize(
    ("prefix", "continuation", "expected"),
    [
        ("algo-", "rithms", "algorithms"),
        ("algo\u00ad", "rithms", "algorithms"),
        ("algo-", "Rithms", "algo- Rithms"),
    ],
)
def test_merge_elements_dehyphenates_lowercase_continuations(
    prefix: str, continuation: str, expected: str
) -> None:
    item = _list_item(prefix)
    model = ReadingOrderModel(ReadingOrderOptions())

    model._merge_elements(
        _element(1, prefix),
        _element(2, continuation),
        item,
        page_height=100,
    )

    assert item.text == expected
    assert item.orig == expected


@pytest.mark.parametrize("prefix", ["algo-", "algo\u00ad", "A😀-"])
def test_cross_page_merge_preserves_exact_character_provenance(prefix: str) -> None:
    item = _list_item(prefix)
    continuation = _element(2, "rithms-")
    continuation.page_no = 2
    model = ReadingOrderModel(ReadingOrderOptions())
    model._merge_elements(_element(1, prefix), continuation, item, page_height=100)
    last = _element(3, "continued")
    last.page_no = 3
    model._merge_elements(continuation, last, item, page_height=100)

    boundary = len(prefix) - 1
    assert [prov.charspan for prov in item.prov] == [
        (0, boundary),
        (boundary, boundary + 6),
        (boundary + 6, len(item.text)),
    ]
    assert [item.text[start:end] for start, end in (p.charspan for p in item.prov)] == [
        prefix[:-1],
        "rithms",
        "continued",
    ]
    assert [prov.page_no for prov in item.prov] == [1, 2, 3]


def test_nonhyphenated_merge_keeps_separator_outside_page_provenance() -> None:
    item = _list_item("algo-")
    ReadingOrderModel(ReadingOrderOptions())._merge_elements(
        _element(1, "algo-"),
        _element(2, "Rithms"),
        item,
        page_height=100,
    )
    assert [prov.charspan for prov in item.prov] == [(0, 5), (6, 12)]

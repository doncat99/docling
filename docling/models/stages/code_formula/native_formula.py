# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Conservative transcription of native PDF fractions with encoded rule glyphs.

This handles horizontal, non-nested fractions whose bars are underscore text
cells. It does not infer missing glyphs or repair OCR. Every word must be used
exactly once. Unsupported geometry leaves the existing formula unchanged.
"""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median

from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    TextItem,
)
from docling_core.types.doc.page import TextCell

from docling.datamodel.document import ConversionResult
from docling.datamodel.progress import EnrichmentStep
from docling.models.base_model import GenericEnrichmentModel


@dataclass(frozen=True)
class _Token:
    text: str
    box: BoundingBox
    latex: bool = False


def _escape(text: str) -> str:
    return re.sub(r"([%&#${}_])", r"\\\1", text)


def _render_run(tokens: list[_Token]) -> str | None:
    """Render one baseline, separating source subscripts from prose labels."""
    result: list[str] = []
    prose: list[str] = []

    def flush() -> None:
        if prose:
            result.append(r"\text{" + _escape(" ".join(prose)) + "}")
            prose.clear()

    operators = {
        "\N{MULTIPLICATION SIGN}": r"\times",
        "÷": r"\div",
        "⋅": r"\cdot",
        "\N{MINUS SIGN}": "-",
        "=": "=",
        "+": "+",
        "-": "-",
        "[": "[",
        "]": "]",
        "(": "(",
        ")": ")",
        ".": ".",
    }
    # Tall brackets and fraction boxes must not move the surrounding text baseline.
    prose_tokens = [t for t in tokens if not t.latex and t.text not in operators]
    baseline_tokens = prose_tokens or tokens
    height = median(t.box.height for t in baseline_tokens)
    baseline = median(t.box.b for t in baseline_tokens)
    for token in sorted(tokens, key=lambda t: t.box.l):
        text = token.text
        if token.latex or text in operators:
            flush()
            result.append(text if token.latex else operators[text])
        elif len(text) == 1 and text.isascii() and text.isalpha():
            if token.box.b < baseline - height * 0.3:
                return None
            if (
                token.box.height < height * 0.98
                and token.box.b > baseline + height * 0.15
            ):
                flush()
                if not result:
                    return None
                result[-1] += "_{" + text + "}"
            elif prose:
                prose.append(text)
            else:
                result.append(text)
        elif re.fullmatch("[A-Za-z0-9(),.'\N{RIGHT SINGLE QUOTATION MARK} /-]+", text):
            if abs(token.box.b - baseline) > height * 0.3:
                return None
            prose.append(text)
        else:
            return None
    flush()
    return " ".join(result)


def transcribe_native_fraction(cells: list[TextCell]) -> str | None:
    """Return source-faithful LaTeX only for the supported native fraction grammar."""
    if not cells or any(
        c.from_ocr or any(unicodedata.category(ch).startswith("C") for ch in c.text)
        for c in cells
    ):
        return None
    tokens = [
        _Token(c.text.strip(), c.rect.to_bounding_box())
        for c in cells
        if c.text.strip()
    ]
    bars = [t for t in tokens if re.fullmatch(r"_{3,}", t.text)]
    if not bars:
        return None
    # Nested or horizontally overlapping bars need a different grammar.
    if any(
        min(a.box.r, b.box.r) > max(a.box.l, b.box.l)
        for i, a in enumerate(bars)
        for b in bars[i + 1 :]
    ):
        return None
    remaining = [t for t in tokens if t not in bars]
    for bar in bars:
        box = bar.box
        operands = [
            t
            for t in remaining
            if t.box.l >= box.l - 1
            and t.box.r <= box.r + 1
            and t.box.t >= box.t - 1.5 * box.height
            and t.box.b <= box.b + 2 * box.height
        ]
        numerator = [t for t in operands if (t.box.t + t.box.b) / 2 < box.b]
        denominator = [t for t in operands if t not in numerator]
        if not numerator or not denominator:
            return None
        top = _render_run(numerator)
        bottom = _render_run(denominator)
        if top is None or bottom is None:
            return None
        remaining = [t for t in remaining if t not in operands]
        remaining.append(
            _Token(
                r"\frac{" + top + "}{" + bottom + "}",
                BoundingBox.enclosing_bbox([t.box for t in [bar, *operands]]),
                True,
            )
        )
    for summation in [t for t in remaining if t.text == "∑"]:
        center = (summation.box.t + summation.box.b) / 2
        limits = [
            t
            for t in remaining
            if 0 <= t.box.l - summation.box.r <= summation.box.height * 0.5
            and len(t.text) == 1
            and t.text.isascii()
            and t.text.isalnum()
            and abs((t.box.t + t.box.b) / 2 - center) < summation.box.height
        ]
        upper = [t for t in limits if (t.box.t + t.box.b) / 2 < center]
        lower = [t for t in limits if t not in upper]
        if len(upper) != 1 or len(lower) != 1:
            return None
        remaining = [t for t in remaining if t not in [summation, *limits]]
        remaining.append(
            _Token(
                r"\sum_{" + lower[0].text + "}^{" + upper[0].text + "}",
                summation.box,
                True,
            )
        )
    # Group by vertical centre; full-height parentheses stay on the fraction line.
    rows: list[list[_Token]] = []
    for token in sorted(remaining, key=lambda t: (t.box.t + t.box.b) / 2):
        center = (token.box.t + token.box.b) / 2
        row = next(
            (
                r
                for r in rows
                if abs(center - median((t.box.t + t.box.b) / 2 for t in r))
                <= median(t.box.height for t in r) * 0.6
            ),
            None,
        )
        if row is None:
            rows.append([token])
        else:
            row.append(token)
    rendered = [_render_run(row) for row in rows]
    if any(value is None for value in rendered):
        return None
    # Additional lines must explicitly continue the expression.
    if any(
        not value.startswith((r"\times", r"\div", r"\cdot", "+", "-", "="))
        for value in rendered[1:]
        if value
    ):
        return None
    return " ".join(value for value in rendered if value)


class NativeFormulaModel(GenericEnrichmentModel[tuple[TextItem, str]]):
    """Use native word positions, without a neural model or mutable document cache."""

    progress_step = EnrichmentStep.CODE_FORMULA

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return (
            self.enabled
            and isinstance(element, TextItem)
            and element.label == DocItemLabel.FORMULA
        )

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> tuple[TextItem, str] | None:
        if not self.is_processable(conv_res.document, element):
            return None
        assert isinstance(element, TextItem)
        if len(element.prov) != 1:
            return None
        prov = element.prov[0]
        page = next((p for p in conv_res.pages if p.page_no == prov.page_no), None)
        if page is None or page.parsed_page is None or page.size is None:
            return None
        box = prov.bbox.to_top_left_origin(page.size.height)
        cells = []
        for cell in page.parsed_page.word_cells:
            native = cell.model_copy(deep=True)
            native.to_top_left_origin(page.size.height)
            overlap = native.rect.to_bounding_box().intersection_over_self(box)
            if 0.1 < overlap <= 0.9:
                return None
            if overlap > 0.9:
                cells.append(native)
        text = transcribe_native_fraction(cells)
        return (element, text) if text is not None else None

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[tuple[TextItem, str]]
    ) -> Iterable[NodeItem]:
        for item, text in element_batch:
            item.text = text
            yield item

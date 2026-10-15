# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.models.stages.code_formula.native_formula import transcribe_native_fraction


def word(text, left, top, right, bottom, index=0):
    return TextCell(
        index=index,
        text=text,
        orig=text,
        from_ocr=False,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=left, t=top, r=right, b=bottom)
        ),
    )


def fraction_words():
    return [
        word("Total", 0, 10, 25, 18),
        word("=", 28, 10, 34, 18),
        word("Exposure", 43, 4, 78, 11),
        word("(USD)", 81, 4, 104, 11),
        word("________________", 40, 7, 110, 14),
        word("PPP", 43, 16, 57, 23),
        word("GDP", 60, 16, 78, 23),
        word("\N{MULTIPLICATION SIGN}", 0, 30, 6, 38),
        word("Emissions", 9, 30, 50, 38),
        word("(tCO2e)", 53, 30, 86, 38),
    ]


def test_native_fraction_preserves_labels_units_and_continuation():
    assert transcribe_native_fraction(fraction_words()) == (
        r"\text{Total} = \frac{\text{Exposure (USD)}}{\text{PPP GDP}} "
        r"\times \text{Emissions (tCO2e)}"
    )


def test_sum_limits_follow_source_positions_not_convention():
    cells = fraction_words()[2:7]
    cells += [
        word("∑", 15, 4, 25, 25),
        word("i", 26, 0, 28, 7),
        word("n", 26, 24, 30, 31),
    ]
    assert transcribe_native_fraction(cells) == (
        r"\sum_{n}^{i} \frac{\text{Exposure (USD)}}{\text{PPP GDP}}"
    )


def test_fraction_subscript_is_not_merged_into_the_label():
    cells = [*fraction_words()[2:7], word("i", 105, 7, 107, 13.5)]
    assert transcribe_native_fraction(cells) == (
        r"\frac{\text{Exposure (USD)}_{i}}{\text{PPP GDP}}"
    )


def test_single_letter_unit_in_a_label_is_preserved():
    cells = fraction_words()
    cells[3].text = "USD"
    cells.append(word("m", 105, 4, 108, 11))
    assert r"\text{Exposure USD m}" in transcribe_native_fraction(cells)


def test_tall_fraction_and_brackets_do_not_shift_the_prose_baseline():
    cells = fraction_words()
    cells += [
        word("[", 36, 0, 40, 26),
        word("]", 111, 0, 115, 26),
        word(".", 116, 18, 118, 26),
    ]
    assert transcribe_native_fraction(cells) == (
        r"\text{Total} = [ \frac{\text{Exposure (USD)}}{\text{PPP GDP}} ] . "
        r"\times \text{Emissions (tCO2e)}"
    )


def test_unsupported_superscript_is_not_flattened():
    cells = [*fraction_words(), word("z", 105, 0, 108, 5)]
    assert transcribe_native_fraction(cells) is None


@pytest.mark.parametrize(
    "failure", ["ocr", "no_bar", "empty_denominator", "nested_bar", "unmapped"]
)
def test_ambiguous_or_untrusted_native_content_is_not_transcribed(failure):
    cells = fraction_words()
    if failure == "ocr":
        cells[2].from_ocr = True
    elif failure == "no_bar":
        cells = [c for c in cells if "___" not in c.text]
    elif failure == "empty_denominator":
        cells = [c for c in cells if c.text not in ("PPP", "GDP")]
    elif failure == "nested_bar":
        cells.append(word("________", 45, 5, 70, 12))
    else:
        cells[2].text = "\ufffd"
    assert transcribe_native_fraction(cells) is None

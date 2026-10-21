# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
from unittest.mock import Mock

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import TableStructureOptions
from docling.models.stages.table_structure.table_structure_model import TableStructureModel


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0"])
def test_tableformer_uses_resolved_pipeline_device(monkeypatch, device: str) -> None:
    predictor = Mock()
    monkeypatch.setattr(
        "docling.models.stages.table_structure.table_structure_model.decide_device",
        lambda requested: device,
    )
    monkeypatch.setattr(
        "docling_ibm_models.tableformer.common.read_config",
        lambda path: {"model": {"type": "TableModel04_rs"}},
    )
    monkeypatch.setattr(
        "docling_ibm_models.tableformer.data_management.tf_predictor.TFPredictor",
        predictor,
    )

    TableStructureModel(
        enabled=True,
        artifacts_path=Path("/unused-models"),
        options=TableStructureOptions(),
        accelerator_options=AcceleratorOptions(num_threads=4),
    )

    assert predictor.call_args.args[1:] == (device, 4)

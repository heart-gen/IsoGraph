"""Pydantic validation models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MatrixSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assay_name: str
    filename: str
    n_features: int = Field(gt=0)
    n_samples: int = Field(gt=0)
    encoding: Literal["dense_npz"] = "dense_npz"


class FeatureTableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "gene",
        "transcript",
        "psi",
        "junction",
        "truth_module",
        "truth_switch",
        "truth_switch_event",
        "truth_abundance",
        "truth_channel_role",
    ]
    filename: str
    n_rows: int = Field(ge=0)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    suite_name: str
    description: str
    sample_table: str
    feature_tables: list[FeatureTableSpec]
    matrices: list[MatrixSpec]
    provenance: dict[str, str]
    truth_tables: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tables(self) -> "DatasetManifest":
        if not self.feature_tables:
            raise ValueError("feature_tables must not be empty")
        if not self.matrices:
            raise ValueError("matrices must not be empty")
        return self


class AssayBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_table: str
    gene_table: str
    matrix: str
    n_samples: int = Field(gt=0)
    n_features: int = Field(gt=0)


class FreezeSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gene_panel_size: int | None = Field(default=None, gt=0)
    allowed_diagnoses: list[str]

    @field_validator("allowed_diagnoses")
    @classmethod
    def diagnoses_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("allowed_diagnoses must not be empty")
        return value


class LoadedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: Path
    n_samples: int = Field(gt=0)
    n_genes: int = Field(gt=0)
    available_assays: list[str]

    @model_validator(mode="after")
    def assays_present(self) -> "LoadedDataset":
        if not self.available_assays:
            raise ValueError("available_assays must not be empty")
        return self

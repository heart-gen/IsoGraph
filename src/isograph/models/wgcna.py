"""WGCNA network backend — subprocess wrapper around wgcna_runner.R."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.features.switch import gene_switch_coordinates
from isograph.models.base import FitArtifacts, NetworkModel
from isograph.workflow.config import WgcnaModelConfig

_RUNNER_R = Path(__file__).parent / "wgcna_runner.R"


def _check_rscript() -> None:
    if shutil.which("Rscript") is None:
        raise ImportError(
            "Rscript not found in PATH. Install R and the WGCNA package to use the wgcna backend."
        )


@dataclass
class WgcnaNetworkModel(NetworkModel):
    config: WgcnaModelConfig

    def fit(
        self,
        transcript_counts: np.ndarray,
        transcript_table: pd.DataFrame,
        sample_table: pd.DataFrame,
    ) -> FitArtifacts:
        _check_rscript()

        switch_matrix, feature_info = gene_switch_coordinates(transcript_counts, transcript_table)
        if switch_matrix.size:
            design = build_design_matrix(sample_table, self.config.residualize_covariates)
            switch_matrix = residualize_rows(switch_matrix, design)

        gene_ids = feature_info["gene_id"].tolist()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "gene_matrix.csv"
            output_json = tmp / "wgcna_result.json"

            # Write genes x samples CSV (first column = gene_id)
            df = pd.DataFrame(switch_matrix, index=gene_ids)
            df.index.name = "gene_id"
            df.to_csv(input_csv)

            power_str = (
                "auto" if self.config.power is None else str(self.config.power)
            )
            power_range_str = ",".join(str(p) for p in self.config.power_range)

            cmd = [
                "Rscript",
                str(_RUNNER_R),
                f"input={input_csv}",
                f"output={output_json}",
                f"power={power_str}",
                f"power_range={power_range_str}",
                f"sft_r2={self.config.sft_r2_threshold}",
                f"min_module_size={self.config.min_module_size}",
                f"merge_cut_height={self.config.merge_cut_height}",
                f"deep_split={self.config.deep_split}",
                f"network_type={self.config.network_type}",
                f"seed={self.config.random_state}",
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"WGCNA R script failed (exit {proc.returncode}):\n{proc.stderr}"
                )

            result = json.loads(output_json.read_text())

        module_table = pd.DataFrame(result["modules"])
        if module_table.empty:
            module_table = pd.DataFrame(columns=["gene_id", "module_id"])

        edge_table = pd.DataFrame(result["edges"])
        if edge_table.empty:
            edge_table = pd.DataFrame(columns=["source", "target", "weight"])

        trait_table = pd.DataFrame(columns=["module_id", "trait", "effect", "pvalue"])

        feature_scores = pd.DataFrame(switch_matrix, index=gene_ids)
        feature_scores = feature_scores.reset_index().rename(columns={"index": "gene_id"})

        calibration = result.get("calibration", {})

        return FitArtifacts(
            module_table=module_table,
            edge_table=edge_table,
            trait_table=trait_table,
            feature_scores=feature_scores,
            calibration=calibration,
        )

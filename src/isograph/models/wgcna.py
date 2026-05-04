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

from isograph.features.channels import feature_sample_columns, gene_feature_channels, make_feature_scores
from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.models.base import FitArtifacts, NetworkModel, compute_module_gene_roles
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
        gene_counts: np.ndarray | None = None,
        gene_table: pd.DataFrame | None = None,
    ) -> FitArtifacts:
        _check_rscript()

        switch_matrix, feature_info = gene_feature_channels(
            transcript_counts, transcript_table, gene_counts, gene_table
        )
        if switch_matrix.size:
            design = build_design_matrix(sample_table, self.config.residualize_covariates)
            switch_matrix = residualize_rows(switch_matrix, design)

        feature_ids = feature_info["feature_id"].tolist()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "gene_matrix.csv"
            output_json = tmp / "wgcna_result.json"

            # Write genes x samples CSV (first column = gene_id)
            df = pd.DataFrame(switch_matrix, index=feature_ids)
            df.index.name = "feature_id"
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
        else:
            feature_to_gene = feature_info.set_index("feature_id")["gene_id"].to_dict()
            module_table = module_table.rename(columns={"gene_id": "feature_id"})
            module_table["gene_id"] = module_table["feature_id"].map(feature_to_gene)
            module_table = module_table.dropna(subset=["gene_id"])
            module_table = module_table[["gene_id", "module_id"]].drop_duplicates().reset_index(drop=True)

        edge_table = pd.DataFrame(result["edges"])
        if edge_table.empty:
            edge_table = pd.DataFrame(columns=["source", "target", "weight"])
        else:
            feature_to_gene = feature_info.set_index("feature_id")["gene_id"].to_dict()
            edge_table = edge_table.rename(columns={"source": "source_feature_id", "target": "target_feature_id"})
            edge_table["source"] = edge_table["source_feature_id"].map(feature_to_gene)
            edge_table["target"] = edge_table["target_feature_id"].map(feature_to_gene)
            edge_table = edge_table.dropna(subset=["source", "target"])
            edge_table = edge_table.loc[edge_table["source"] != edge_table["target"]].reset_index(drop=True)

        trait_table = pd.DataFrame(columns=["module_id", "trait", "effect", "pvalue"])

        feature_scores = make_feature_scores(switch_matrix, feature_info, sample_table)

        calibration = result.get("calibration", {})

        sample_ids = sample_table["sample_id"].tolist() if "sample_id" in sample_table.columns else list(range(len(sample_table)))
        score_sample_ids = feature_sample_columns(feature_scores)
        if not module_table.empty:
            eigengene_rows: dict[str, list] = {}
            for module_id in sorted(module_table["module_id"].unique()):
                genes = module_table.loc[module_table["module_id"] == module_id, "gene_id"]
                subset = feature_scores.loc[feature_scores["gene_id"].isin(genes)]
                vec = subset[score_sample_ids].to_numpy(dtype=float).mean(axis=0)
                eigengene_rows[module_id] = vec.tolist()
            eigengene_table: pd.DataFrame = (
                pd.DataFrame(eigengene_rows, index=sample_ids)
                .T.reset_index()
                .rename(columns={"index": "module_id"})
            )
        else:
            eigengene_table = pd.DataFrame(columns=["module_id"] + sample_ids)
        module_gene_roles = compute_module_gene_roles(module_table, feature_scores, sample_table)

        return FitArtifacts(
            module_table=module_table,
            edge_table=edge_table,
            trait_table=trait_table,
            feature_scores=feature_scores,
            calibration=calibration,
            eigengene_table=eigengene_table,
            module_gene_roles=module_gene_roles,
        )

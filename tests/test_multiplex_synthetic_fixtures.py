from __future__ import annotations

import numpy as np

from isograph.benchmarks.synthetic import generate_multiplex_suite
from isograph.evaluation.runner import prepare_multiplex_suite
from isograph.features.channels import gene_feature_channels
from isograph.io.artifacts import load_dataset_bundle
from isograph.workflow.config import BenchmarkCommandConfig


def test_multiplex_suite_truth_tables_round_trip(tmp_path):
    paths = {path.name: path for path in generate_multiplex_suite(tmp_path / "datasets", seed=7)}
    bundle = load_dataset_bundle(paths["toy_multiplex_v1"])

    assert set(bundle.truth_tables) == {
        "truth_modules.parquet",
        "truth_switch.parquet",
        "truth_abundance.parquet",
        "truth_channel_role.parquet",
    }
    truth_roles = bundle.feature_tables["truth_channel_role"]
    assert set(truth_roles["truth_role"]) == {
        "switch_only",
        "abundance_only",
        "coupled",
        "discordant",
    }
    assert truth_roles["has_switch"].sum() == 30
    assert truth_roles["has_abundance"].sum() == 30


def test_multiplex_suite_contains_single_transcript_abundance_genes(tmp_path):
    paths = {path.name: path for path in generate_multiplex_suite(tmp_path / "datasets", seed=11)}
    bundle = load_dataset_bundle(paths["medium_multiplex_v1"])
    tx_per_gene = bundle.feature_tables["transcript"].groupby("gene_id").size()
    truth_roles = bundle.feature_tables["truth_channel_role"].set_index("gene_id")
    single_gene_ids = set(tx_per_gene[tx_per_gene == 1].index)

    abundance_only_single = [
        gene_id
        for gene_id in single_gene_ids
        if truth_roles.loc[gene_id, "truth_role"] == "abundance_only"
    ]
    assert abundance_only_single

    _, feature_info = gene_feature_channels(
        bundle.matrices["transcript_counts"],
        bundle.feature_tables["transcript"],
        bundle.matrices["gene_counts"],
        bundle.feature_tables["gene"],
    )
    feature_counts = feature_info.groupby(["gene_id", "feature_type"]).size().unstack(fill_value=0)
    for gene_id in abundance_only_single:
        assert feature_counts.loc[gene_id, "abundance"] == 1
        assert feature_counts.loc[gene_id].get("switch", 0) == 0


def test_multiplex_abundance_truth_has_gene_level_signal(tmp_path):
    paths = {path.name: path for path in generate_multiplex_suite(tmp_path / "datasets", seed=13)}
    bundle = load_dataset_bundle(paths["toy_multiplex_v1"])
    truth_roles = bundle.feature_tables["truth_channel_role"].set_index("gene_id")
    gene_table = bundle.feature_tables["gene"].reset_index(drop=True)
    counts = bundle.matrices["gene_counts"]

    module_genes = truth_roles.index[truth_roles["truth_role"] == "abundance_only"].tolist()
    background_genes = truth_roles.index[truth_roles["truth_role"] == "background"].tolist()
    if not background_genes:
        background_genes = truth_roles.index[truth_roles["truth_role"] == "switch_only"].tolist()

    gene_index = {gene_id: idx for idx, gene_id in enumerate(gene_table["gene_id"])}
    module_rows = [gene_index[gene_id] for gene_id in module_genes]
    background_rows = [gene_index[gene_id] for gene_id in background_genes]
    module_corr = np.corrcoef(np.log2(counts[module_rows] + 0.5))
    background_corr = np.corrcoef(np.log2(counts[background_rows] + 0.5))

    assert np.nanmean(np.abs(module_corr[np.triu_indices_from(module_corr, k=1)])) > np.nanmean(
        np.abs(background_corr[np.triu_indices_from(background_corr, k=1)])
    )


def test_prepare_multiplex_suite_respects_fixture_filter(tmp_path):
    config = BenchmarkCommandConfig(
        dataset_suite="multiplex_v1",
        dataset_root=tmp_path / "datasets",
        fixture_filter="noisy_multiplex_v1",
        seed=17,
    )
    paths = prepare_multiplex_suite(config)
    assert [path.name for path in paths] == ["noisy_multiplex_v1"]

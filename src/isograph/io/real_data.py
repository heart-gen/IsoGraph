"""Pure-Python real-data freeze pipeline."""

from __future__ import annotations

import gzip
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
import zlib

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    import pyarrow.dataset as pads
except ImportError:  # pragma: no cover - optional acceleration path
    pa = None
    pc = None
    pacsv = None
    pads = None

from isograph.io.artifacts import (
    DatasetBundle,
    build_feature_spec,
    build_matrix_spec,
    save_dataset_bundle,
)
from isograph.utils import ensure_dir, write_json
from isograph.validation import DatasetManifest, FreezeSelectionConfig
from isograph.workflow.config import RealDataFreezeConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
GENE_CACHE_METADATA_COLUMNS = ["Geneid", "Chr", "Start", "End", "Strand", "Length"]
TRANSCRIPT_CACHE_BUCKETS = 128
TRANSCRIPT_CACHE_SCHEMA_VERSION = 1


def _read_table(path: Path, sep: str = "\t") -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=sep, compression="infer")
    except (OSError, ValueError):
        return pd.read_csv(path, sep=sep, compression=None)


def _log_phase(label: str, started_at: float, payload: str = "") -> None:
    elapsed = perf_counter() - started_at
    suffix = f" | {payload}" if payload else ""
    print(f"[freeze-real] {label}: {elapsed:.2f}s{suffix}", flush=True)


def _select_samples(config: RealDataFreezeConfig) -> pd.DataFrame:
    sample_df = _read_table(config.phenotype_tsv)
    keep = (
        sample_df["Region"].eq("Caudate")
        & sample_df["Race"].eq("AA")
        & sample_df["Age"].gt(17)
        & sample_df["dropped"].eq("f")
        & sample_df["Dx"].isin(config.allowed_diagnoses)
    )
    sample_df = sample_df.loc[keep].copy()
    sample_df = sample_df.sort_values("RNum").reset_index(drop=True)
    sample_df["Age"] = sample_df["Age"].astype(float)
    return sample_df


def _join_ancestry(sample_df: pd.DataFrame, ancestry_path: Path) -> pd.DataFrame:
    if not ancestry_path.exists():
        return sample_df
    ancestry = pd.read_csv(ancestry_path, sep=r"\s+", engine="python")
    if "#FID" in ancestry.columns:
        ancestry = ancestry.rename(columns={"#FID": "BrNum"})
    if "id" in ancestry.columns:
        ancestry = ancestry.rename(columns={"id": "BrNum"})
    if "BrNum" not in ancestry.columns:
        return sample_df
    sample_df = sample_df.merge(ancestry, on="BrNum", how="left")
    return sample_df


def _load_count_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", compression="infer")


def _sample_columns(sample_df: pd.DataFrame, columns: list[str]) -> list[str]:
    rnums = set(sample_df["RNum"])
    return [column for column in columns if column in rnums]


def _standardize_annotation_columns(table: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.lower() for column in table.columns}
    return table.rename(columns=renamed)


def _choose_gene_panel(gene_counts: pd.DataFrame, sample_cols: list[str], panel_size: int) -> pd.Index:
    totals = gene_counts[sample_cols].to_numpy(dtype=float)
    score = totals.mean(axis=1) * (1.0 + totals.var(axis=1))
    top_idx = np.argsort(score)[::-1][:panel_size]
    return gene_counts.iloc[top_idx]["Geneid"]


def _panel_score(gene_counts: pd.DataFrame, sample_cols: list[str]) -> np.ndarray:
    totals = gene_counts[sample_cols].to_numpy(dtype=float)
    return totals.mean(axis=1) * (1.0 + totals.var(axis=1))


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _selection_key(config: RealDataFreezeConfig) -> str:
    diagnoses = "-".join(value.lower() for value in sorted(config.allowed_diagnoses))
    return f"caudate_aa_adult_{diagnoses}"


def _fixture_cache_key(config: RealDataFreezeConfig) -> str:
    return f"{config.output_name}_genes{config.gene_panel_size}"


def _transcript_cache_dir(source_cache_dir: Path) -> Path:
    return source_cache_dir / "transcript_counts"


def _transcript_cache_metadata_path(source_cache_dir: Path) -> Path:
    return source_cache_dir / "transcript_counts.cache.json"


def _gene_bucket(gene_id: str, n_buckets: int = TRANSCRIPT_CACHE_BUCKETS) -> int:
    return zlib.crc32(gene_id.encode("utf-8")) % n_buckets


def _cache_ready(path: Path) -> bool:
    return path.exists()


def _transcript_partitioning() -> object:
    _require_pyarrow_transcript_cache()
    assert pa is not None
    assert pads is not None
    return pads.partitioning(pa.schema([("gene_bucket", pa.int16())]), flavor="hive")


def _copy_tree(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def _load_or_cache_samples(config: RealDataFreezeConfig, source_cache_dir: Path) -> pd.DataFrame:
    sample_cache_path = source_cache_dir / "samples.parquet"
    if sample_cache_path.exists():
        started_at = perf_counter()
        samples = pd.read_parquet(sample_cache_path)
        _log_phase("sample-selection-cache-hit", started_at, f"path={sample_cache_path}, n_samples={len(samples)}")
        return samples

    started_at = perf_counter()
    samples = _join_ancestry(_select_samples(config), Path(config.ancestry_tsv))
    samples.to_parquet(sample_cache_path, index=False)
    write_json(
        source_cache_dir / "samples.cache.json",
        {
            "selection_key": _selection_key(config),
            "phenotype_tsv": str(config.phenotype_tsv),
            "ancestry_tsv": str(config.ancestry_tsv),
            "n_samples": int(len(samples)),
        },
    )
    _log_phase("sample-selection-cache-build", started_at, f"path={sample_cache_path}, n_samples={len(samples)}")
    return samples


def _load_or_cache_gene_projection(
    config: RealDataFreezeConfig,
    samples: pd.DataFrame,
    counts_root: Path,
    source_cache_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    gene_cache_path = source_cache_dir / "gene_projection.parquet"
    sample_cols = list(samples["RNum"])
    if gene_cache_path.exists():
        started_at = perf_counter()
        gene_counts = pd.read_parquet(gene_cache_path)
        sample_cols = _sample_columns(samples, list(gene_counts.columns))
        _log_phase(
            "gene-panel-cache-hit",
            started_at,
            f"path={gene_cache_path}, gene_rows={len(gene_counts)}, n_samples={len(sample_cols)}",
        )
        return gene_counts, sample_cols

    started_at = perf_counter()
    raw_gene_counts = _load_count_table(counts_root / "gene-counts.tsv")
    sample_cols = _sample_columns(samples, list(raw_gene_counts.columns))
    projection_cols = [col for col in GENE_CACHE_METADATA_COLUMNS if col in raw_gene_counts.columns] + sample_cols
    gene_counts = raw_gene_counts[projection_cols].copy()
    gene_counts["panel_score"] = _panel_score(raw_gene_counts, sample_cols)
    for column in sample_cols:
        gene_counts[column] = gene_counts[column].astype(np.float32)
    gene_counts.to_parquet(gene_cache_path, index=False)
    write_json(
        source_cache_dir / "gene_projection.cache.json",
        {
            "selection_key": _selection_key(config),
            "counts_root": str(counts_root),
            "gene_cache_path": str(gene_cache_path),
            "n_genes": int(len(gene_counts)),
            "n_samples": int(len(sample_cols)),
        },
    )
    _log_phase(
        "gene-panel-cache-build",
        started_at,
        f"path={gene_cache_path}, gene_rows={len(gene_counts)}, n_samples={len(sample_cols)}",
    )
    return gene_counts, sample_cols


def _require_pyarrow_transcript_cache() -> None:
    if pa is None or pc is None or pacsv is None or pads is None:
        raise RuntimeError("pyarrow with csv and dataset support is required for transcript caching")


def _load_transcript_annotation_map(annot_root: Path) -> pd.DataFrame:
    tx_gene_map = _read_selected_columns(
        annot_root / "transcript_annotation.tsv.gz",
        ["transcript_id", "gene_id"],
    )
    tx_gene_map = tx_gene_map.dropna(subset=["transcript_id", "gene_id"]).drop_duplicates()
    return tx_gene_map.reset_index(drop=True)


def _load_transcript_cache_metadata(source_cache_dir: Path) -> dict[str, object] | None:
    metadata_path = _transcript_cache_metadata_path(source_cache_dir)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def _build_transcript_cache(
    counts_root: Path,
    annot_root: Path,
    sample_cols: list[str],
    source_cache_dir: Path,
) -> tuple[Path, dict[str, object]]:
    _require_pyarrow_transcript_cache()
    dataset_dir = _transcript_cache_dir(source_cache_dir)
    metadata_path = _transcript_cache_metadata_path(source_cache_dir)
    if dataset_dir.exists() and metadata_path.exists():
        metadata = _load_transcript_cache_metadata(source_cache_dir)
        assert metadata is not None
        return dataset_dir, metadata

    tx_gene_map = _load_transcript_annotation_map(annot_root)
    tx_ids = pa.array(tx_gene_map["transcript_id"].tolist())
    gene_ids = pa.array(tx_gene_map["gene_id"].tolist())
    tx_columns = ["Name", "Length", "EffectiveLength", *sample_cols]
    reader = pacsv.open_csv(
        counts_root / "tx-counts.tsv",
        read_options=pacsv.ReadOptions(use_threads=True, block_size=8 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(include_columns=tx_columns),
    )

    temp_dir = source_cache_dir / "transcript_counts.build"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    ensure_dir(temp_dir)

    total_rows = 0
    started_at = perf_counter()
    for batch_index, batch in enumerate(reader):
        tx_index = pc.index_in(batch.column("Name"), value_set=tx_ids)
        keep_mask = pc.invert(pc.is_null(tx_index))
        if pc.sum(pc.cast(keep_mask, pa.int64())).as_py() == 0:
            continue
        filtered_index = pc.cast(pc.filter(tx_index, keep_mask), pa.int64())
        filtered_gene_ids = pc.take(gene_ids, filtered_index)
        filtered_tx_ids = pc.filter(batch.column("Name"), keep_mask)
        filtered_length = pc.filter(batch.column("Length"), keep_mask)
        filtered_effective_length = pc.filter(batch.column("EffectiveLength"), keep_mask)
        bucket_values = pa.array(
            [_gene_bucket(gene_id) for gene_id in filtered_gene_ids.to_pylist()],
            type=pa.int16(),
        )
        arrays = [
            filtered_gene_ids,
            filtered_tx_ids,
            filtered_length,
            filtered_effective_length,
            bucket_values,
        ]
        names = ["gene_id", "transcript_id", "length", "effective_length", "gene_bucket"]
        for sample_col in sample_cols:
            arrays.append(pc.filter(batch.column(sample_col), keep_mask))
            names.append(sample_col)
        table = pa.Table.from_arrays(arrays, names=names)
        total_rows += table.num_rows
        pads.write_dataset(
            table,
            base_dir=temp_dir,
            format="parquet",
            partitioning=_transcript_partitioning(),
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"part-{batch_index}-{{i}}.parquet",
            max_rows_per_group=4096,
        )

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    temp_dir.replace(dataset_dir)
    metadata = {
        "schema_version": TRANSCRIPT_CACHE_SCHEMA_VERSION,
        "selection_key": source_cache_dir.name,
        "tx_counts_tsv": str(counts_root / "tx-counts.tsv"),
        "annotation_tsv": str(annot_root / "transcript_annotation.tsv.gz"),
        "selected_samples": sample_cols,
        "bucket_count": TRANSCRIPT_CACHE_BUCKETS,
        "row_count": int(total_rows),
        "dataset_dir": str(dataset_dir),
    }
    write_json(metadata_path, metadata)
    _log_phase(
        "transcript-cache-build",
        started_at,
        f"path={dataset_dir}, rows={total_rows}, n_samples={len(sample_cols)}",
    )
    return dataset_dir, metadata


def _load_or_cache_transcript_dataset(
    counts_root: Path,
    annot_root: Path,
    sample_cols: list[str],
    source_cache_dir: Path,
) -> tuple[Path, dict[str, object]]:
    dataset_dir = _transcript_cache_dir(source_cache_dir)
    metadata_path = _transcript_cache_metadata_path(source_cache_dir)
    if _cache_ready(dataset_dir) and metadata_path.exists():
        started_at = perf_counter()
        metadata = _load_transcript_cache_metadata(source_cache_dir)
        assert metadata is not None
        _log_phase(
            "transcript-cache-hit",
            started_at,
            f"path={dataset_dir}, rows={metadata.get('row_count', 'unknown')}",
        )
        return dataset_dir, metadata
    return _build_transcript_cache(counts_root, annot_root, sample_cols, source_cache_dir)


def _read_transcript_cache(
    source_cache_dir: Path,
    sample_cols: list[str],
    selected_genes: set[str],
) -> pd.DataFrame:
    _require_pyarrow_transcript_cache()
    dataset_dir = _transcript_cache_dir(source_cache_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Transcript cache dataset not found: {dataset_dir}")
    gene_buckets = sorted({_gene_bucket(gene_id) for gene_id in selected_genes})
    filter_expression = pads.field("gene_bucket").isin(gene_buckets)
    dataset = pads.dataset(dataset_dir, format="parquet", partitioning=_transcript_partitioning())
    columns = ["gene_id", "transcript_id", "length", "effective_length", *sample_cols]
    table = dataset.to_table(columns=columns, filter=filter_expression)
    if table.num_rows == 0:
        return pd.DataFrame(columns=columns)
    frame = table.to_pandas()
    frame = frame.loc[frame["gene_id"].isin(selected_genes)].reset_index(drop=True)
    return frame


def _select_projection_indices(header: list[str], columns: list[str]) -> list[int]:
    index_map = {column: idx for idx, column in enumerate(header)}
    missing = [column for column in columns if column not in index_map]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return [index_map[column] for column in columns]


def _read_selected_columns(path: Path, columns: list[str], sep: str = "\t") -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=sep, compression="infer", usecols=columns)
    except (OSError, ValueError):
        return pd.read_csv(path, sep=sep, compression=None, usecols=columns)


def _filter_plain_tsv_with_grep(
    counts_path: Path,
    required_columns: list[str],
    selected_ids: set[str],
) -> pd.DataFrame | None:
    if shutil.which("grep") is None or shutil.which("head") is None:
        return None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        for value in sorted(selected_ids):
            handle.write(f"{value}\t\n")
        pattern_path = Path(handle.name)
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    try:
        header_proc = subprocess.run(
            ["head", "-n", "1", str(counts_path)],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        grep_proc = subprocess.run(
            ["grep", "-F", "-f", str(pattern_path), str(counts_path)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    finally:
        pattern_path.unlink(missing_ok=True)
    if grep_proc.returncode not in (0, 1):
        raise subprocess.CalledProcessError(
            grep_proc.returncode,
            grep_proc.args,
            output=grep_proc.stdout,
            stderr=grep_proc.stderr,
        )
    output = header_proc.stdout + grep_proc.stdout
    if not output.strip():
        return pd.DataFrame(columns=required_columns)
    return pd.read_csv(StringIO(output), sep="\t", usecols=required_columns)


def _filter_tsv_with_awk(
    counts_path: Path,
    required_columns: list[str],
    selected_ids: set[str],
    compressed: bool = False,
) -> pd.DataFrame | None:
    if shutil.which("awk") is None:
        return None
    if compressed and shutil.which("gzip") is None:
        return None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        for value in sorted(selected_ids):
            handle.write(f"{value}\n")
        pattern_path = Path(handle.name)
    awk_program = r"NR==FNR{keep[$1]=1; next} FNR==1 || ($1 in keep)"
    try:
        if compressed:
            gzip_proc = subprocess.Popen(
                ["gzip", "-cd", str(counts_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            awk_proc = subprocess.run(
                ["awk", "-F", "\t", awk_program, str(pattern_path), "-"],
                stdin=gzip_proc.stdout,
                capture_output=True,
                text=True,
                check=True,
            )
            assert gzip_proc.stdout is not None
            gzip_proc.stdout.close()
            gzip_proc.wait()
            output = awk_proc.stdout
        else:
            awk_proc = subprocess.run(
                ["awk", "-F", "\t", awk_program, str(pattern_path), str(counts_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            output = awk_proc.stdout
    finally:
        pattern_path.unlink(missing_ok=True)
    if not output.strip():
        return pd.DataFrame(columns=required_columns)
    return pd.read_csv(StringIO(output), sep="\t", usecols=required_columns)


def _read_filtered_transcripts_arrow(
    counts_path: Path,
    required_columns: list[str],
    selected_tx: set[str],
) -> pd.DataFrame | None:
    if pacsv is None or pa is None or pc is None:
        return None
    if not selected_tx:
        return pd.DataFrame(columns=required_columns)
    reader = pacsv.open_csv(
        counts_path,
        read_options=pacsv.ReadOptions(use_threads=True, block_size=8 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(include_columns=required_columns),
    )
    selected_values = pa.array(sorted(selected_tx))
    batches = []
    for batch in reader:
        mask = pc.is_in(batch.column("Name"), value_set=selected_values)
        filtered = batch.filter(mask)
        if filtered.num_rows:
            batches.append(filtered)
    if not batches:
        return pd.DataFrame(columns=required_columns)
    return pa.Table.from_batches(batches).select(required_columns).to_pandas()


def _read_filtered_transcripts(
    counts_path: Path,
    sample_cols: list[str],
    selected_tx: set[str],
) -> pd.DataFrame:
    required_columns = ["Name", "Length", "EffectiveLength", *sample_cols]
    arrow_frame = _read_filtered_transcripts_arrow(counts_path, required_columns, selected_tx)
    if arrow_frame is not None:
        return arrow_frame
    grep_frame = _filter_plain_tsv_with_grep(counts_path, required_columns, selected_tx)
    if grep_frame is not None:
        return grep_frame
    awk_frame = _filter_tsv_with_awk(counts_path, required_columns, selected_tx, compressed=False)
    if awk_frame is not None:
        return awk_frame
    rows: list[list[str]] = []
    with counts_path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        projection = _select_projection_indices(header, required_columns)
        for row_index, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if fields[0] not in selected_tx:
                continue
            rows.append([fields[idx] for idx in projection])
            if row_index % 100_000 == 0:
                print(
                    f"[freeze-real] transcript-scan-progress: rows={row_index}, matches={len(rows)}",
                    flush=True,
                )
    return pd.DataFrame(rows, columns=required_columns)


def _read_filtered_psi(
    counts_path: Path,
    sample_cols: list[str],
    selected_genes: set[str],
) -> pd.DataFrame:
    required_columns = [
        "gene_id",
        "event_type",
        "event_info",
        "chr",
        "strand",
        *sample_cols,
    ]
    awk_frame = _filter_tsv_with_awk(counts_path, required_columns, selected_genes, compressed=True)
    if awk_frame is not None:
        return awk_frame
    rows: list[list[str]] = []
    with gzip.open(counts_path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        projection = _select_projection_indices(header, required_columns)
        for row_index, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if fields[0] not in selected_genes:
                continue
            rows.append([fields[idx] for idx in projection])
            if row_index % 100_000 == 0:
                print(
                    f"[freeze-real] psi-scan-progress: rows={row_index}, matches={len(rows)}",
                    flush=True,
                )
    return pd.DataFrame(rows, columns=required_columns)


def freeze_real_dataset(config: RealDataFreezeConfig, suite_dir: Path) -> Path:
    FreezeSelectionConfig(
        gene_panel_size=config.gene_panel_size, allowed_diagnoses=config.allowed_diagnoses
    )
    suite_dir = ensure_dir(suite_dir)
    output_dir = suite_dir / config.output_name
    counts_root = Path(config.counts_root)
    annot_root = Path(config.annotations_root)
    cache_root = ensure_dir(_resolve_repo_path(config.cache_root))
    source_cache_dir = ensure_dir(cache_root / "sources" / _selection_key(config))
    fixture_cache_dir = cache_root / "fixtures" / _fixture_cache_key(config)

    if (fixture_cache_dir / "manifest.json").exists():
        started_at = perf_counter()
        saved = _copy_tree(fixture_cache_dir, output_dir)
        _log_phase("fixture-cache-hit", started_at, f"source={fixture_cache_dir}, path={saved}")
        return saved

    samples = _load_or_cache_samples(config, source_cache_dir)
    sample_cols = list(samples["RNum"])
    samples = samples.loc[samples["RNum"].isin(sample_cols)].copy().reset_index(drop=True)

    started_at = perf_counter()
    gene_counts, sample_cols = _load_or_cache_gene_projection(config, samples, counts_root, source_cache_dir)
    samples = samples.loc[samples["RNum"].isin(sample_cols)].copy().reset_index(drop=True)
    ranked_genes = gene_counts.sort_values("panel_score", ascending=False, kind="mergesort")
    selected_gene_ids = list(ranked_genes["Geneid"].head(config.gene_panel_size))
    gene_subset = gene_counts.set_index("Geneid").loc[selected_gene_ids].reset_index()
    selected_genes = set(selected_gene_ids)
    gene_matrix = gene_subset[sample_cols].to_numpy(dtype=float)
    _log_phase(
        "gene-panel",
        started_at,
        f"gene_rows={len(gene_counts)}, n_selected_genes={len(selected_genes)}, n_samples={len(sample_cols)}",
    )

    started_at = perf_counter()
    _load_or_cache_transcript_dataset(counts_root, annot_root, sample_cols, source_cache_dir)
    tx_subset = _read_transcript_cache(source_cache_dir, sample_cols, selected_genes)
    selected_tx = set(tx_subset["transcript_id"]) if "transcript_id" in tx_subset.columns else set()
    tx_matrix = tx_subset[sample_cols].to_numpy(dtype=float)
    _log_phase(
        "transcript-extraction",
        started_at,
        f"n_selected_tx={len(selected_tx)}, extracted_rows={len(tx_subset)}",
    )

    started_at = perf_counter()
    psi_counts = _read_filtered_psi(counts_root / "psi-events.tsv.gz", sample_cols, selected_genes)
    psi_annot = _standardize_annotation_columns(
        _read_selected_columns(
            annot_root / "psi_annotation.tsv.gz",
            ["gene_id", "event_type", "event_info", "chr", "strand", "psi_uid"],
        )
    )
    psi_annot = psi_annot.loc[psi_annot["gene_id"].isin(selected_genes)].copy()
    psi_key_cols = ["gene_id", "event_info", "event_type", "chr", "strand"]
    available_key_cols = [col for col in psi_key_cols if col in psi_counts.columns and col in psi_annot.columns]
    psi_subset = psi_counts.merge(psi_annot, on=available_key_cols, how="inner")
    psi_matrix = psi_subset[sample_cols].to_numpy(dtype=float)
    if "psi_uid" not in psi_subset.columns:
        psi_subset["psi_uid"] = [f"psi_{index}" for index in range(len(psi_subset))]
    _log_phase(
        "psi-extraction",
        started_at,
        f"psi_rows={len(psi_counts)}, annotated_rows={len(psi_subset)}",
    )

    gene_feature_table = gene_subset[["Geneid", "Chr", "Start", "End", "Strand", "Length"]].rename(
        columns={"Geneid": "gene_id", "Chr": "chrom", "Start": "start", "End": "end", "Strand": "strand", "Length": "length"}
    )
    tx_feature_cols = ["transcript_id", "gene_id", "length", "effective_length"]
    if "Length" in tx_subset.columns or "EffectiveLength" in tx_subset.columns:
        tx_subset = tx_subset.rename(columns={"Length": "length", "EffectiveLength": "effective_length"})
    tx_feature_table = tx_subset[tx_feature_cols].copy()
    psi_feature_cols = [col for col in ["psi_uid", "gene_id", "event_type", "event_info", "chr", "strand"] if col in psi_subset.columns]
    psi_feature_table = psi_subset[psi_feature_cols].drop_duplicates().reset_index(drop=True)

    manifest = DatasetManifest(
        dataset_name=config.output_name,
        suite_name=suite_dir.name,
        description="Frozen adult AA caudate subset with Control and SCZD samples only. Splicing is represented by PSI only.",
        sample_table="samples.parquet",
        feature_tables=[
            build_feature_spec("gene", "genes.parquet", gene_feature_table),
            build_feature_spec("transcript", "transcripts.parquet", tx_feature_table),
            build_feature_spec("psi", "psi.parquet", psi_feature_table),
        ],
        matrices=[
            build_matrix_spec("gene_counts", "gene_counts.npz", gene_matrix),
            build_matrix_spec("transcript_counts", "transcript_counts.npz", tx_matrix),
            build_matrix_spec("psi", "psi.npz", psi_matrix),
        ],
        provenance={
            "counts_root": str(counts_root),
            "annotations_root": str(annot_root),
            "phenotype_tsv": str(config.phenotype_tsv),
            "selection": "Region=Caudate,Race=AA,Age>17,dropped=f,Dx in {Control,SCZD}",
            "splicing_feature_type": "psi",
            "cache_root": str(cache_root),
            "source_cache_dir": str(source_cache_dir),
            "fixture_cache_dir": str(fixture_cache_dir),
            "transcript_cache_metadata": str(_transcript_cache_metadata_path(source_cache_dir)),
        },
        truth_tables=[],
    )
    bundle = DatasetBundle(
        manifest=manifest,
        sample_table=samples,
        feature_tables={
            "gene": gene_feature_table,
            "transcript": tx_feature_table,
            "psi": psi_feature_table,
        },
        matrices={
            "gene_counts": gene_matrix,
            "transcript_counts": tx_matrix,
            "psi": psi_matrix,
        },
        truth_tables={},
    )
    started_at = perf_counter()
    saved = save_dataset_bundle(bundle, fixture_cache_dir)
    _log_phase(
        "artifact-write",
        started_at,
        f"path={saved}, feature_tables={len(bundle.feature_tables)}, matrices={len(bundle.matrices)}",
    )
    if saved != output_dir:
        started_at = perf_counter()
        saved = _copy_tree(saved, output_dir)
        _log_phase("fixture-cache-materialize", started_at, f"source={fixture_cache_dir}, path={saved}")
    return saved

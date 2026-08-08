"""GTF-based transcript structural comparison (Stage 8C)."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_BOOLEAN_LABELS = [
    "first_exon_changed",
    "last_exon_changed",
    "internal_exon_difference",
    "cds_changed",
    "utr_changed",
    "biotype_switch",
    "coding_status_change",
]
_NUMERIC_LABELS = [
    "transcript_length_delta",
    "cds_length_delta",
    "shared_exon_fraction",
]
STRUCTURAL_LABELS = _BOOLEAN_LABELS + _NUMERIC_LABELS

# Biotype attribute names used by different GTF sources
# GENCODE uses "transcript_type"; Ensembl uses "transcript_biotype"
_BIOTYPE_KEYS = ("transcript_type", "transcript_biotype")


@dataclass
class TranscriptRecord:
    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    biotype: str
    exons: list[tuple[int, int]] = field(default_factory=list)
    cds: list[tuple[int, int]] = field(default_factory=list)


def _extract_attr(attributes: str, key: str) -> str:
    m = re.search(rf'{key} "([^"]+)"', attributes)
    return m.group(1) if m else ""


def _extract_biotype(attributes: str) -> str:
    """Extract transcript biotype, checking GENCODE and Ensembl attribute names."""
    for key in _BIOTYPE_KEYS:
        val = _extract_attr(attributes, key)
        if val:
            return val
    return ""


def _extract_attr_fast(attributes: str, key: str) -> str:
    """Fast attribute extraction using str.find — avoids regex for hot path."""
    needle = f'{key} "'
    idx = attributes.find(needle)
    if idx == -1:
        return ""
    start = idx + len(needle)
    end = attributes.find('"', start)
    return attributes[start:end] if end != -1 else ""


def _build_records(df: pd.DataFrame) -> dict[str, TranscriptRecord]:
    """Build TranscriptRecord dict from a flat DataFrame (feature/start/end rows)."""
    tx_df = df[df["feature"] == "transcript"]
    ex_df = df[df["feature"] == "exon"]
    cds_df = df[df["feature"] == "CDS"]

    records: dict[str, TranscriptRecord] = {}

    for row in tx_df.itertuples(index=False):
        records[row.transcript_id] = TranscriptRecord(
            transcript_id=row.transcript_id,
            gene_id=row.gene_id,
            chrom=str(row.chrom),
            strand=str(row.strand),
            biotype=str(row.biotype),
        )

    for row in ex_df.itertuples(index=False):
        tx_id = row.transcript_id
        if tx_id not in records:
            records[tx_id] = TranscriptRecord(
                transcript_id=tx_id,
                gene_id=row.gene_id,
                chrom=str(row.chrom),
                strand=str(row.strand),
                biotype=str(row.biotype),
            )
        records[tx_id].exons.append((row.start, row.end))

    for row in cds_df.itertuples(index=False):
        tx_id = row.transcript_id
        if tx_id not in records:
            records[tx_id] = TranscriptRecord(
                transcript_id=tx_id,
                gene_id=row.gene_id,
                chrom=str(row.chrom),
                strand=str(row.strand),
                biotype=str(row.biotype),
            )
        records[tx_id].cds.append((row.start, row.end))

    for rec in records.values():
        rec.exons.sort()
        rec.cds.sort()

    return records


def _read_gtf_as_df(path: Path) -> pd.DataFrame:
    """Read GTF file and extract relevant columns via vectorized regex."""
    compression = "gzip" if path.suffix == ".gz" else "infer"
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 2, 3, 4, 6, 8],
        names=["chrom", "feature", "start", "end", "strand", "attrs"],
        dtype={"chrom": "category", "feature": "category", "strand": "category", "attrs": str},
        compression=compression,
        low_memory=False,
    )
    df = df[df["feature"].isin({"transcript", "exon", "CDS"})].copy()
    df["feature"] = df["feature"].astype(str)

    attrs = df["attrs"]
    df["transcript_id"] = attrs.str.extract(r'transcript_id "([^"]+)"', expand=False)
    df["gene_id"] = attrs.str.extract(r'gene_id "([^"]+)"', expand=False)

    # GENCODE: transcript_type; Ensembl: transcript_biotype
    df["biotype"] = attrs.str.extract(r'transcript_type "([^"]+)"', expand=False)
    mask_missing = df["biotype"].isna()
    if mask_missing.any():
        df.loc[mask_missing, "biotype"] = attrs[mask_missing].str.extract(
            r'transcript_biotype "([^"]+)"', expand=False
        )
    df["biotype"] = df["biotype"].fillna("")
    df = df.dropna(subset=["transcript_id"])

    return df[["transcript_id", "gene_id", "chrom", "strand", "biotype", "feature", "start", "end"]]


def parse_gtf(
    path: Path | str,
    cache: Path | str | None = None,
) -> dict[str, TranscriptRecord]:
    """Parse a GTF/GTF.gz file and return transcript_id → TranscriptRecord.

    Uses pandas CSV reader + vectorized string extraction for fast processing.
    Handles both GENCODE (transcript_type) and Ensembl (transcript_biotype)
    attribute naming conventions.

    Args:
        path: Path to GTF or GTF.gz file.
        cache: Optional path for a parquet cache file. If the cache exists and
            is newer than the GTF, it is loaded directly (much faster on repeat
            calls, especially from network filesystems). Written automatically
            after the first full parse.
    """
    path = Path(path)

    if cache is not None:
        cache = Path(cache)
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            df = pd.read_parquet(cache)
            return _build_records(df)

    df = _read_gtf_as_df(path)

    if cache is not None:
        df.to_parquet(cache, index=False)

    return _build_records(df)


def _ordered_exons(tx: TranscriptRecord) -> list[tuple[int, int]]:
    """Return exons in biological (5′→3′) order."""
    if tx.strand == "-":
        return list(reversed(tx.exons))
    return list(tx.exons)


def _jaccard(s1: set, s2: set) -> float:
    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / len(union)


def _interval_length(intervals: list[tuple[int, int]]) -> int:
    return sum(e - s + 1 for s, e in intervals)


def _subtract_intervals(
    exons: list[tuple[int, int]], cds: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return exon regions not covered by CDS (i.e., UTR)."""
    if not cds:
        return list(exons)
    utr: list[tuple[int, int]] = []
    cds_set: set[int] = set()
    for cs, ce in cds:
        cds_set.update(range(cs, ce + 1))
    for es, ee in exons:
        block: list[int] = [p for p in range(es, ee + 1) if p not in cds_set]
        if block:
            # Reconstruct contiguous intervals
            start = block[0]
            prev = block[0]
            for pos in block[1:]:
                if pos != prev + 1:
                    utr.append((start, prev))
                    start = pos
                prev = pos
            utr.append((start, prev))
    return utr


def _utr_changed(tx1: TranscriptRecord, tx2: TranscriptRecord) -> bool:
    """True if UTR regions differ between two coding transcripts."""
    if not tx1.cds or not tx2.cds:
        return False
    utr1 = set(_subtract_intervals(tx1.exons, tx1.cds))
    utr2 = set(_subtract_intervals(tx2.exons, tx2.cds))
    return utr1 != utr2


def compare_pair(tx1: TranscriptRecord, tx2: TranscriptRecord) -> dict[str, bool | int | float]:
    """Compute structural labels for a pair of transcripts from the same gene."""
    oe1 = _ordered_exons(tx1)
    oe2 = _ordered_exons(tx2)

    # Internal exons = all except first and last (empty if ≤1 exon)
    internal1 = set(oe1[1:-1]) if len(oe1) > 2 else set()
    internal2 = set(oe2[1:-1]) if len(oe2) > 2 else set()

    tx_len1 = _interval_length(tx1.exons)
    tx_len2 = _interval_length(tx2.exons)
    cds_len1 = _interval_length(tx1.cds)
    cds_len2 = _interval_length(tx2.cds)

    return {
        "first_exon_changed": oe1[0] != oe2[0],
        "last_exon_changed": oe1[-1] != oe2[-1],
        "internal_exon_difference": internal1 != internal2,
        "cds_changed": set(tx1.cds) != set(tx2.cds),
        "utr_changed": _utr_changed(tx1, tx2),
        "biotype_switch": tx1.biotype != tx2.biotype,
        "coding_status_change": bool(tx1.cds) != bool(tx2.cds),
        "transcript_length_delta": abs(tx_len1 - tx_len2),
        "cds_length_delta": abs(cds_len1 - cds_len2),
        "shared_exon_fraction": _jaccard(set(tx1.exons), set(tx2.exons)),
    }


def annotate_switch_pairs(
    switch_pairs: pd.DataFrame,
    gtf_path: Path | str,
    gtf_cache: Path | str | None = None,
) -> pd.DataFrame:
    """Compute per-transcript structural labels from transcript switch pairs.

    Args:
        switch_pairs: DataFrame with columns gene_id, transcript_id_1, transcript_id_2.
        gtf_path: Path to GTF or GTF.gz file.
        gtf_cache: Optional path for a GTF parse cache (Parquet). Written on first
            run, reloaded on subsequent runs — much faster on network filesystems.

    Returns:
        Per-transcript DataFrame with transcript_id, gene_id, and structural
        label columns. Boolean labels use pd.BooleanDtype(); numerics are float64.
        Transcripts absent from the GTF get pd.NA for all label columns.
    """
    required = {"gene_id", "transcript_id_1", "transcript_id_2"}
    missing = required - set(switch_pairs.columns)
    if missing:
        raise ValueError(f"switch_pairs is missing required columns: {sorted(missing)}")

    tx_db = parse_gtf(gtf_path, cache=gtf_cache)

    # Collect all unique transcripts and their gene_id from switch_pairs
    tx_gene: dict[str, str] = {}
    for _, row in switch_pairs.iterrows():
        tx_gene[row["transcript_id_1"]] = row["gene_id"]
        tx_gene[row["transcript_id_2"]] = row["gene_id"]

    missing_ids = [t for t in tx_gene if t not in tx_db]
    if missing_ids:
        warnings.warn(
            f"{len(missing_ids)} transcript ID(s) not found in GTF and will be skipped: "
            f"{missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}",
            UserWarning,
            stacklevel=2,
        )

    # Build per-transcript aggregated labels
    # For each transcript T: gather comparison results across all its pairs
    tx_comparisons: dict[str, list[dict]] = {t: [] for t in tx_gene}

    for _, row in switch_pairs.iterrows():
        t1 = row["transcript_id_1"]
        t2 = row["transcript_id_2"]
        if t1 not in tx_db or t2 not in tx_db:
            continue
        result = compare_pair(tx_db[t1], tx_db[t2])
        tx_comparisons[t1].append(result)
        tx_comparisons[t2].append(result)

    rows: list[dict] = []
    for tx_id, gene_id in tx_gene.items():
        comps = tx_comparisons[tx_id]
        if not comps:
            # No valid comparisons (e.g., partner missing from GTF)
            row_dict: dict = {"transcript_id": tx_id, "gene_id": gene_id}
            for col in _BOOLEAN_LABELS:
                row_dict[col] = pd.NA
            for col in _NUMERIC_LABELS:
                row_dict[col] = float("nan")
            rows.append(row_dict)
            continue

        row_dict = {"transcript_id": tx_id, "gene_id": gene_id}
        for col in _BOOLEAN_LABELS:
            vals = [c[col] for c in comps if c[col] is not None]
            row_dict[col] = any(vals) if vals else pd.NA
        row_dict["transcript_length_delta"] = float(
            max(c["transcript_length_delta"] for c in comps)
        )
        row_dict["cds_length_delta"] = float(max(c["cds_length_delta"] for c in comps))
        row_dict["shared_exon_fraction"] = float(min(c["shared_exon_fraction"] for c in comps))
        rows.append(row_dict)

    if not rows:
        return pd.DataFrame(columns=["transcript_id", "gene_id"] + STRUCTURAL_LABELS)

    result_df = pd.DataFrame(rows)
    for col in _BOOLEAN_LABELS:
        result_df[col] = pd.array(result_df[col].tolist(), dtype=pd.BooleanDtype())
    for col in _NUMERIC_LABELS:
        result_df[col] = result_df[col].astype("float64")

    return result_df[["transcript_id", "gene_id"] + STRUCTURAL_LABELS].reset_index(drop=True)

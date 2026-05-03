# Own-Data Data Model

IsoGraph does not ingest arbitrary raw count folders directly. For custom analyses, first
package your data as an IsoGraph dataset bundle with aligned metadata tables, feature
tables, and dense matrices.

## Minimum Practical Bundle

For custom-data work, build a bundle containing at least:

- a `manifest.json`
- a sample table aligned to the matrix columns
- a gene feature table
- a transcript feature table with `transcript_id` and `gene_id`
- a `transcript_counts` dense matrix

For parity with the bundled fixtures and export tooling, include `gene_counts` and a gene
table as well. Add `psi` feature tables and matrices when you have splicing event data
you want to preserve in the bundle.

## Required Alignment Rules

- Rows of each feature table must align with the rows of its matching matrix.
- Rows of the sample table must align with the columns of every matrix.
- The transcript table must include `transcript_id` and `gene_id`.
- A gene table should include `gene_id`.
- Covariates or traits referenced by model configs, such as `Age`, `Dx`, `RIN`, `PMI`,
  `mito_mapping_rate`, and `percent_assigned`, must exist in the sample table if you want
  them used. Missing columns are skipped rather than inferred.

## Building a Bundle

```python
from pathlib import Path

import numpy as np
import pandas as pd

from isograph.io.artifacts import (
    DatasetBundle,
    build_feature_spec,
    build_matrix_spec,
    save_dataset_bundle,
)
from isograph.validation import DatasetManifest

sample_table = pd.DataFrame(
    {
        "sample_id": ["S1", "S2", "S3"],
        "Age": [64.0, 59.0, 71.0],
        "Dx": ["Control", "SCZD", "Control"],
    }
)

gene_table = pd.DataFrame({"gene_id": ["G1", "G2"]})
transcript_table = pd.DataFrame(
    {
        "transcript_id": ["T1", "T2", "T3", "T4"],
        "gene_id": ["G1", "G1", "G2", "G2"],
    }
)

gene_counts = np.array(
    [
        [120.0, 80.0, 95.0],
        [60.0, 110.0, 90.0],
    ]
)
transcript_counts = np.array(
    [
        [70.0, 50.0, 60.0],
        [50.0, 30.0, 35.0],
        [20.0, 55.0, 40.0],
        [40.0, 55.0, 50.0],
    ]
)

manifest = DatasetManifest(
    dataset_name="my_cohort_v1",
    suite_name="custom",
    description="Example custom cohort packaged for IsoGraph",
    sample_table="samples.parquet",
    feature_tables=[
        build_feature_spec("gene", "genes.parquet", gene_table),
        build_feature_spec("transcript", "transcripts.parquet", transcript_table),
    ],
    matrices=[
        build_matrix_spec("gene_counts", "gene_counts.npz", gene_counts),
        build_matrix_spec("transcript_counts", "transcript_counts.npz", transcript_counts),
    ],
    provenance={"source": "custom cohort"},
)

bundle = DatasetBundle(
    manifest=manifest,
    sample_table=sample_table,
    feature_tables={
        "gene": gene_table,
        "transcript": transcript_table,
    },
    matrices={
        "gene_counts": gene_counts,
        "transcript_counts": transcript_counts,
    },
    truth_tables={},
)

save_dataset_bundle(bundle, Path("benchmarks/datasets/custom/my_cohort_v1"))
```

## Running the VAE Backend from the CLI

VAE is the default backend for `isograph fit`:

```bash
isograph fit \
  --dataset-path benchmarks/datasets/custom/my_cohort_v1 \
  --output-dir artifacts/fits/my_cohort_v1
```

To use a different backend, pass `--backend <name>`:

```bash
isograph fit \
  --dataset-path benchmarks/datasets/custom/my_cohort_v1 \
  --backend baseline \
  --output-dir artifacts/fits/my_cohort_v1_baseline
```

## Running Backends from Python

```python
from pathlib import Path

from isograph.io.artifacts import load_dataset_bundle
from isograph.models.latent import LatentNetworkModel
from isograph.workflow.config import LatentModelConfig

bundle = load_dataset_bundle(Path("benchmarks/datasets/custom/my_cohort_v1"))

model = LatentNetworkModel(LatentModelConfig(alpha=0.05, n_components=5))
artifacts = model.fit(
    transcript_counts=bundle.matrices["transcript_counts"],
    transcript_table=bundle.feature_tables["transcript"],
    sample_table=bundle.sample_table,
)

print(artifacts.module_table.head())
print(artifacts.edge_table.head())
```

The same pattern applies to `GraphNetworkModel` and, with PyTorch installed,
`VaeNetworkModel`.

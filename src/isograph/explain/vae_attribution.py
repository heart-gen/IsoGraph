"""VAE decoder attribution (Stage 8D — stub).

Only used when a VAE checkpoint is available in artifact_dir.
"""

from __future__ import annotations


def compute_decoder_jacobian(checkpoint_path, eigengene, feature_ids):
    """Compute Jacobian of VAE decoder w.r.t. latent eigengene direction.

    Returns DataFrame(feature_id, jacobian_weight).
    """
    raise NotImplementedError("Stage 8D")

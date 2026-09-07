"""Tests for multimodal projected feature fusion and scale balancing."""

import numpy as np

from quant.services.event_service import compute_projected_feature_vector


def test_projected_feature_vector_dimension() -> None:
    # 768-dimensional mock FinBERT dense embedding
    dense_768 = [0.05] * 768
    sentiment = [0.5, 0.2, 0.1]
    urgency = 0.8
    centrality = 0.95
    target_dim = 16

    fused = compute_projected_feature_vector(
        dense_embedding=dense_768,
        sentiment_vector=sentiment,
        urgency=urgency,
        centrality=centrality,
        target_dense_dim=target_dim,
    )

    # 16 (projected dense) + 3 (sentiment) + 1 (urgency) + 1 (centrality) = 21 dimensions
    assert isinstance(fused, np.ndarray)
    assert fused.shape == (21,)

    # Verify dense portion is normalized
    dense_part = fused[:target_dim]
    norm = np.linalg.norm(dense_part)
    assert np.isclose(norm, 1.0, atol=1e-5)

    # Verify scalar portion is preserved exactly
    np.testing.assert_allclose(fused[target_dim:], [0.5, 0.2, 0.1, 0.8, 0.95])


def test_projected_feature_vector_empty_embedding() -> None:
    sentiment = [-0.2, 0.4, 0.0]
    urgency = 0.3
    centrality = 0.7
    target_dim = 16

    fused = compute_projected_feature_vector(
        dense_embedding=[],
        sentiment_vector=sentiment,
        urgency=urgency,
        centrality=centrality,
        target_dense_dim=target_dim,
    )

    assert fused.shape == (21,)
    # Dense portion should be zeros
    assert np.all(fused[:target_dim] == 0.0)
    np.testing.assert_allclose(fused[target_dim:], [-0.2, 0.4, 0.0, 0.3, 0.7])

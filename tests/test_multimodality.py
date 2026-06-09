from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pg3d.multimodality import (
    _plotly_html,
    _trajectory_pca3_scores,
    first_sequence_index_for_episode,
    multimodality_stats,
    pairwise_rms_distances,
    select_diverse_action_chunks,
)


def test_pairwise_rms_distances() -> None:
    flattened = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )

    distances = pairwise_rms_distances(flattened)

    np.testing.assert_allclose(distances, [1.0, np.sqrt(2.0), 1.0])


def test_multimodality_stats_reports_pairwise_and_final_spread() -> None:
    chunks = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )

    stats = multimodality_stats(chunks)

    assert stats["mean_pairwise_rms"] > 0.0
    assert stats["max_pairwise_rms"] > stats["mean_pairwise_rms"]
    assert stats["final_action_l2_std"] > 0.0
    assert len(stats["final_action_dim_std"]) == 2


def test_trajectory_pca3_scores_preserves_sample_and_demo_shapes() -> None:
    chunks = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    demo = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)

    sample_scores, demo_scores = _trajectory_pca3_scores(
        action_chunks=chunks,
        demo_action=demo,
        n_obs_steps=2,
    )

    assert sample_scores.shape == (3, 4, 3)
    assert demo_scores is not None
    assert demo_scores.shape == (4, 3)


def test_plotly_html_contains_trace_payload() -> None:
    html = _plotly_html(
        [{"type": "scatter3d", "x": [0.0], "y": [1.0], "z": [2.0]}],
        {"title": "demo"},
    )

    assert "cdn.plot.ly" in html
    assert "Plotly.newPlot" in html
    assert "scatter3d" in html


def test_select_diverse_action_chunks_picks_spread_subset() -> None:
    chunks = np.asarray(
        [
            [[0.0], [0.0]],
            [[0.1], [0.1]],
            [[5.0], [5.0]],
            [[10.0], [10.0]],
            [[10.1], [10.1]],
        ],
        dtype=np.float32,
    )

    selected = select_diverse_action_chunks(chunks, count=3)

    assert selected.shape == (3,)
    assert set(selected.tolist()) == {0, 2, 4}


def test_first_sequence_index_for_episode() -> None:
    dataset = SimpleNamespace(
        num_episodes=3,
        episode_ends=np.asarray([5, 10, 15]),
        indices=np.asarray(
            [
                [0, 4, 1, 5],
                [5, 9, 1, 5],
                [6, 10, 0, 4],
                [10, 14, 1, 5],
            ],
            dtype=np.int64,
        ),
    )

    assert first_sequence_index_for_episode(dataset, 0) == 0
    assert first_sequence_index_for_episode(dataset, 1) == 1
    assert first_sequence_index_for_episode(dataset, 2) == 3


def test_first_sequence_index_for_episode_rejects_out_of_range() -> None:
    dataset = SimpleNamespace(
        num_episodes=1,
        episode_ends=np.asarray([5]),
        indices=np.asarray([[0, 4, 1, 5]], dtype=np.int64),
    )

    with pytest.raises(IndexError, match="out of range"):
        first_sequence_index_for_episode(dataset, 1)

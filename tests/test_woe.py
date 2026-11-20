"""Tests for src/woe.py — WoE encoding and Information Value."""
import numpy as np
import pandas as pd

from src.woe import WoeEncoder, iv_strength


def test_woe_iv_is_non_negative_and_detects_signal():
    # Strong signal: low x -> almost never default, high x -> often default.
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    y = (x + rng.normal(0, 0.5, 2000) > 1.0).astype(int)
    enc = WoeEncoder(n_bins=5).fit(pd.DataFrame({"x": x}), pd.Series(y))
    assert enc.iv_["x"] > 0.1, f"expected informative IV, got {enc.iv_['x']:.3f}"
    # WoE must be monotone decreasing in x here (higher x = more bad = lower WoE,
    # since WoE = log(good%/bad%)).
    woe_values = list(enc.woe_maps_["x"].values())
    assert woe_values[0] > woe_values[-1]


def test_woe_transform_shape_and_unseen_category():
    train = pd.DataFrame({
        "grade": ["A", "A", "B", "B", "C", "C"] * 100,
        "dti": np.tile([10.0, 20.0, 30.0], 200),
    })
    y = pd.Series([0, 0, 0, 1, 1, 1] * 100)
    enc = WoeEncoder().fit(train, y)
    test = pd.DataFrame({"grade": ["A", "Z"], "dti": [10.0, 99.0]})
    out = enc.transform(test)
    assert out.shape == (2, 2)
    assert out.columns.tolist() == ["grade_woe", "dti_woe"]
    # Unseen category "Z" -> neutral WoE 0, not NaN.
    assert out["grade_woe"].iloc[1] == 0.0
    assert np.isfinite(out.values).all()


def test_iv_strength_labels():
    assert iv_strength(0.01) == "unpredictive"
    assert iv_strength(0.05) == "weak"
    assert iv_strength(0.2) == "medium"
    assert iv_strength(0.4) == "strong"

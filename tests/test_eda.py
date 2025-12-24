import numpy as np

from src.phase9_runner import fit_eda


def test_fit_eda_shapes_and_finite():
    rng = np.random.default_rng(0)

    # 3 lớp, 2D, tạo cụm tách nhau
    n = 50
    X0 = rng.normal(loc=(0.0, 0.0), scale=1.0, size=(n, 2))
    X1 = rng.normal(loc=(5.0, 0.0), scale=1.0, size=(n, 2))
    X2 = rng.normal(loc=(0.0, 5.0), scale=1.0, size=(n, 2))
    X = np.vstack([X0, X1, X2]).astype(np.float32)
    y = np.array([0]*n + [1]*n + [2]*n)

    W = fit_eda(X, y, n_components=2, reg_eps=1e-6)

    assert W.shape == (2, 2)
    assert np.isfinite(W).all()

    Z = X @ W
    assert Z.shape == (3*n, 2)
    assert np.isfinite(Z).all()

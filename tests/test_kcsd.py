"""Tests for the kCSD port (Vectorization, chunking, reconstruction quality,
optional cross-validation) against a reference Elephant-style loop
implementation of the estimator."""

import numpy as np
import pytest

from invivosuite.csd import KCSD1D, KCSD2D, generate_lfp
from invivosuite.csd.kcsd import KCSD
from invivosuite.csd.utils import gauss_1d_dipole, small_source_2d

RNG = np.random.default_rng(7)


def _reference_values(k, estimate="CSD"):
    """Elephant's nested-loop implementation of ``values`` (bit of history:
    loops over every time point and every electrode)."""
    table = k._estimation_table(estimate)
    k_inv = np.linalg.inv(k.k_pot + k.lambd * np.identity(k.n_ele))
    estimation = np.zeros((k.n_estm, k.pots.shape[1]))
    for t in range(k.pots.shape[1]):
        beta = np.dot(k_inv, k.pots[:, t])
        for i in range(k.n_ele):
            estimation[:, t] += table[:, i] * beta[i]
    return estimation


# ---------------------------------------------------------------------------
# 1D: linear algebra / vectorization properties
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def k1d():
    ele_pos = np.linspace(0.05, 0.95, 16)
    pots = RNG.normal(size=(16, 30))
    return KCSD1D(
        ele_pos, pots, gdx=0.02, n_src_init=300, R_init=0.1, lambd=1e-8
    )


def test_vectorized_matches_reference_loops(k1d):
    fast = k1d.estimate()
    slow = _reference_values(k1d)
    np.testing.assert_allclose(fast, slow, rtol=1e-6, atol=1e-9)


def test_values_reshape(k1d):
    v = k1d.values()
    assert v.shape == (k1d.ngx, k1d.pots.shape[1])
    np.testing.assert_allclose(v.ravel(), k1d.estimate().ravel())


def test_chunked_equals_full(k1d):
    full = k1d.estimate()
    chunked = k1d.estimate(chunk_size=7)
    np.testing.assert_allclose(full, chunked, rtol=1e-12, atol=1e-12)


def test_linearity(k1d):
    p1 = RNG.normal(size=(k1d.n_ele, 10))
    p2 = RNG.normal(size=(k1d.n_ele, 10))
    combo = k1d.estimate(3.0 * p1 - 0.5 * p2)
    expected = 3.0 * k1d.estimate(p1) - 0.5 * k1d.estimate(p2)
    np.testing.assert_allclose(combo, expected, rtol=1e-9, atol=1e-12)


def test_zero_pots_give_zero_csd(k1d):
    zeros = k1d.estimate(np.zeros((k1d.n_ele, 5)))
    np.testing.assert_array_equal(zeros, 0.0)


def test_apply_at_nearest_rows(k1d):
    pots = RNG.normal(size=(k1d.n_ele, 4))
    full = k1d.estimate(pots)
    rows, estm_pts = k1d.nearest_rows(k1d.ele_pos)
    subset = k1d.apply(pots, rows=rows)
    np.testing.assert_allclose(subset, full[rows], rtol=1e-12, atol=1e-15)
    # each returned grid point should be within one grid step of an electrode
    ele_sorted = np.sort(k1d.ele_pos.ravel())
    assert np.max(np.abs(estm_pts.ravel() - ele_sorted)) < 2 * k1d.gdx


def test_cross_validate_segment(k1d):
    lambdas = np.array([0.0, 1e-6, 1e-3, 1e-1])
    cv_R, cv_lambda = k1d.cross_validate(lambdas=lambdas, segment=(0, 10))
    assert cv_R == k1d.R
    assert cv_lambda in lambdas
    assert k1d.cv_error is not None
    # operator reflects the selected lambda
    np.testing.assert_allclose(
        k1d.apply(k1d.pots), _reference_values(k1d), rtol=1e-9, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Ground-truth reconstruction
# ---------------------------------------------------------------------------

def test_1d_dipole_reconstruction():
    ele_pos = np.linspace(0.05, 0.95, 20)
    pots = generate_lfp(gauss_1d_dipole, ele_pos, resolution=50)
    pots = pots[:, np.newaxis]
    k = KCSD1D(ele_pos, pots, gdx=0.02, n_src_init=300, R_init=0.1, lambd=1e-6)
    est = k.estimate()[:, 0]
    true_csd = gauss_1d_dipole(k.estm_x)
    corr = np.corrcoef(est, true_csd)[0, 1]
    assert corr > 0.75, corr


def test_1d_narrow_dipole_localization():
    # A localized (narrow-gaussian) dipole: source at 0.65, sink at 0.35.
    def narrow_dipole(x):
        return 0.5 * np.exp(-((x - 0.65) ** 2) / (2 * 0.03**2)) - 0.5 * np.exp(
            -((x - 0.35) ** 2) / (2 * 0.03**2)
        )

    ele_pos = np.linspace(0.05, 0.95, 30)
    pots = generate_lfp(narrow_dipole, ele_pos, resolution=100)
    pots = pots[:, np.newaxis]
    k = KCSD1D(ele_pos, pots, gdx=0.01, n_src_init=500, R_init=0.05, lambd=1e-9)
    est = k.estimate()[:, 0]
    assert abs(k.estm_x[np.argmin(est)] - 0.35) < 0.12
    assert abs(k.estm_x[np.argmax(est)] - 0.65) < 0.12
    corr = np.corrcoef(est, narrow_dipole(k.estm_x))[0, 1]
    assert corr > 0.6, corr


def test_2d_small_source_reconstruction():
    grid = np.linspace(0.1, 0.8, 6)
    xg, yg = np.meshgrid(grid, grid, indexing="ij")
    ele_pos = np.column_stack((xg.ravel(), yg.ravel()))
    pots = generate_lfp(
        small_source_2d, ele_pos[:, 0], ele_pos[:, 1], resolution=50, h=1.0
    )
    pots = pots[:, np.newaxis]
    k = KCSD2D(
        ele_pos, pots, gdx=0.02, gdy=0.02, n_src_init=1000, R_init=0.08, lambd=1e-6
    )
    est = k.estimate()[:, 0].reshape(k.ngx, k.ngy)
    true_csd = small_source_2d(k.estm_x, k.estm_y)
    corr = np.corrcoef(est.ravel(), true_csd.ravel())[0, 1]
    assert corr > 0.6, corr


def test_kcsd2d_rejects_degenerate_geometry():
    ele_pos = np.column_stack((np.linspace(0, 0.5, 9), np.zeros(9)))
    with pytest.raises(ValueError, match="KCSD1D"):
        KCSD2D(ele_pos, RNG.normal(size=(9, 2)))


def test_invalid_kwarg_raises():
    ele_pos = np.linspace(0, 1, 8)
    with pytest.raises(TypeError):
        KCSD1D(ele_pos, RNG.normal(size=(8, 2)), bogus_param=1)

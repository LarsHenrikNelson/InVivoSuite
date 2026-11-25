"""Kernel Current Source Density (kCSD) estimation.

Port of the kCSD implementation from the Elephant package
(`elephant/current_source_density_src/KCSD.py`, BSD-3-Clause), originally by
Chaitanya Chintaluri and Michal Czerwinski (Nencki Institute of Experimental
Biology, Warsaw), implementing Potworowski, Jakuczun, Blonski, Wójcik (2012)
"Kernel current source density method", Neuroinformatics 10:151.

Deviations from the Elephant version
------------------------------------
1. **Plain NumPy**: no `neo`/`quantities` unit arrays anywhere. Potentials are
   ``(n_ele, n_time)`` arrays, electrode positions are ``(n_ele, dim)`` arrays
   in any self-consistent length unit (e.g. um).
2. **Vectorized estimation**: Elephant evaluates the estimate with nested
   Python loops over every time point and every electrode. Here the operator
   ``M = K_interp_cross @ inv(K_pot + lambd * I)`` is precomputed once per
   (geometry, R, lambda) and applied as a single BLAS matrix multiplication
   (optionally streamed in time chunks), which is orders of magnitude faster
   for long recordings (minutes of LFP at >= 1 kHz).
3. **Cross-validation is optional**: Elephant's ``cross_validate`` grid-searches
   lambdas/Rs with a leave-one-out over electrodes, recomputing the estimate
   over all time samples each time. Here it is not run at init, the error is
   computed with vectorized folds, and a time ``segment`` can be supplied so
   parameters are selected once from a short stretch of data instead of the
   full recording.
"""

import numpy as np
from numpy.linalg import LinAlgError
from scipy import interpolate
from scipy.spatial import distance

from . import basis, utils

__all__ = ["KCSD"]


class KCSD:
    """KCSD - The base class for all the KCSD variants.

    Estimates the Current Source Density for a given configuration of
    electrode positions and recorded potentials, based on Potworowski
    et al. 2012. Use `KCSD1D` (laminar probes) or `KCSD2D` (planar arrays);
    this base class contains the shared parameter handling, kernel
    construction, vectorized estimation, and optional cross-validation.

    Parameters
    ----------
    ele_pos : np.ndarray, (n_ele, dim)
        Electrode positions in any consistent length unit.
    pots : np.ndarray, (n_ele, n_time)
        Measured potentials; time along axis 1.
    src_type : {'gauss', 'step', 'gauss_lim'}
        Basis function type. Default: 'gauss'.
    sigma : float
        Conductivity of the tissue, consistent with the ele_pos length unit.
    h : float
        Thickness of the analyzed tissue slice. Default: 1.0.
    n_src_init : int
        Requested number of basis sources. Default: 1000.
    lambd : float
        Tikhonov regularization parameter. Default: 0.0.
    R_init : float
        Initial basis-function radius. Default: 0.23.

    Attributes
    ----------
    operator : np.ndarray, (n_estm, n_ele)
        Precomputed estimate operator M; ``csd = operator @ pots``.
    cv_error : float | None
        Leave-one-out error of the selected (R, lambda), or None if
        `cross_validate` has not been run.
    """

    def __init__(self, ele_pos, pots, **kwargs):
        self.validate(ele_pos, pots)
        self.ele_pos = np.asarray(ele_pos, dtype=float)
        self.pots = np.asarray(pots, dtype=float)
        self.n_ele = self.ele_pos.shape[0]
        self.n_time = self.pots.shape[1]
        self.dim = self.ele_pos.shape[1]
        self.cv_error = None
        self.parameters(**kwargs)
        self.estimate_at()
        self.place_basis()
        self.create_src_dist_tables()
        self.method()
        self.update_operator(estimate="CSD")

    def validate(self, ele_pos, pots):
        """Basic checks to see if inputs are okay."""
        if ele_pos.shape[0] != pots.shape[0]:
            raise ValueError(
                "Number of measured potentials is not equal to electrode number!"
            )
        if ele_pos.shape[0] < 1 + ele_pos.shape[1]:  # Dim+1
            raise ValueError(
                f"Number of electrodes must be at least {1 + ele_pos.shape[1]}"
            )
        if utils.contains_duplicated_electrodes(ele_pos):
            raise ValueError("Error! Duplicated electrode!")

    def parameters(self, **kwargs):
        """Define the default parameter values; unknown kwargs are an error.

        Raises
        ------
        TypeError
            If invalid keyword arguments were passed.
        """
        self.src_type = kwargs.pop("src_type", "gauss")
        self.sigma = kwargs.pop("sigma", 1.0)
        self.h = kwargs.pop("h", 1.0)
        self.n_src_init = kwargs.pop("n_src_init", 1000)
        self.lambd = kwargs.pop("lambd", 0.0)
        self.R_init = kwargs.pop("R_init", 0.23)
        self.ext_x = kwargs.pop("ext_x", 0.0)
        self.xmin = kwargs.pop("xmin", np.min(self.ele_pos[:, 0]))
        self.xmax = kwargs.pop("xmax", np.max(self.ele_pos[:, 0]))
        self.gdx = kwargs.pop("gdx", 0.01 * (self.xmax - self.xmin))
        if self.dim >= 2:
            self.ext_y = kwargs.pop("ext_y", 0.0)
            self.ymin = kwargs.pop("ymin", np.min(self.ele_pos[:, 1]))
            self.ymax = kwargs.pop("ymax", np.max(self.ele_pos[:, 1]))
            self.gdy = kwargs.pop("gdy", 0.01 * (self.ymax - self.ymin))
        if kwargs:
            raise TypeError("Invalid keyword arguments:", kwargs.keys())

    def method(self):
        """Sequence of kernel-construction methods.

        Defines ``self.k_pot``, ``self.k_interp_cross`` and
        ``self.k_interp_pot``.
        """
        self.create_lookup()
        self.update_b_pot()
        self.update_b_src()
        self.update_b_interp_pot()

    def create_lookup(self, dist_table_density=20):
        """Lookup table of the potential generated by a basis source at
        tabulated distances (one-off numerical integrations, not per time
        sample)."""
        xs = np.logspace(0.0, np.log10(self.dist_max + 1.0), dist_table_density)
        xs = xs - 1.0  # starting from 0
        dist_table = np.zeros(len(xs))
        for i, pos in enumerate(xs):
            dist_table[i] = self.forward_model(
                pos, self.R, self.h, self.sigma, self.basis
            )
        self.interpolate_pot_at = interpolate.interp1d(xs, dist_table, kind="cubic")

    def update_b_pot(self):
        """Potential basis matrix ``b_pot`` (n_src, n_ele) and potential kernel
        ``k_pot`` (n_ele, n_ele); Eq. 9, Jan 2012."""
        self.b_pot = self.interpolate_pot_at(self.src_ele_dists)
        self.k_pot = np.dot(self.b_pot.T, self.b_pot)
        self.k_pot /= self.n_src

    def update_b_src(self):
        """Source basis matrix and cross kernel ``k_interp_cross``
        (n_estm, n_ele); K_t(x,y), Eq. 17."""
        self.b_src = self.basis(self.src_estm_dists, self.R).T
        self.k_interp_cross = np.dot(self.b_src, self.b_pot)
        self.k_interp_cross /= self.n_src

    def update_b_interp_pot(self):
        """Potential-interpolation kernel ``k_interp_pot`` (n_estm, n_ele)."""
        self.b_interp_pot = self.interpolate_pot_at(self.src_estm_dists).T
        self.k_interp_pot = np.dot(self.b_interp_pot, self.b_pot)
        self.k_interp_pot /= self.n_src

    # ------------------------------------------------------------------
    # Vectorized estimation (replaces Elephant's nested for loops)
    # ------------------------------------------------------------------

    def _estimation_table(self, estimate="CSD"):
        if estimate == "CSD":
            return self.k_interp_cross
        if estimate == "POT":
            return self.k_interp_pot
        raise ValueError("estimate must be 'CSD' or 'POT'")

    def update_operator(self, lambd=None, estimate="CSD"):
        """Precompute the estimation operator
        ``M = table @ inv(K_pot + lambd * I)`` (n_estm, n_ele).

        After this, estimating the CSD for any pots matrix is a single
        matrix multiplication (`apply`), so long recordings can be streamed
        through the operator in time chunks. Elephant instead loops over
        every time point and electrode per call.
        """
        if lambd is not None:
            self.lambd = lambd
        k_reg = self.k_pot + self.lambd * np.identity(self.n_ele)
        try:
            k_inv = np.linalg.solve(k_reg, np.identity(self.n_ele))
        except LinAlgError as exc:
            raise LinAlgError(
                "Encountered singular matrix: try a nonzero lambd "
                "or changing ele_pos slightly"
            ) from exc
        self.operator = np.dot(self._estimation_table(estimate), k_inv)
        self._operator_estimate = estimate
        return self.operator

    def apply(self, pots, rows=None):
        """Apply the precomputed operator to a pots matrix.

        Parameters
        ----------
        pots : np.ndarray, (n_ele, n_time) or (n_ele,)
            Potentials; a 1D array is treated as a single time column.
        rows : np.ndarray | None
            Optional indices into the flattened estimation grid; supplying
            them computes the estimate only for those rows (e.g. the grid
            points nearest to electrodes) for cheap per-channel traces.

        Returns
        -------
        np.ndarray, (n_rows or n_estm, n_time)
        """
        pots = np.asarray(pots, dtype=float)
        if pots.ndim == 1:
            pots = pots[:, np.newaxis]
        elif pots.shape[0] != self.n_ele:
            raise ValueError(f"pots must have {self.n_ele} rows (electrodes)")
        operator = self.operator if rows is None else self.operator[rows]
        return np.dot(operator, pots)

    def nearest_rows(self, points):
        """Indices of the estimation-grid locations nearest to `points`.

        Parameters
        ----------
        points : np.ndarray, (n_points, dim)
            Coordinates, same units as ``ele_pos``.

        Returns
        -------
        rows : np.ndarray, (n_points,)
            Row indices into the flattened estimation grid.
        estm_points : np.ndarray, (n_points, dim)
            The selected estimation-grid coordinates.
        """
        estm_loc = self.estm_points()
        dists = distance.cdist(np.atleast_2d(points), estm_loc, "euclidean")
        rows = np.argmin(dists, axis=1)
        return rows, estm_loc[rows]

    def apply_at(self, points, pots):
        """Estimate the CSD at the estimation-grid locations nearest to
        `points` (e.g. electrode positions)."""
        rows, _ = self.nearest_rows(points)
        return self.apply(pots, rows=rows)

    def estimate(self, pots=None, estimate="CSD", chunk_size=None):
        """Estimate CSD (or potentials) from pots: the fast, flat-array API.

        Parameters
        ----------
        pots : np.ndarray | None, (n_ele, n_time)
            Potentials at the electrodes. Defaults to the pots supplied at
            construction time.
        estimate : {'CSD', 'POT'}
            Which quantity to reconstruct.
        chunk_size : int | None
            Number of time samples processed per matrix multiplication.
            Bounds peak memory for long recordings; results are identical
            to the single-shot computation.

        Returns
        -------
        np.ndarray, (n_estm, n_time)
        """
        if (
            not hasattr(self, "operator")
            or getattr(self, "_operator_estimate", None) != estimate
        ):
            self.update_operator(estimate=estimate)
        if pots is None:
            pots = self.pots
        pots = np.asarray(pots, dtype=float)
        single_shot = chunk_size is None or pots.shape[1] <= chunk_size
        if single_shot:
            return self.apply(pots)
        out = np.empty((self.operator.shape[0], pots.shape[1]))
        for t0 in range(0, pots.shape[1], chunk_size):
            t1 = min(t0 + chunk_size, pots.shape[1])
            out[:, t0:t1] = self.apply(pots[:, t0:t1])
        return out

    def values(self, estimate="CSD", pots=None, chunk_size=None):
        """Elephant-compatible estimator returning the reshaped estimate
        ((ngx,) (ngy,) n_time); see `estimate` for the flat-array version."""
        estimation = self.estimate(pots=pots, estimate=estimate, chunk_size=chunk_size)
        return self.process_estimate(estimation)

    def process_estimate(self, estimation):
        """Rearrange the flat (n_estm, n_time) estimation by dimension."""
        if self.dim == 1:
            return estimation.reshape(self.ngx, -1)
        return estimation.reshape(self.ngx, self.ngy, -1)

    # ------------------------------------------------------------------
    # Optional, vectorized cross-validation (NOT run at construction time)
    # ------------------------------------------------------------------

    def update_R(self, R):
        """Update the width of the basis function - used in cross validation."""
        self.R = R
        self.dist_max = (
            max(np.max(self.src_ele_dists), np.max(self.src_estm_dists)) + self.R
        )
        self.method()
        self.update_operator(estimate=self._operator_estimate)

    def update_lambda(self, lambd):
        """Update the regularization parameter - used in cross validation."""
        self.lambd = lambd
        self.update_operator(estimate=self._operator_estimate)

    def cross_validate(self, lambdas=None, Rs=None, segment=None):
        """Grid-search cross validation with leave-one-out over electrodes.

        By default only lambdas are searched at the current R (Elephant's
        default behavior); pass an ``Rs`` array to search both. The CV error
        is computed with vectorized folds — and, crucially for long
        recordings, only over the supplied time ``segment``, so (R, lambda)
        can be selected once from a short, representative stretch of data
        instead of being recomputed over every sample of a 10-minute file.

        Parameters
        ----------
        lambdas : np.ndarray | None
            Default: np.logspace(-2, -25, 25) with 0 appended.
        Rs : np.ndarray | None
            Default: the current R only.
        segment : tuple[int, int] | None
            (start, stop) time window used for the error; None = all time.

        Returns
        -------
        (R, lambda) : the selected parameters (also applied to the estimator).
        """
        if lambdas is None:
            lambdas = np.hstack((np.logspace(-2, -25, 25, base=10.0), np.zeros(1)))
        if Rs is None:
            Rs = np.array((self.R,)).flatten()
        pots = self.pots
        if segment is not None:
            pots = pots[:, segment[0] : segment[1]]
        errs = np.zeros((Rs.size, lambdas.size))
        index_generator = [
            (np.delete(np.arange(self.n_ele), ii), np.array([ii]))
            for ii in range(self.n_ele)
        ]
        for r_idx, R in enumerate(Rs):
            if Rs.size > 1:
                self.update_R(R)
            for l_idx, lambd in enumerate(lambdas):
                errs[r_idx, l_idx] = self.compute_cverror(
                    lambd, index_generator, pots=pots
                )
        best = np.unravel_index(np.argmin(errs), errs.shape)
        cv_R = Rs[best[0]]
        cv_lambda = lambdas[best[1]]
        self.cv_error = errs[best]
        if Rs.size > 1:
            self.update_R(cv_R)
        self.update_lambda(cv_lambda)
        return cv_R, cv_lambda

    def compute_cverror(self, lambd, index_generator, pots=None):
        """Leave-one-out CV error over electrode folds (vectorized in time).

        For each train/test split the held-out potentials are predicted from
        the trained kernel, ``V_est = K[test, train] @ inv(K[train] + l I) V_train``.
        Elephant computes the same quantity with nested Python loops over
        electrodes and every time sample; here each fold is a small solve and
        one matrix multiplication covering all requested time at once.
        """
        if pots is None:
            pots = self.pots
        err = 0.0
        for idx_train, idx_test in index_generator:
            b_train = self.k_pot[np.ix_(idx_train, idx_train)]
            b_train = b_train + lambd * np.identity(len(idx_train))
            v_train = pots[idx_train]
            v_test = pots[idx_test]
            try:
                beta = np.linalg.solve(b_train, v_train)
            except LinAlgError as exc:
                raise LinAlgError(
                    "Encountered singular matrix: try changing ele_pos slightly"
                ) from exc
            b_test = self.k_pot[np.ix_(idx_test, idx_train)]
            v_est = np.dot(b_test, beta)
            err += np.linalg.norm(v_est - v_test)
        return err

    def sanity(self, true_csd):
        """RMSE between a ground-truth CSD profile (flat, matching the
        estimation grid) and the mean estimate for the construction pots."""
        estimation = self.estimate().mean(axis=1)
        return np.sqrt(np.mean(np.square(true_csd - estimation)))




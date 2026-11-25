"""Current Source Density estimation (kCSD).

Ported from the Elephant package (``elephant.current_source_density_src``,
BSD-3-Clause) by Chaitanya Chintaluri and Michal Czerwinski (Nencki
Institute), implementing Potworowski et al. 2012, "Kernel current source
density method", Neuroinformatics 10:151. This port uses plain NumPy arrays
(no `neo`/`quantities`), precomputes the estimation operator so estimates
are a single matrix multiplication, and makes cross-validation optional and
segment-based.

Quick start
-----------
>>> import numpy as np
>>> from invivosuite.csd import KCSD2D, generate_lfp
>>> ele_pos = np.column_stack([np.repeat(np.linspace(0, 0.5, 5), 5),
...                            np.tile(np.linspace(0, 0.5, 5), 5)])
>>> from invivosuite.csd.utils import small_source_2d
>>> pots = generate_lfp(small_source_2d, ele_pos[:, 0], ele_pos[:, 1])
>>> pots = pots[:, np.newaxis]
>>> k = KCSD2D(ele_pos, pots, gdx=0.025, gdy=0.025, R_init=0.08,
...            n_src_init=500)
>>> csd = k.values()  # (ngx, ngy, n_time)
"""

from .basis import basis_1D, basis_2D
from .kcsd import KCSD
from .kcsd1d import KCSD1D
from .kcsd2d import KCSD2D
from .utils import (
    contains_duplicated_electrodes,
    distribute_srcs_1D,
    distribute_srcs_2D,
    gauss_1d_dipole,
    gauss_2d_dipole,
    generate_lfp,
    small_source_2d,
)

__all__ = [
    "KCSD",
    "KCSD1D",
    "KCSD2D",
    "basis_1D",
    "basis_2D",
    "contains_duplicated_electrodes",
    "distribute_srcs_1D",
    "distribute_srcs_2D",
    "gauss_1d_dipole",
    "gauss_2d_dipole",
    "generate_lfp",
    "small_source_2d",
]

import numpy as np
from scipy import interpolate

from . import utils
from .kcsd import sub_lookup, values


class KCSD:
    """KCSD - The base class for all the KCSD variants.
    This estimates the Current Source Density, for a given configuration of
    electrod positions and recorded potentials, electrodes.
    The method implented here is based on the original paper
    by Jan Potworowski et.al. 2012.
    """

    def __init__(self, ele_pos, pots, **kwargs):
        self.validate(ele_pos, pots)
        self.ele_pos = ele_pos
        self.pots = pots
        self.n_ele = self.ele_pos.shape[0]
        self.n_time = self.pots.shape[1]
        self.dim = self.ele_pos.shape[1]
        self.cv_error = None
        self.parameters(**kwargs)
        # self.estimate_at()
        # self.place_basis()
        # self.create_src_dist_tables()
        # self.method()

    def validate(self, ele_pos, pots):
        """Basic checks to see if inputs are okay
        Parameters
        ----------
        ele_pos : numpy array
            positions of electrodes
        pots : numpy array
            potentials measured by electrodes
        """
        if ele_pos.shape[0] != pots.shape[0]:
            raise Exception(
                "Number of measured potentials is not equal " "to electrode number!"
            )
        if ele_pos.shape[0] < 1 + ele_pos.shape[1]:  # Dim+1
            raise Exception(
                "Number of electrodes must be at least :", 1 + ele_pos.shape[1]
            )
        if utils.contains_duplicated_electrodes(ele_pos):
            raise Exception("Error! Duplicated electrode!")

    def sanity(self, true_csd, pos_csd, estimate="CSD"):
        """Useful for comparing TrueCSD with reconstructed CSD. Computes, the RMS error
        between the true_csd and the reconstructed csd at pos_csd using the
        method defined.
        Parameters
        ----------
        true_csd : csd values used to generate potentials
        pos_csd : csd estimatation from the method
        Returns
        -------
        RMSE : root mean squared difference
        """
        if estimate == "CSD":
            estimation_table = self.k_interp_cross
        elif estimate == "POT":
            estimation_table = self.k_interp_pot
        csd = values(
            pos_csd,
            estimation_table,
            self.k_pot,
            self.lambd,
            self.n_estm,
            self.n_time,
            self.pots,
            self.n_ele,
            estimate="CSD",
        )
        csd = self.process_estimate(csd)
        RMSE = np.sqrt(np.mean(np.square(true_csd - csd)))
        return RMSE

    def parameters(self, **kwargs):
        """Defining the default values of the method passed as kwargs
        Parameters
        ----------
        **kwargs
            Same as those passed to initialize the Class
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
        if self.dim == 3:
            self.ext_z = kwargs.pop("ext_z", 0.0)
            self.zmin = kwargs.pop("zmin", np.min(self.ele_pos[:, 2]))
            self.zmax = kwargs.pop("zmax", np.max(self.ele_pos[:, 2]))
            self.gdz = kwargs.pop("gdz", 0.01 * (self.zmax - self.zmin))
        if kwargs:
            raise TypeError("Invalid keyword arguments:", kwargs.keys())

    def method(self):
        """Actual sequence of methods called for KCSD
        Defines:
        self.k_pot and self.k_interp_cross matrices
        Parameters
        ----------
        None
        """
        self.create_lookup()  # Look up table
        self.update_b_pot()  # update kernel
        self.update_b_src()  # update crskernel
        self.update_b_interp_pot()  # update pot interp

    def create_lookup(self, dist_table_density=20):
        """Creates a table for easy potential estimation from CSD.
        Updates and Returns the potentials due to a
        given basis source like a lookup
        table whose shape=(dist_table_density,)
        Parameters
        ----------
        dist_table_density : int
            number of distance values at which potentials are computed.
            Default 100
        """
        xs, dist_table = sub_lookup(
            self.dist_maxs,
            self.R,
            self.h,
            self.sigma,
            self.basis,
            dist_table_density=20,
            method="kcsd2d",
        )
        self.interpolate_pot_at = interpolate.interp1d(xs, dist_table, kind="cubic")

    def update_b_interp_pot(self):
        """Compute the matrix of potentials generated by every source
        basis function at every position in the interpolated space.
        Updates b_interp_pot
        Updates k_interp_pot
        Parameters
        ----------
        None
        """
        self.b_interp_pot = self.interpolate_pot_at(self.src_estm_dists).T
        self.k_interp_pot = np.dot(self.b_interp_pot, self.b_pot)
        self.k_interp_pot /= self.n_src

    def update_b_src(self):
        """Updates the b_src in the shape of (#_est_pts, #_basis_sources)
        Updates the k_interp_cross - K_t(x,y) Eq17
        Calculate b_src - matrix containing containing the values of
        all the source basis functions in all the points at which we want to
        calculate the solution (essential for calculating the cross_matrix)
        Parameters
        ----------
        None
        """
        self.b_src = self.basis(self.src_estm_dists, self.R).T
        self.k_interp_cross = np.dot(self.b_src, self.b_pot)  # K_t(x,y) Eq17
        self.k_interp_cross /= self.n_src

    def process_estimate(self, estimation):
        """Function used to rearrange estimation according to dimension, to be
        used by the fuctions values
        Parameters
        ----------
        estimation : np.array
        Returns
        -------
        estimation : np.array
            estimated quantity of shape (ngx, ngy, ngz, nt)
        """
        if self.dim == 1:
            estimation = estimation.reshape(self.ngx, self.n_time)
        elif self.dim == 2:
            estimation = estimation.reshape(self.ngx, self.ngy, self.n_time)
        elif self.dim == 3:
            estimation = estimation.reshape(self.ngx, self.ngy, self.ngz, self.n_time)
        return estimation

from collections import namedtuple
from typing import Literal

import numpy as np
from scipy import linalg, stats

from .mvar_algs import mcarns, mcarvm, clmsm, arfitcaps
from .pdc_utilities import a_to_f

MvarOutput = namedtuple(
    "MvarOutput", ["IP", "pf", "A", "pb", "B", "ef", "eb", "vaic", "vaicv"]
)


def mvar(
    u,
    max_ip=0,
    alg: Literal["ns", "mlsm", "vm", "qr"] = "ns",
    criterion: Literal["aic", "hq", "fpe", "s", "fixed"] = "aic",
    flgverbose=False,
):
    nchannels, nsamples = u

    if criterion == "fixed":
        if max_ip == 0:
            pf = u.dot(u.T)
            pb = pf
            ef = u
            eb = u
            npf = pf.shape
            A = np.zeros((npf, npf, 0))
            B = A
            vaic = np.linalg.det(pf)
            vaicv = vaic
            IP = 0
            return IP, pf, A, pb, B, ef, eb, vaic, vaicv
        IP = max_ip
        if alg == "ns":
            pf, A, pb, B, ef, eb = mcarns(u, IP)
            pf = pf / nsamples
        elif alg == "mlsm":
            pf, A, ef = clmsm(u, IP)
            pf = pf / nsamples
        elif alg == "vm":
            pf, A, pb, B, ef, eb = mcarvm(u, IP)
            pf = pf / nsamples
        elif alg == "qr":
            pf, A, ef = arfitcaps(u, IP)
            B = []
            eb = []
            pb = []
        else:
            raise ValueError("alg method not recognized")
        vaic = nsamples * np.log(np.det(pf)) + 2 * (nchannels**2) * IP
        vaicv = vaic
        return IP, pf, A, pb, B, ef, eb, vaic, vaicv

    vaicv_int = 0

    max_order = 30
    u_bound = int((3 * np.sqrt(nsamples)) / nchannels)
    u_bound = np.max([u_bound, max_order])
    if max_order != max_ip:
        max_order = max_ip
        u_bound = max_ip
    IP = 1

    if flgverbose:
        print(f"max order limited to: {max_order}")

    vaicv = np.zeros(max_order + 1, 1)

    while IP < u_bound:
        m = IP
        if alg == "ns":
            npf, na, npb, nb, nef, neb = mcarns(u, IP)
        elif alg == "mlsm":
            npf, na, nef = clmsm(u, IP)
        elif alg == "qr":
            npf, na, npb, nef, neb = mcarvm(u, IP)
        else:
            npf, na, nef = arfitcaps(u, IP)

        if criterion == "aic":
            vaic = nsamples * np.log(np.det(pf)) + 2 * (nchannels**2) * IP
        elif criterion == "hq":
            vaic = (
                nsamples * np.log(nsamples)
                + 2 * np.log(np.log(nsamples)) * (nchannels**2) * IP
            )
        elif criterion == "s":
            vaic = (
                nsamples * np.log(np.det(pf)) + np.log(nsamples) * (nchannels**2) * IP
            )
        else:
            vaic = np.log(
                np.det(npf)
                * ((nsamples + nchannels * IP + 1) / (nsamples - nchannels * IP - 1))
                ** nchannels
            )
        vaicv[IP + 1] = vaic

        if flgverbose:
            print(f"IP = {IP} and vaic = {vaic}")

        if np.all(vaic > vaicv_int) and IP != 1:
            vaic = vaicv_int

        vaicv_int = vaic
        pf = npf
        A = na
        ef = nef

        if alg == "ns" or alg == "vm":
            B = nb
            eb = neb
            pb = npb
        else:
            B = []
            eb = []
            pb = []

        IP += 1

    if flgverbose:
        print(" ")

    IP -= 1
    vaic = vaicv_int
    vaicv = vaicv[1 : IP + 2]

    if alg == "ns" or alg == "vm":
        pf = pf / nsamples
    elif alg == "qr":
        pass
    else:
        pf = pf / nsamples


def asymp_pdc(u, A, pf, nFreqs, metric, alpha):
    nchannels, nsamples = u.shape
    if nchannels > nsamples:
        raise AttributeError("Data needs to be orientented nchannels x nsamples")

    nchannels, _, p = A.shape

    Af = a_to_f()


def fastasymalg(
    s,
    ARcoef,
    ecov,
    SR=None,
    alpha=0,
    freq=(4, 10),
    measure: Literal["DTF", "iDTF", "gDTF", "PDC", "iPDC", "gPDC"] = "PDC",
):
    measure = measure.lower()
    if measure in {"dtf", "idtf", "gdtf"}:
        metric = "gamma_group"
    elif measure in {"pdc", "ipdc", "gpdc"}:
        metric = "pi_group"
    else:
        raise ValueError("measure needs to be DTF, iDTF, gDTF, PDC, iPDC or gPDC")

    n_samples, nchannels = s.shape

    if SR is None:
        SR = 2 * freq[1]

    if s.shape[0] < s.shape[1]:
        s = s.T
    s -= s.mean(axis=0)

    icdf_norm_alpha = stats.norm.ppf(1 - alpha / 2.0, 0, 1)

    if not linalg.issymmetric(ecov):
        ecov = np.triu(ecov) + np.triu(ecov).T

    p = ARcoef.shape[2]
    AR = np.reshape(ARcoef, (nchannels, nchannels * p))
    Ecovdotproduct = ecov * ecov.T

    if metric == "gamma_group":
        if measure == "dtf":
            evar_d = np.eye(nchannels)
        else:
            evar_d = np.eye(nchannels).dot(ecov)
        Sd = evar_d
        if measure == "idtf":
            Sd = ecov
    else:
        if measure == "pdc":
            evar_d = np.eye(nchannels)
        else:
            evar_d = np.inv(np.eye(nchannels).dot(ecov))
        if measure == "ipdc":
            Sd = np.pinv(ecov)
        else:
            Sd = evar_d
        ce = np.diag(evar_d).dot(np.diag(ecov))

    Phi = np.zeros((nchannels, nchannels, len(freq)))
    pvalues = np.zeros((nchannels, nchannels, len(freq)))
    TH = np.zeros((nchannels, nchannels, len(freq)))
    CIup = np.zeros((nchannels, nchannels, len(freq)))
    CIlow = np.zeros((nchannels, nchannels, len(freq)))
    SS = np.zeros((nchannels, nchannels, len(freq)))
    gamma = np.zeros(((nchannels * p, nchannels * p)))

    for m in range(p):
        for n in range(m, p):
            SlagOut = (
                s[n - m : s.shape[0] - m + 1, :].T
                * s[1 : s.shape[1] - n + 1, :]
                / n_samples
            )
            gamma[
                m * nchannels + 1 : m * nchannels,
                n * nchannels + 1 : n * nchannels,
            ] = SlagOut
            gamma[
                n * nchannels + 1 : n * nchannels,
                m * nchannels + 1 : m * nchannels,
            ] = SlagOut.T

    invgamma = np.linalg.pinv(gamma)

    for ff in range(0, freq):
        f1 = freq[0]
        f = (f1 - 1) / (SR)
        C1 = np.cos(2 * np.pi * f * np.arange(1, p + 1))
        S1 = np.sin(2 * np.pi * f * np.arange(1, p + 1))
        C = np.array([C1, -S1])
        Af = np.eye(nchannels) - AR.dot(np.kron(C1 - 1j * S1, np.eye(nchannels)).T)

        Hf = np.linalg.pinv(Af)
        SS = (Hf.dot(ecov)).dot(Hf.T)
        omega_x = np.zeros((nchannels, nchannels))
        omega_ecov = np.zeros((nchannels, nchannels))

        if metric == "gamma_group":
            X = Hf
            num = (abs(X) ** 2).dot(evar_d)
            if measure == "idtf":
                den = np.diag(np.real((X.dot(ecov)).dot(X.T)))
            else:
                den = np.sum(num, axis=1)
        else:
            X = Af
            num = evar_d.dot(abs(X)) ** 2
            if measure == "ipdc":
                den = np.diag(np.real((X.T.dot(np.pinv(ecov))).dot(X)))
                den = den.T
            else:
                den = np.sum(num)

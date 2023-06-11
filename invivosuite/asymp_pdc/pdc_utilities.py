import numpy as np


def a_to_f(A, nf):
    nchannels, _, p = A.shape
    jimag = 1j

    x1 = np.arange(0, nf)
    x2 = np.arange(1, p + 1)
    exponents = np.reshape(((-jimag * np.pi * np.kron(x1, x2)) / nf, (p, nf))).T

    Areshaped = np.reshape(A, (nchannels, nchannels, 1, p))

    Af = np.zeros((nchannels, nchannels, nf, p))

    for kk in range(nf):
        Af[:, :, kk, :] = Areshaped

    for i in range(nchannels):
        for k in range(nchannels):
            Af[i, k, :, :] = np.reshape(Af[i, k, :, :], (nf, p)) * np.exp(exponents)

    Af = Af.transpose([2, 0, 1, 3])

    Al = np.zeros(nf, nchannels, nchannels)

    for kk in range(nf):
        temp = np.zeros(nchannels, nchannels)
        for k in range(p):
            temp = temp + np.reshape(Af[kk, :, :, k], (nchannels, nchannels))
        temp = np.eye(nchannels) - temp
        Al[kk, :, :] = np.reshape(temp, (1, nchannels, nchannels))

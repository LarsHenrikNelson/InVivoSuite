from typing import NamedTuple
import numpy as np


class Neuron(NamedTuple):
    number: int
    a: float
    b: float
    c: float
    d: float


inhibitory = Neuron(200, 0.02, 0.2, -65, 2)
excitatory = Neuron(800, 0.02, 0.2, -65, 8)


def simulation(
    excitatory: Neuron = excitatory,
    inhibitory: Neuron = inhibitory,
    conduction_delay: int = 20,
    synaptic_strength: float = 10.0,
    time_of_sim: int = 60,
    num_synapses: int = 100,
    fs: int = 1000,
    seed: int = 42,
):
    Ne = excitatory.number
    Ni = inhibitory.number
    N = Ne + Ni
    a = np.ones((N, 1), dtype=float)
    a[:Ne] *= excitatory.a
    a[Ne:] *= inhibitory.a

    d = np.ones((N, 1), dtype=float)
    d[:Ne] *= excitatory.d
    d[Ne:] *= inhibitory.d

    v = np.ones((N, 1), dtype=float)
    v[:Ne] *= excitatory.c
    v[Ne:] *= inhibitory.c

    u = v.copy()
    u[:Ne] *= excitatory.b
    u[Ne:] *= inhibitory.b

    rng = np.random.default_rng(seed)

    post = rng.uniform((N, num_synapses))
    post[:Ne, :] *= N
    post[Ne:, :] *= Ne

    weights = np.zeros((N, num_synapses))
    weights_deriv = np.zeros((N, num_synapses))
    delays = [[None for _ in range(num_synapses)] for _ in range(N)]
    scalar = int(num_synapses / conduction_delay)
    for i in range(N):
        if i < Ne:
            for j in range(conduction_delay):
                delays[i, j] = scalar * j + np.arange(scalar)


def simulation_no_stdp(
    excitatory: Neuron = excitatory,
    inhibitory: Neuron = inhibitory,
    seed: int = 42,
    time: int = 1000,
    synaptic_strength: int | tuple = (5, 2),
):
    rng = np.random.default_rng(seed)

    Ne = excitatory.number
    Ni = inhibitory.number
    N = Ne + Ni

    re = rng.random(Ne)
    ri = rng.random(Ni)

    a = np.zeros(N)
    a[:Ne] = np.full(Ne, excitatory.a)
    a[Ne:] = inhibitory.a + ri * 0.08

    b = np.zeros(N)
    b[:Ne] = np.full(Ne, excitatory.b)
    b[Ne:] = inhibitory.b - ri * 0.05

    c = np.zeros(N)
    c[:Ne] = excitatory.c + 15 * re**2
    c[Ne:] = np.full(Ni, inhibitory.c)

    d = np.zeros(N)
    d[:Ne] = excitatory.d - 6 * re**2
    d[Ne:] = np.full(Ni, inhibitory.d)

    S = rng.random((N, N))
    S[:, :Ne] *= 0.5
    S[:, Ne:] *= -1.0

    v = np.full(N, -65)
    u = b * v

    firings = {"timestamps": [], "units": []}

    if isinstance(synaptic_strength, tuple):
        ss = np.zeros(N)
        ss[:Ne] += synaptic_strength[0]
        ss[Ne:] += synaptic_strength[1]
    else:
        ss = np.full(N, synaptic_strength)

    for t in range(time):
        current = rng.normal(size=N) * ss  # thalamic input
        fired = np.where(v >= 30.0)[0]
        firings["timestamps"].extend(fired * 0 + t)
        firings["units"].extend(fired)
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]
        current += S[:, fired].sum(axis=1)
        dv = 0.5 * (0.04 * v**2 + 5 * v + 140 - u + current)
        v = v + dv  # step 0.5 ms
        v = v + dv  # for numerical
        u = u + a * (b * v - u)  # stability
    return firings


def izhikevich_neuron(
    current,
    a=0.02,
    b=0.2,
    c=-65,
    d=2,
    v0=-65,
    threshold=30,
    time_step = 0.1,
    params: tuple[float, float, float] = (0.04, 5, 140),
):
    v = np.zeros(current.size)
    v[0] = v0
    u =  np.zeros(current.size)
    u[0] = b * v0
    one, two, three = params
    for index in range(1, current.size):
        vc = v[index-1]
        uc = u[index-1]

        if vc >= threshold:
            vc = v[index]
            v[index] = c
            u[index] = uc+d
        else:
            Ic = current[index-1]
            dv = (one * vc**2 + two * vc + three - uc +Ic) * 0.5
            du = a * (b * vc - uc)
            v[index] = vc + dv*time_step
            u[index] = uc + du*time_step
    return v

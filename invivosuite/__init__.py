from . import (
    spectral,
    utils,
)
from .acq import (
    AcqManager,
    LFPManager,
    SpkLFPManager,
    SpkManager,
    load_hdf5_acqs,
)
from .functions import *

spectral.load_wisdom()

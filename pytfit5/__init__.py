import numpy as np

# Handle the NumPy 1.x to 2.0 trapezoid rename seamlessly.
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from . import bls_cpu
from . import transitPy5
from . import transitmodel
from . import keplerian
from . import transitfit
from . import transitplot
from . import transitmcmc
from . import mcmcroutines
from . import synthetic
from . import period_validation
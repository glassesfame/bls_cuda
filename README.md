# pytfit5

**An implementation of TransitFit5 and BLS in Python (with Numba).**

This package provides tools for transit search and fitting, including Box-Least-Squares (BLS) algorithms optimized for both CPU and GPU (via CUDA).

## Installation

### Prerequisites
*   Python 3.8+

### Installing from Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jasonfrowe/bls_cuda.git
    cd bls_cuda
    ```

2.  **Install via pip:**
    ```bash
    pip install .
    ```

## Features

### Synthetic Lightcurve Generation
Generate realistic synthetic transit lightcurves with configurable stellar noise:

```python
import pytfit5.synthetic as syn

# Generate synthetic data with multi-component stellar noise
phot, sol = syn.generate_synthetic_lightcurve(
    time,
    period=10.0,              # Orbital period (days)
    t0=0.0,                   # Transit center time
    depth=0.01,               # Transit depth (1%)
    duration=0.1,             # Transit duration (days)
    white_noise=0.001,        # White noise amplitude
    stellar_noise_type='multi',  # Noise type: 'none', 'red', 'drw', 'multi'
    stellar_noise_amplitude=0.01,  # Stellar noise amplitude
    rotation_period=25.0      # Stellar rotation period (days)
)
```

**Stellar Noise Models:**
- `'none'`: White noise only
- `'red'`: Power-law (1/f^α) red noise
- `'drw'`: Damped random walk with exponential covariance
- `'multi'`: Multi-component model (granulation + activity + rotation)

**Returns:**
- `phot`: `phot_class` object with time, flux, ferr, itime, tflag, icut, flux_f arrays
- `sol`: Dictionary containing injected transit parameters

### Iterative Detrending
Automatically detect and protect transit signals during detrending:

The `run_polyfilter_iterative()` function iteratively identifies transit-like features (negative deviations) and excludes them from detrending, preventing signal suppression. This is especially useful when:
- Transit times are unknown
- Multiple transits of different depths exist
- Red noise is significant

**How it works:**
1. Apply initial polynomial detrending
2. Detect significant negative deviations (< -sigma_threshold)
3. Group consecutive outliers and filter by minimum duration
4. Mask detected features and re-detrend
5. Repeat until convergence or max_iters reached

## Usage

### Python Import (Recommended)
You can import the package and access the submodules (BLS, transitfit5, MCMC, etc.) directly.

```python
import pytfit5 as pytfit5

# Access submodules using the built-in aliases:
# pytfit5.gbls   -> bls_cpu
# pytfit5.tpy5   -> transitPy5
# pytfit5.kep    -> keplerian
# pytfit5.tmcmc  -> transitmcmc
```

### Jupyter Notebook Example
For a full demonstration of how to run the code, please refer to the example notebook included in this repository:

**`pytfit5_example.ipynb`**

### API Reference: Unified Configuration Class

The `tpy5_inputs_class()` is a **unified configuration class** that controls all parameters for photometry processing, detrending, and BLS transit search:

> **Note:** `gbls_inputs_class` has been deprecated and consolidated into `tpy5_inputs_class`. For backward compatibility, `gbls_inputs_class` is aliased to `tpy5_inputs_class`, but new code should use `tpy5_inputs_class` directly.

#### Detrending Parameters
- `boxbin` (float, default: 2.0): Detrending window width in days
- `nfitp` (int, default: 2): Polynomial order for detrending (2 = quadratic)
- `gapsize` (float, default: 0.5): Gap detection threshold in days

#### Iterative Detrending Parameters (for `run_polyfilter_iterative`)
- `iter_max_iters` (int, default: 5): Maximum iterations for automatic transit detection
- `iter_sigma_threshold` (float, default: 3.0): Sigma threshold for detecting transit-like features
- `iter_min_duration` (float, default: 0.05): Minimum transit duration in days (~1 hour)

#### Outlier Clipping Parameters
- `dsigclip` (float, default: 3.0): Sigma threshold for derivative-based clipping
- `sigclip` (float, default: 3.0): Sigma threshold for simple clipping
- `boxwindow` (float, default: 2.0): Time window for rolling statistics in days
- `boxsigma` (float, default: 3.0): Sigma threshold for time-window clipping
- `nsampmax` (int, default: 6): Sample size for derivative routine
- `fstd_cut` (int, default: 5): Simple sigma-clipping threshold

#### Stellar Parameters
- `mstar` (float, default: 1.0): Stellar mass in solar masses
- `rstar` (float, default: 1.0): Stellar radius in solar radii
- `teff` (float, default: 5777): Effective temperature in Kelvin
- `logg` (float, default: 4.5): Surface gravity (log g) in cgs
- `feh` (float, default: 0.0): Metallicity [Fe/H]

#### BLS Transit Search Parameters
- `filename` (str, default: "filename.txt"): Lightcurve filename
- `lcdir` (str, default: ""): Lightcurve directory path
- `zerotime` (float, default: 0.0): Time offset for epochs
- `freq1` (float, default: -1): Minimum search frequency in cycles/day (auto if -1)
- `freq2` (float, default: -1): Maximum search frequency in cycles/day (auto if -1)
- `ofac` (float, default: 8.0): Oversampling factor for frequency grid
- `minbin` (int, default: 5): Minimum number of bins in transit
- `plots` (int, default: 1): Plot mode (0=none, 1=display, 2=save+display, 3=save only)
- `multipro` (int, default: 1): Enable multiprocessing (0=single thread, 1=parallel)
- `normalize` (str, default: "iterative_baseline"): BLS normalization method
- `return_spectrum` (bool, default: False): Return full BLS spectrum arrays
- `oneoverf_correction` (bool, default: True): Apply 1/f noise correction for long periods

**Example Usage:**
```python
import pytfit5.transitPy5 as tpy5

# Read photometry
phot = tpy5.readphot("lightcurve.dat")

# Configure unified parameters (detrending + BLS)
tpy5_inputs = tpy5.tpy5_inputs_class()

# Detrending settings
tpy5_inputs.boxbin = 3.0  # 3-day detrending window
tpy5_inputs.nfitp = 2     # Quadratic polynomial

# BLS settings
tpy5_inputs.zerotime = 0.0
tpy5_inputs.mstar = 1.0
tpy5_inputs.rstar = 1.0
tpy5_inputs.freq1 = 0.1  # Search down to 10-day periods
tpy5_inputs.normalize = "iterative_baseline"

# Standard detrending
tpy5.run_polyfilter(phot, tpy5_inputs)

# OR use iterative detrending to auto-detect and protect transits
tpy5_inputs.iter_max_iters = 5       # Maximum 5 iterations
tpy5_inputs.iter_sigma_threshold = 3.0  # 3-sigma detection
tpy5_inputs.iter_min_duration = 0.05    # 1-hour minimum duration
tpy5.run_polyfilter_iterative(phot, tpy5_inputs)

# Run BLS with the same unified config
import pytfit5.bls_cpu as gbls
gbls_ans = gbls.bls(tpy5_inputs, phot.time, phot.flux_f)
```

### Lightcurve Visualization

The `plot_lightcurve_summary()` function provides comprehensive visualization of photometry data:

```python
import pytfit5.transitplot as transitp
import pytfit5.bls_cpu as gbls

# Basic plot: raw and detrended flux only
fig = transitp.plot_lightcurve_summary(phot, tpy5_inputs=tpy5_inputs)

# Enhanced plot: includes BLS results with phased lightcurve and transit closeup
gbls_ans = gbls.bls(tpy5_inputs, phot.time, phot.flux_f)
fig = transitp.plot_lightcurve_summary(phot, bls_ans=gbls_ans, tpy5_inputs=tpy5_inputs)
```

**Features:**
- **2-panel mode** (no BLS): Shows raw flux and detrended flux in vertical stack
- **4-panel mode** (with BLS): Adds phased lightcurve and transit closeup with statistics
- Automatically marks transit locations in time series
- Transit closeup shows ±1 duration window around transit center
- Statistics panel displays period, T0, SNR, duration, and depth

### ⚠️ Deprecation Notice
**Command Line Usage:** Previous versions of this code allowed for execution via command line scripts. This method is **deprecated**. Please use the Python API as described above.

## Dependencies
This package requires the following libraries (installed automatically):
*   `numpy`
*   `matplotlib`
*   `tqdm`
*   `numba`
*   `scipy`
*   `pandas`
*   `astroquery`

## License
This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

## Contact
Jason Rowe - jason@jasonrowe.org

## Contributing
If you'd like to contribute to this project, go for it! There are a number of to-dos 
1. ~~Code speed can likely be made much faster.  (shared memory vs global memory)~~
2. ~~Better choices of blocks and threads-per-block needs to be explored~~
3. ~~Making the code base into an installable package~~
4. ~~Make CPU threading more efficient (spread around short-period jobs that take longer)~~
5. Optimizing GPU memory transfers
6. Allow transit modelling to have different parameters for different planets
7. Better examples for TTV fitting
8. and much more.. 

## License
This project is licensed under the GNU General Public License (GPL) version 3 or later.

## Acknowledgments
Thank you to Canada Research Chairs, NSERC Discovery, Digital Alliance Canada, Calcul Quebec, FRQNT for financial and hardware support.

This code was initially developed during the Bishop's University Winter Reading Week, making good use of profession development resources. 

The implementation of [TransitFit5](https://github.com/jasonfrowe/Kepler) in Python was developed by Alexis Roy (Universite de Sherbrooke) supported by an NSERC Undergraduate Student Research Award (USRA) and iREx Trottier Fellowship.

This code is directly adopted from Kovacs et al. 2002 : A box-fitting algorithm in the search for periodic transits 

If you find these codes useful please reference:  
Rowe et al. 2014 ApJ, 784, 45   
Rowe et al. 2015 ApJs, 217, 16  

## Change Log
2025/11/27 : Big refresh of the code base.  First steps to pip installable package
2025/03/08 : Initial Update  
2025/03/09 : Added a 'V2'.  V2 works best with TESS CVZ lc, V1 works best with Kepler.  
2025/03/09 : Added CPU version (Numba + threading)  
2025/03/10 : V2 is now faster for CPU  

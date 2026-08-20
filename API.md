# pytfit5 API Reference

**Version 1.0.0**

This document provides detailed API documentation for the `pytfit5` package, covering Box-Least-Squares (BLS) transit search, transit modeling, and synthetic lightcurve generation.

---

## Table of Contents

1. [Installation](#installation)
2. [Package Structure](#package-structure)
3. [BLS Module (`pytfit5.gbls`)](#bls-module-pytfit5gbls)
4. [Transit Model Module (`pytfit5.transitmodel`)](#transit-model-module-pytfit5transitmodel)
5. [Synthetic Lightcurves (`pytfit5.synthetic`)](#synthetic-lightcurves-pytfit5synthetic)
6. [Keplerian Utilities (`pytfit5.kep`)](#keplerian-utilities-pytfit5kep)
7. [MCMC Module (`pytfit5.tmcmc`)](#mcmc-module-pytfit5tmcmc)
8. [Examples](#examples)

---

## Installation

```bash
git clone https://github.com/jasonfrowe/bls_cuda.git
cd bls_cuda
pip install -e .
```

### Dependencies

- `numpy` - Array operations
- `scipy` - Signal processing and optimization
- `matplotlib` - Plotting
- `numba` - JIT compilation for performance
- `tqdm` - Progress bars
- `pandas` - Data handling
- `astroquery` - Astronomical data access

---

## Package Structure

```python
import pytfit5

# Main submodules (accessible via aliases):
pytfit5.gbls   # BLS transit search (bls_cpu module)
pytfit5.tpy5   # Transit fitting (transitPy5 module)
pytfit5.kep    # Keplerian orbit utilities
pytfit5.tmcmc  # MCMC transit fitting
pytfit5.transitmodel  # Transit light curve modeling
pytfit5.occult        # Occultation modeling
pytfit5.effects       # Astrophysical effects (beaming, ellipsoidal, etc.)
pytfit5.synthetic     # Synthetic lightcurve generation
```

---

## BLS Module (`pytfit5.gbls`)

The Box-Least-Squares (BLS) algorithm is used to detect periodic transit signals in photometric time series data.

### Core Classes

#### `tpy5_inputs_class`

Unified configuration class for BLS search, detrending, and transit fitting parameters.

**BLS Search Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | str | `"filename.txt"` | Input light curve file path |
| `lcdir` | str | `""` | Light curve directory |
| `zerotime` | float | `0.0` | Reference time offset subtracted from epochs (days). Use this to match your time system (e.g., BJD-2457000) |
| `freq1` | float | `-1` | Minimum search frequency (c/d) |
| `freq2` | float | `-1` | Maximum search frequency (c/d) |
| `ofac` | float | `8.0` | Oversampling factor for frequency grid |
| `mstar` | float | `1.0` | Stellar mass (solar masses) |
| `rstar` | float | `1.0` | Stellar radius (solar radii) |
| `nper` | int | `50000` | Maximum number of periods to search |
| `minbin` | int | `5` | Minimum bins in transit |
| `plots` | int | `1` | Plot mode: 0=none, 1=X11, 2=PNG+X11, 3=PNG |
| `multipro` | int | `1` | Enable multiprocessing (0=off, 1=on) |
| `normalize` | str | `"iterative_baseline"` | Normalization mode (see below) |
| `return_spectrum` | bool | `False` | If True, return full BLS spectrum in result |
| `oneoverf_correction` | bool | `False` | Enable 1/f baseline correction |
| `oneoverf_extrapolate` | bool | `True` | Enable baseline/noise extrapolation for long periods |

**Pulse Search Parameters (for single-event detection):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pulse_min_duration_hours` | float | `1.0` | Minimum event duration to search (hours) |
| `pulse_max_duration_hours` | float | `12.0` | Maximum event duration to search (hours) |
| `pulse_bin_duration_days` | float | `10.0/1440.0` | Binning resolution (~10 minutes) |

**Detrending Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `boxbin` | float | `5.0` | Detrending window size (days) |
| `nfitp` | int | `3` | Polynomial order for detrending |
| `iter_max_iters` | int | `5` | Maximum iterations for iterative detrending |
| `iter_sigma_threshold` | float | `3.0` | Sigma threshold for outlier rejection |
| `iter_min_duration` | float | `0.05` | Minimum transit duration to protect (days) |

**Normalization Modes:**

- `"none"` - No normalization, raw BLS power
- `"mad"` - Median Absolute Deviation normalization with rolling window
- `"percentile_mad"` - 75th percentile baseline with MAD noise estimation
- `"coverage_mad"` - MAD weighted by expected duty cycle at each frequency
- `"iterative_baseline"` - **Default and recommended** - Uses sigma-clipping to robustly identify continuum by iteratively masking strong peaks. Prevents high-SNR signals from biasing baseline estimation. Most robust for varying signal strengths.

**1/f Baseline Correction and Extrapolation:**

When `oneoverf_correction=True`, the BLS code removes low-frequency trends (red noise, stellar variability) before computing the periodogram:

1. **Iterative Baseline Estimation**: Uses sigma-clipping to identify the true continuum level while masking strong transit peaks
2. **Baseline Subtraction**: Removes the estimated baseline before normalization

When `oneoverf_extrapolate=True` (default), the code handles long-period signals more robustly:

1. **Detection Threshold**: For periods > 0.5× baseline, fewer than 2 full periods are observed
2. **Well-Sampled Region**: Computes baseline and noise statistics for periods < 0.3× baseline
3. **Flat Extrapolation**: Extends baseline and noise as constant values for long periods
4. **Smooth Transition**: Uses cosine-tapered blending in the 0.3-0.5× baseline transition region

This prevents artificial suppression of long-period signals and maintains proper SNR scaling.

**Note**: The `gbls_inputs_class` is deprecated. Use `tpy5_inputs_class` instead, which provides unified configuration for BLS, detrending, and transit fitting.

**Example:**

```python
import pytfit5.transitPy5 as tpy5

# Configure BLS search using unified inputs class
inputs = tpy5.tpy5_inputs_class()
inputs.freq1 = 1.0 / 60.0  # Search periods from 1 to 60 days
inputs.freq2 = 2.0
inputs.ofac = 10.0
inputs.mstar = 1.0
inputs.rstar = 1.0
inputs.normalize = "iterative_baseline"
inputs.oneoverf_correction = True  # Enable 1/f baseline correction
inputs.oneoverf_extrapolate = True  # Enable long-period extrapolation
```

---

#### `gbls_ans_class`

Results class storing BLS detection information.

**Attributes:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `epo` | float | Best-fit epoch (days). For periodic: first transit time. For pulse: event time |
| `bper` | float | Best-fit period (days). **Set to 0 for single pulse detections** |
| `bpower` | float | Maximum BLS power |
| `snr` | float | Signal-to-noise ratio |
| `tdur` | float | Transit/event duration (days) |
| `depth` | float | Transit/event depth (fractional) |
| `periods` | np.ndarray | Full period array (if `return_spectrum=True`) |
| `power` | np.ndarray | Full power array (if `return_spectrum=True`) |
| `freqs` | np.ndarray | Full frequency array (if `return_spectrum=True`) |

**Note:** When using `bls_pulse()`, check `ans.bper`:
- `bper > 0`: Periodic signal detected (repeating transits)
- `bper == 0`: Single pulse detected (one-time event)

---

### Core Functions

pytfit5 provides three search modes to accommodate different scenarios:

1. **`bls()`** - Periodic transit search only
2. **`compute_pulse_search()`** - Single event search only (fast)
3. **`bls_pulse()`** - Combined search (runs both, returns best)

---

#### `bls(inputs, time, flux)`

**Periodic transit search** - Searches for repeating transit signals.

**Parameters:**

- `inputs` (tpy5_inputs_class): BLS configuration
- `time` (np.ndarray): Time array (days)
- `flux` (np.ndarray): Flux array (normalized around 1)

**Returns:**

- `ans` (gbls_ans_class): Detection results with `bper > 0`

**When to use:**
- You expect **repeating transits** (planet with multiple transits observed)
- You want the standard BLS periodogram
- You need high precision period determination

**Example:**

```python
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5

# Load or generate data
time, flux = gbls.readfile("lightcurve.txt")

# Configure search
inputs = tpy5.tpy5_inputs_class()
inputs.freq1 = 0.05  # Minimum frequency (max period ~20 days)
inputs.freq2 = 2.0   # Maximum frequency (min period 0.5 days)
inputs.normalize = "iterative_baseline"
inputs.oneoverf_correction = True

# Run periodic BLS
ans = gbls.bls(inputs, time, flux)

print(f"Period: {ans.bper:.6f} days")
print(f"Epoch: {ans.epo:.6f} days")
print(f"SNR: {ans.snr:.2f}")
```

---

#### `compute_pulse_search(time, flux, min_duration_hours=1.0, max_duration_hours=12.0, bin_duration_days=None)`

**Single event search** - Fast detection of one-time events (no periodicity assumed).

**Parameters:**

- `time` (np.ndarray): Time array (days), **must start at 0** (pre-shifted)
- `flux` (np.ndarray): Flux array, **must be zero-centered** (pre-processed)
- `min_duration_hours` (float): Minimum event duration to search (hours)
- `max_duration_hours` (float): Maximum event duration to search (hours)
- `bin_duration_days` (float, optional): Binning resolution (default: 10 minutes)

**Returns:**

- `result` (dict or None): Detection results containing:
  - `'t0'`: Event center time in time array frame
  - `'duration'`: Event duration (days)
  - `'depth'`: Event depth (negative for dips)
  - `'snr'`: Signal-to-noise ratio
  - `'spectrum_duration'`: Duration grid (hours)
  - `'spectrum_snr'`: SNR vs duration spectrum

**When to use:**
- **Single transit events** (planet transits only once in dataset)
- **Stellar flares** or other one-off astrophysical events  
- **Fast screening** (much faster than full BLS)
- You want to control preprocessing explicitly

**Important:** This is a **low-level function** - you must preprocess the data:
1. Shift time to start at 0: `time_proc = time - np.min(time)`
2. Zero-center flux: `flux_proc = flux - np.median(flux)`

**Example:**

```python
import pytfit5.bls_cpu as gbls
import numpy as np

# Load data
time, flux = gbls.readfile("lightcurve.txt")

# REQUIRED: Preprocess data
mintime = np.min(time)
time_proc = time - mintime  # Shift to start at 0
flux_proc = flux - np.median(flux)  # Zero-center

# Run pulse search only
result = gbls.compute_pulse_search(
    time_proc,
    flux_proc,
    min_duration_hours=1.0,   # 1-12 hour events
    max_duration_hours=12.0
)

if result is not None:
    # Adjust T0 back to original time frame
    t0_original = result['t0'] + mintime
    print(f"Event time: {t0_original:.6f} days")
    print(f"Duration: {result['duration']*24:.2f} hours")
    print(f"SNR: {result['snr']:.2f}")
```

---

#### `bls_pulse(inputs, time, flux)`

**Combined periodic + pulse search** - Runs both searches, returns the best detection.

**Parameters:**

- `inputs` (tpy5_inputs_class): Configuration with both BLS and pulse parameters
- `time` (np.ndarray): Time array (days)
- `flux` (np.ndarray): Flux array (normalized around 1)

**Returns:**

- `ans` (gbls_ans_class): Best detection (highest SNR) with:
  - `bper > 0`: Periodic signal won
  - `bper == 0`: Single pulse won

**When to use:**
- **Exploratory analysis** when signal type is unknown
- Searching for both repeating and single transits
- You want the algorithm to choose automatically

**Example:**

```python
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5

# Configure search (includes both periodic and pulse parameters)
inputs = tpy5.tpy5_inputs_class()

# Periodic search settings
inputs.freq1 = 1.0 / 60.0  # Down to 60-day periods
inputs.freq2 = 2.0
inputs.normalize = "iterative_baseline"
inputs.oneoverf_correction = True

# Pulse search settings
inputs.pulse_min_duration_hours = 1.0
inputs.pulse_max_duration_hours = 8.0

inputs.plots = 1  # Show diagnostic plots

# Run combined search
ans = gbls.bls_pulse(inputs, time, flux)

# Check what was found
if ans.bper == 0:
    print(">> SINGLE PULSE DETECTED")
    print(f"Event time: {ans.epo:.6f} days")
    print(f"Duration: {ans.tdur*24:.2f} hours")
else:
    print(">> PERIODIC SIGNAL DETECTED")
    print(f"Period: {ans.bper:.6f} days")
    print(f"Epoch: {ans.epo:.6f} days")

print(f"SNR: {ans.snr:.2f}")
print(f"Depth: {ans.depth*1e6:.1f} ppm")
```

**How it works:**
1. Runs full periodic BLS search
2. Runs pulse search across all times and durations
3. Compares SNR values
4. Returns result with highest SNR
5. Generates appropriate plots based on winner

---

### Function Comparison

| Feature | `bls()` | `compute_pulse_search()` | `bls_pulse()` |
|---------|---------|--------------------------|---------------|
| **Speed** | Slow | Very Fast | Slow (runs both) |
| **Finds periodic signals** | ✓ | ✗ | ✓ |
| **Finds single events** | ✗ | ✓ | ✓ |
| **Preprocessing** | Automatic | Manual | Automatic |
| **Returns period** | Always > 0 | N/A | 0 if pulse wins |
| **Best for** | Known repeating | Single transits | Unknown signal type |

---

#### `calc_eph(p, freqs, inputs, time)`

Calculate normalized BLS periodogram with advanced baseline and noise modeling.

**Parameters:**

- `p` (np.ndarray): Raw BLS power array
- `freqs` (np.ndarray): Frequency grid (c/d)
- `inputs` (gbls_inputs_class): Configuration
- `time` (np.ndarray): Time array for extrapolation

**Returns:**

- `power` (np.ndarray): Normalized BLS power spectrum

**Features:**

- **Duty-cycle informed filtering**: Adapts kernel width based on expected transit duration
- **Baseline extrapolation**: For periods > 0.3×T_baseline, extrapolates rolling median baseline from well-sampled regions to prevent long-period suppression
- **Noise extrapolation**: Maintains proper noise floor at low frequencies
- **Multiple normalization modes**: Robust to high-SNR signals and varying coverage

---

#### `readfile(filename)`

Read two-column light curve files (time, flux).

**Parameters:**

- `filename` (str): Path to ASCII file

**Returns:**

- `time` (np.ndarray): Time array
- `flux` (np.ndarray): Flux array

**File Format:**

```
# Header line (ignored)
2459000.1234  0.9998
2459000.1445  1.0002
2459000.1656  0.9995
...
```

---

### Advanced Features

#### Baseline and Noise Extrapolation

For long-period signals (approaching dataset length), the BLS normalization uses extrapolation to prevent suppression:

1. **Threshold Detection**: Identifies low-frequency bins where periods > 0.5 × T_baseline (need at least 2 full periods)
2. **Linear Fit**: Fits baseline and noise vs. log(frequency) in well-sampled region (periods < 0.25 × T_baseline)
3. **Extrapolation**: Applies fitted trend to low-frequency region

This ensures long-period transits maintain proper SNR in the normalized periodogram.

---

## Transit Model Module (`pytfit5.transitmodel`)

### Core Classes

#### `transit_model_class`

Complete transit model solution container.

**Key Attributes:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `npl` | int | Number of planets |
| `t0` | list[float] | Transit center times (days) |
| `per` | list[float] | Orbital periods (days) |
| `bb` | list[float] | Impact parameters |
| `rdr` | list[float] | Radius ratios (Rp/R*) |
| `ecw` | list[float] | e·cos(ω) eccentricity components |
| `esw` | list[float] | e·sin(ω) eccentricity components |
| `nl3` | float | Quadratic limb darkening coefficient u1 |
| `nl4` | float | Quadratic limb darkening coefficient u2 |
| `rho` | float | Stellar density (g/cm³) |
| `zpt` | float | Flux zero point |
| `dil` | float | Dilution factor |

---

### Core Functions

#### `transitModel(sol, time, itime, nintg=41, ntt=-1, tobs=-1, omc=-1)`

Generate transit light curve from model parameters.

**Parameters:**

- `sol` (transit_model_class): Model parameters
- `time` (np.ndarray): Time array (days)
- `itime` (np.ndarray): Integration time per point (days)
- `nintg` (int): Number of integration steps (default: 41)
- `ntt` (int): Number of time points (default: -1, auto-detect)
- `tobs` (float): Observation time (default: -1, auto-detect)
- `omc` (float): O-C correction (default: -1, unused)

**Returns:**

- `model` (np.ndarray): Model flux array

**Example:**

```python
import pytfit5.transitmodel as tm
import numpy as np

# Create model
sol = tm.transit_model_class()
sol.npl = 1
sol.t0 = [2459000.5]
sol.per = [3.5]
sol.bb = [0.3]
sol.rdr = [0.1]  # 10% radius ratio
sol.ecw = [0.0]
sol.esw = [0.0]
sol.nl3 = 0.311  # TESS-like limb darkening
sol.nl4 = 0.270
sol.rho = 1.4  # Solar-like density

# Generate time array
time = np.linspace(2459000.0, 2459010.0, 1000)
itime = np.full_like(time, 30.0/1440.0)  # 30-minute cadence

# Compute model
model = tm.transitModel(sol, time, itime, nintg=41)
```

---

## Synthetic Lightcurves (`pytfit5.synthetic`)

Utilities for generating synthetic transit lightcurves for BLS testing and validation.

### Functions

#### `generate_synthetic_lightcurve(t0, per, time_length, depth, snr, cadence=1/48, seed=None)`

Generate realistic synthetic transit lightcurve with controlled SNR.

**Parameters:**

- `t0` (float): Transit center time (days)
- `per` (float): Orbital period (days)
- `time_length` (float): Total observation duration (days)
- `depth` (float): Transit depth (fractional, e.g., 0.01 = 1%)
- `snr` (float): Desired signal-to-noise ratio (integrated over transit)
- `cadence` (float): Sampling interval (days, default: 1/48 = 30 min)
- `seed` (int, optional): Random seed for reproducibility

**Returns:**

- `time` (np.ndarray): Time array (days)
- `flux` (np.ndarray): Flux array with noise, baseline ~1.0

**SNR Definition:**

$$\text{SNR} = \frac{\sum_{i \in \text{transit}} |1 - f_i|}{\sigma \sqrt{N_{\text{transit}}}}$$

where $\sigma$ is the noise level, computed to achieve the requested SNR.

**Example:**

```python
from pytfit5.synthetic import generate_synthetic_lightcurve

# Generate 60-day lightcurve with 20-day period, SNR=10
# Generate synthetic data
time, flux, sol_injected = generate_synthetic_lightcurve(
    t0=5.0,
    per=20.05,
    time_length=60.0,
    depth=0.01,  # 1% transit
    snr=10.0,
    cadence=1/48,
    seed=42
)

# Run BLS on synthetic data
import pytfit5.gbls as gbls

inputs = gbls.gbls_inputs_class()
inputs.freq1 = 1.0/60.0
inputs.freq2 = 2.0
inputs.normalize = "iterative_baseline"

ans, freqs, power = gbls.bls(time, flux, inputs)
```

---

#### `compare_bls_injection(time, sol_injected, sol_bls, verbose=True)`

Compare BLS detection results with injection parameters and calculate recovery statistics.

**Parameters:**

- `time` (np.ndarray): Array of observation times (days)
- `sol_injected` (transit_model_class): Solution object with injected transit parameters
- `sol_bls` (transit_model_class): Solution object with BLS recovered parameters
- `verbose` (bool): Print comparison results (default: True)

**Returns:**

- `result` (dict): Recovery statistics containing:
  - `overlap_fraction`: Fraction of true in-transit points recovered (0-1)
  - `precision`: Fraction of recovered points that are truly in-transit (0-1)
  - `true_positive`: Number of correctly identified in-transit points
  - `false_positive`: Number of incorrectly identified in-transit points
  - `false_negative`: Number of missed in-transit points
  - `period_factor`: Best matching period factor (1 = exact, 2 = 2× alias, etc.)
  - `period_factor_type`: 'exact', 'multiple', 'fraction', or 'mismatch'
  - `is_recovered`: Boolean indicating successful recovery (overlap > 50% and precision > 50%)

**Features:**

- Automatically tests for period aliases (integer multiples and fractions up to 5×)
- Handles phase-shifted detections
- Calculates overlap between true and recovered transit windows
- Provides detailed recovery statistics

**Example:**

```python
from pytfit5.synthetic import generate_synthetic_lightcurve, compare_bls_injection
import pytfit5.gbls as gbls
import pytfit5.transitmodel as transitm

# Generate synthetic data (returns solution object)
time, flux, sol_injected = generate_synthetic_lightcurve(
    t0=5.0, per=10.5, time_length=60.0, depth=0.01, snr=15.0, seed=42
)

# Run BLS
inputs = gbls.gbls_inputs_class()
inputs.freq1 = 0.05
inputs.freq2 = 2.0
inputs.normalize = "iterative_baseline"
ans = gbls.bls(inputs, time, flux)

# Create solution object from BLS results
sol_bls = transitm.transit_model_class()
sol_bls.npl = 1
sol_bls.t0 = [ans.epo]
sol_bls.per = [ans.bper]
sol_bls.bb = [0.5]  # Assumed impact parameter
sol_bls.rdr = [np.sqrt(ans.depth)]
sol_bls.rho = 1.4  # Assumed stellar density (g/cm³)

# Compare results using solution objects
recovery = compare_bls_injection(
    time=time,
    sol_injected=sol_injected,
    sol_bls=sol_bls,
    verbose=True
)

print(f"Recovery success: {recovery['is_recovered']}")
print(f"Overlap: {recovery['overlap_fraction']:.1%}")
```

---

#### `calculate_transit_overlap(time, true_t0, true_per, true_duration, recovered_t0, recovered_per, recovered_duration, max_period_factor=5)`

Lower-level function to calculate overlap between true and recovered transit windows.

**Parameters:**

- `time` (np.ndarray): Observation times
- `true_t0`, `true_per`, `true_duration` (float): Injected transit parameters
- `recovered_t0`, `recovered_per`, `recovered_duration` (float): BLS results
- `max_period_factor` (int): Maximum integer factor for period aliases (default: 5)

**Returns:**

- `result` (dict): Same statistics as `compare_bls_injection` but without verbose output

---

#### `mark_in_transit(time, t0, per, duration)`

Mark which observations fall within transit windows.

**Parameters:**

- `time` (np.ndarray): Observation times (days)
- `t0` (float): Transit center time (days)
- `per` (float): Orbital period (days)
- `duration` (float): Transit duration (days)

**Returns:**

- `in_transit` (np.ndarray): Boolean array marking in-transit observations

**Example:**

```python
from pytfit5.synthetic import mark_in_transit

in_transit = mark_in_transit(time, t0=5.0, per=10.5, duration=0.1)
print(f"In-transit points: {np.sum(in_transit)}")
```

---

## Period Validation Module (`pytfit5.period_validation`)

Tools for validating BLS-detected periods by checking for aliases and single-transit events.

### Functions

#### `validate_bls_period(time, flux, bls_result, test_factors=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0], optimize_t0=True, verbose=False)`

Validate a BLS-detected period by testing for period aliases and single-transit events.

**Parameters:**

- `time` (np.ndarray): Observation times (days)
- `flux` (np.ndarray): Flux values (normalized around 0)
- `bls_result` (gbls_ans_class): BLS detection result object
- `test_factors` (list): Period factors to test (default: [0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
- `optimize_t0` (bool): Optimize epoch for each period hypothesis (default: True)
- `verbose` (bool): Print detailed validation report (default: False)

**Returns:**

- `results` (dict): Validation results containing:
  - `best_period` (float): Best validated period
  - `best_t0` (float): Optimized epoch for best period
  - `best_factor` (float): Period factor of best match
  - `best_snr` (float): SNR at best period
  - `best_n_transits` (int): Number of transits at best period
  - `is_alias` (bool): True if best period differs from BLS period
  - `is_single_transit` (bool): True if only one transit detected
  - `snr_improvement` (float): SNR ratio (best/original)
  - `all_results` (list): Results for all tested factors
  
- `transit_info` (dict): Individual transit information:
  - `transit_times` (np.ndarray): Time of each detected transit
  - `transit_snrs` (np.ndarray): SNR of each individual transit
  - `transit_depths` (np.ndarray): Depth of each individual transit

**Example:**

```python
from pytfit5 import period_validation as pval
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5

# Run BLS
inputs = tpy5.tpy5_inputs_class()
inputs.freq1 = 0.05
inputs.freq2 = 2.0
inputs.normalize = "iterative_baseline"

bls_ans = gbls.bls(inputs, time, flux)

# Validate the period
results, transit_info = pval.validate_bls_period(
    time=time,
    flux=flux,
    bls_result=bls_ans,
    test_factors=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    optimize_t0=True,
    verbose=True
)

# Handle validation results
if results['is_alias']:
    print(f"Period alias detected! True period: {results['best_period']:.6f} days")
    bls_ans.bper = results['best_period']
    bls_ans.epo = results['best_t0']
    
elif results['is_single_transit']:
    print(f"Single transit detected at {transit_info['transit_times'][0]:.6f} days")
    # Use highest SNR transit as reference
    max_snr_idx = np.argmax(transit_info['transit_snrs'])
    bls_ans.epo = transit_info['transit_times'][max_snr_idx]
    # Estimate period using stellar density
    P_est, P_min, _, _ = pval.estimate_single_transit_period(
        time=time, tdur=bls_ans.tdur,
        rho_star=pval.get_stellar_density(inputs.mstar, inputs.rstar)
    )
    bls_ans.bper = P_est
    
else:
    print(f"Period confirmed: {bls_ans.bper:.6f} days")
    bls_ans.epo = results['best_t0']  # Use optimized epoch
```

---

#### `estimate_single_transit_period(time, tdur, rho_star, margin_factor=1.5)`

Estimate the orbital period for a single-transit event using stellar density constraints.

**Parameters:**

- `time` (np.ndarray): Observation times (days)
- `tdur` (float): Transit duration (days)
- `rho_star` (float): Stellar density (g/cm³)
- `margin_factor` (float): Safety margin for period estimate (default: 1.5)

**Returns:**

- `P_estimate` (float): Estimated period (days)
- `P_min` (float): Minimum period constrained by data (days)
- `is_constrained` (bool): Whether estimate is constrained by stellar density
- `expected_transits` (float): Expected number of transits at estimated period

**Formula:**

For a circular orbit with transit duration $\tau$:

$$P_{\text{min}} \approx \left(\frac{3\pi}{G\rho_\star}\right)^{1/2} \tau^{3/2}$$

The code returns `P_estimate = margin_factor × P_min` to account for non-zero impact parameters.

**Example:**

```python
from pytfit5 import period_validation as pval

# For a single-transit event
P_est, P_min, is_constrained, n_transits = pval.estimate_single_transit_period(
    time=time,
    tdur=0.12,  # 2.88 hours
    rho_star=1.4,  # Solar-like density
    margin_factor=1.5
)

print(f"Estimated period: {P_est:.2f} days")
print(f"Minimum period: {P_min:.2f} days")
print(f"Expected transits in dataset: {n_transits:.2f}")
```

---

#### `compute_transit_snr(time, flux, t0, period, tdur, n_dur=1.5)`

Compute SNR for a specific period and epoch hypothesis.

**Parameters:**

- `time` (np.ndarray): Observation times (days)
- `flux` (np.ndarray): Flux values (normalized around 0)
- `t0` (float): Transit epoch (days)
- `period` (float): Orbital period (days)
- `tdur` (float): Transit duration (days)
- `n_dur` (float): Number of transit durations for in-transit window (default: 1.5)

**Returns:**

- `snr` (float): Signal-to-noise ratio
- `depth` (float): Transit depth (fractional)
- `n_transits` (int): Number of transits detected
- `in_transit_mask` (np.ndarray): Boolean mask of in-transit points

---

#### `get_stellar_density(mass_solar, radius_solar)`

Calculate stellar density from mass and radius.

**Parameters:**

- `mass_solar` (float): Stellar mass (solar masses)
- `radius_solar` (float): Stellar radius (solar radii)

**Returns:**

- `rho` (float): Mean stellar density (g/cm³)

---

## Keplerian Utilities (`pytfit5.kep`)

Orbital mechanics and stellar parameter utilities.

### Functions

#### `rhostar(per, duration)`

Calculate stellar density from period and transit duration.

**Parameters:**

- `per` (float): Orbital period (days)
- `duration` (float): Transit duration (days)

**Returns:**

- `rho` (float): Stellar density (g/cm³)

**Formula:**

$$\rho_\star = \frac{3\pi}{G P^2} \left(\frac{\tau}{P}\right)^{3}$$

---

## MCMC Module (`pytfit5.tmcmc`)

Markov Chain Monte Carlo transit fitting for parameter estimation and uncertainty quantification.

### Functions

#### `run_mcmc(time, flux, ferr, initial_params, ...)`

Run MCMC analysis on transit lightcurve.

**Parameters:**

- `time` (np.ndarray): Time array
- `flux` (np.ndarray): Flux array
- `ferr` (np.ndarray): Flux uncertainties
- `initial_params` (dict): Starting parameter values

**Returns:**

- `samples` (np.ndarray): MCMC chain samples
- `best_fit` (dict): Maximum likelihood parameters

---

## Saving and Loading MCMC Results (`pytfit5.tpy5`)

Save and load complete MCMC analysis results to HDF5 files for reproducibility and later analysis.

### Functions

#### `save_mcmc_results(filename, phot, sol, chain, accept, burnin, params_to_fit, x=None, beta=None, serr=None, params=None)`

Save MCMC results and all necessary data to an HDF5 file.

**Parameters:**

- `filename` (str): Output HDF5 filename
- `phot` (phot_class): Photometry data object
- `sol` (transit_model_class): Transit model solution object
- `chain` (np.ndarray): MCMC chain array
- `accept` (np.ndarray): Acceptance rates array
- `burnin` (int): Burnin period
- `params_to_fit` (list of str): List of parameter names that were fit
- `x` (np.ndarray, optional): Initial parameters for MCMC
- `beta` (np.ndarray, optional): Initial step sizes for MCMC
- `serr` (np.ndarray, optional): Error array for solution parameters
- `params` (list, optional): MCMC configuration parameters

**Saved Data:**

The HDF5 file contains:
- **Photometry group**: time, flux, ferr, itime, tflag, icut, flux_f
- **Solution group**: sol_array, err_array (if available), npl
- **MCMC results group**: chain, accept, burnin, params_to_fit, and optional x, beta, serr, params
- **Metadata**: creation_date, pytfit5_version

**Example:**

```python
import pytfit5.transitPy5 as tpy5
import pytfit5.transitmcmc as tmcmc

# After running MCMC analysis
chain, accept, burnin = tmcmc.demcmcRoutine(x, beta, phot_cut, sol_a, serr, params, lnprob)

# Save all results
tpy5.save_mcmc_results(
    filename='mcmc_results.h5',
    phot=phot_cut,
    sol=sol,
    chain=chain,
    accept=accept,
    burnin=burnin,
    params_to_fit=params_to_fit,
    x=x,
    beta=beta,
    serr=serr,
    params=params
)
```

---

#### `load_mcmc_results(filename)`

Load MCMC results and all necessary data from an HDF5 file.

**Parameters:**

- `filename` (str): Input HDF5 filename

**Returns:**

- `phot` (phot_class): Photometry data object
- `sol` (transit_model_class): Transit model solution object
- `chain` (np.ndarray): MCMC chain array
- `accept` (np.ndarray): Acceptance rates array
- `burnin` (int): Burnin period
- `params_to_fit` (list of str): List of parameter names that were fit
- `metadata` (dict): Dictionary containing optional saved data (x, beta, serr, params)

**Example:**

```python
import pytfit5.transitPy5 as tpy5
import pytfit5.transitmcmc as tmcmc
import pytfit5.transitplot as transitp

# Load saved MCMC results
phot, sol, chain, accept, burnin, params_to_fit, metadata = tpy5.load_mcmc_results('mcmc_results.h5')

# Extract parameters from the chain
sol_mcmc = tmcmc.getParams(chain, burnin, sol, params_to_fit)

# Print results
transitp.printParams(sol_mcmc)

# Plot the transit model
transitp.plotTransit(phot, sol_mcmc, pl_to_plot=1)

# Optional axis controls
transitp.plotTransit(phot, sol_mcmc, pl_to_plot=1, y1=0.997, y2=1.002, xdur_scale=2.0)

# Continue MCMC if needed (using metadata)
if 'x' in metadata and 'beta' in metadata:
    lnprob, x_new, beta_new = tmcmc.genmcmcInput(sol_mcmc, params_to_fit)
    chain2, accept2, burnin2 = tmcmc.demcmcRoutine(
        x_new, beta_new, phot, sol_mcmc.to_array(), 
        sol_mcmc.err_to_array(), metadata['params'], lnprob
    )
```

**Note:** Requires `h5py` package. Install with `pip install h5py`.

---

## Examples

### Example 1: Basic BLS Search with Period Validation

```python
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5
from pytfit5 import period_validation as pval
import numpy as np

# Load data
phot = tpy5.readphot("star_lightcurve.txt")

# Configure BLS using unified inputs class
inputs = tpy5.tpy5_inputs_class()
inputs.freq1 = 0.1  # 10-day maximum period
inputs.freq2 = 4.0  # 0.25-day minimum period
inputs.ofac = 10.0
inputs.mstar = 0.8
inputs.rstar = 0.75
inputs.normalize = "iterative_baseline"
inputs.oneoverf_correction = True
inputs.oneoverf_extrapolate = True

# Search for transits
ans = gbls.bls(inputs, phot.time, phot.flux)

print(f"Detected Period: {ans.bper:.4f} days")
print(f"Transit Epoch: {ans.epo:.4f}")
print(f"SNR: {ans.snr:.1f}")
print(f"Depth: {ans.depth*1e6:.1f} ppm")

# Validate the period (check for aliases and single transits)
results, transit_info = pval.validate_bls_period(
    time=phot.time,
    flux=phot.flux,
    bls_result=ans,
    test_factors=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    optimize_t0=True,
    verbose=True
)

# Update BLS results based on validation
if results['is_alias']:
    print(f"\nPeriod alias detected! Updating to {results['best_period']:.4f} days")
    ans.bper = results['best_period']
    ans.epo = results['best_t0']
elif results['is_single_transit']:
    # Use highest SNR transit as reference
    max_snr_idx = np.argmax(transit_info['transit_snrs'])
    ans.epo = transit_info['transit_times'][max_snr_idx]
    print(f"\nSingle transit at {ans.epo:.4f} days")
else:
    ans.epo = results['best_t0']  # Use optimized epoch
    print(f"\nPeriod confirmed: {ans.bper:.4f} days")
```

### Example 2: High-SNR Synthetic Test

```python
from pytfit5.synthetic import generate_synthetic_lightcurve
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5
import matplotlib.pyplot as plt

# Generate high-SNR synthetic data
phot, sol_injected = generate_synthetic_lightcurve(
    t0=2.5,
    per=8.3,
    time_length=100.0,
    depth=0.02,  # 2% deep transit
    snr=200.0,   # Very high SNR
    seed=123
)

# BLS with iterative baseline (robust to high SNR)
inputs = tpy5.tpy5_inputs_class()
inputs.freq1 = 0.01
inputs.freq2 = 2.0
inputs.normalize = "iterative_baseline"
inputs.oneoverf_correction = True

ans = gbls.bls(inputs, phot.time, phot.flux)

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(time, flux, 'k.', ms=2)
plt.xlabel('Time (days)')
plt.ylabel('Flux')

plt.subplot(122)
plt.plot(1.0/freqs, power, 'b-', lw=0.5)
plt.axvline(ans.bper, color='r', ls='--', label=f'Detected: {ans.bper:.2f} d')
plt.axvline(8.3, color='g', ls='--', label='True: 8.30 d')
plt.xlabel('Period (days)')
plt.ylabel('BLS Power')
plt.legend()
plt.tight_layout()
plt.show()
```

### Example 3: Long-Period Search with Extrapolation

```python
from pytfit5.synthetic import generate_synthetic_lightcurve
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5

# Generate long-period transit in short baseline
phot, sol_injected = generate_synthetic_lightcurve(
    t0=5.0,
    per=45.0,  # Period close to baseline length
    time_length=60.0,
    depth=0.005,
    snr=15.0,
    seed=99
)

# BLS will extrapolate baseline for periods > 0.5*60 = 30 days
inputs = tpy5.tpy5_inputs_class()
inputs.freq1 = 1.0/60.0  # Search up to 60-day periods
inputs.freq2 = 1.0
inputs.normalize = "iterative_baseline"
inputs.oneoverf_correction = True    # Enable baseline correction
inputs.oneoverf_extrapolate = True   # Enable long-period extrapolation

ans = gbls.bls(inputs, phot.time, phot.flux)

print(f"True period: 45.0 days")
print(f"Recovered period: {ans.bper:.2f} days")
print(f"SNR: {ans.snr:.1f}")
print(f"\nExtrapolation active for periods > {0.5*60:.1f} days")
```

### Example 4: Single-Transit Event Handling

```python
from pytfit5.synthetic import generate_synthetic_lightcurve
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5
from pytfit5 import period_validation as pval
import numpy as np

# Generate very long-period transit (only 1 transit in 60-day baseline)
phot, sol_injected = generate_synthetic_lightcurve(
    t0=10.0,
    per=85.0,  # Much longer than baseline
    time_length=60.0,
    depth=0.008,
    snr=12.0,
    seed=456
)

# Configure BLS
inputs = tpy5.tpy5_inputs_class()
inputs.freq1 = 0.5 / 60.0  # Search to 2× baseline
inputs.freq2 = 2.0
inputs.mstar = 1.0
inputs.rstar = 1.0
inputs.normalize = "iterative_baseline"
inputs.oneoverf_correction = True
inputs.oneoverf_extrapolate = True

# Run BLS
ans = gbls.bls(inputs, phot.time, phot.flux)

print(f"BLS detected period: {ans.bper:.2f} days")
print(f"BLS SNR: {ans.snr:.1f}")

# Validate - should detect single transit
results, transit_info = pval.validate_bls_period(
    time=phot.time,
    flux=phot.flux,
    bls_result=ans,
    test_factors=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    optimize_t0=True,
    verbose=True
)

if results['is_single_transit']:
    print(f"\n✓ Single transit detected!")
    
    # Find the highest SNR transit (most reliable)
    max_snr_idx = np.argmax(transit_info['transit_snrs'])
    single_transit_time = transit_info['transit_times'][max_snr_idx]
    single_transit_snr = transit_info['transit_snrs'][max_snr_idx]
    
    print(f"  Transit time: {single_transit_time:.4f} days")
    print(f"  Transit SNR: {single_transit_snr:.2f}")
    
    # Estimate period using stellar density
    P_est, P_min, is_constrained, n_transits = pval.estimate_single_transit_period(
        time=phot.time,
        tdur=ans.tdur,
        rho_star=pval.get_stellar_density(inputs.mstar, inputs.rstar),
        margin_factor=1.5
    )
    
    print(f"\n  Estimated period: {P_est:.2f} days")
    print(f"  Minimum period: {P_min:.2f} days")
    print(f"  Expected transits: {n_transits:.2f}")
    print(f"  True period: {sol_injected.per[0]:.2f} days")
    
    # Update BLS results
    ans.bper = P_est
    ans.epo = single_transit_time  # Use highest SNR transit time
    ans.snr = single_transit_snr
```

---

## Physical Constants

The following constants are defined in `pytfit5.gbls`:

| Constant | Value | Units | Description |
|----------|-------|-------|-------------|
| `G` | 6.674×10⁻¹¹ | N·m²/kg² | Gravitational constant |
| `Rsun` | 6.96265×10⁸ | m | Solar radius |
| `Msun` | 1.9891×10³⁰ | kg | Solar mass |
| `pifac` | 1.083852 | - | (2π)^(2/3)/π |
| `day2sec` | 86400 | s/day | Seconds per day |
| `onehour` | 1/24 | day/hr | Days per hour |

---

## Performance Notes

- **Numba JIT**: Core BLS kernel functions are JIT-compiled for ~100× speedup
- **Multiprocessing**: Set `inputs.multipro = 1` for parallel processing
- **Frequency Grid**: Higher `ofac` values improve frequency resolution but increase computation time
- **Normalization**: `"iterative_baseline"` mode adds ~10-20% overhead but significantly improves robustness

---

## License

GNU General Public License v3.0 (GPL-3.0)

---

## Citation

If you use this software in your research, please cite:

```
Rowe, J. F. et al. (2025). pytfit5: Python Implementation of TransitFit5 and BLS.
https://github.com/jasonfrowe/bls_cuda
```

---

## Support

- **Issues**: https://github.com/jasonfrowe/bls_cuda/issues
- **Documentation**: See `pytfit5_example.ipynb` for interactive examples
- **Contact**: jason@jasonrowe.org

import numpy as np
from numba import jit

G       = 6.674e-11           # N m^2 kg^-2  Gravitation constant
Rsun    = 696265.0e0*1000.0e0 # m  radius of Sun
Msun    = 1.9891e30           # kg  mass of Sun
pifac   = 1.083852140278e0    # (2pi)^(2/3)/pi
day2sec = 86400.0             # seconds in a day

def one_over_f(f, alpha, scale):
  """
  A power-law model for 1/f noise.

  Parameters:
  f (numpy.ndarray): The frequency array.
  alpha (float): The spectral exponent.
  scale (float): The scaling factor.

  Returns:
  numpy.ndarray: The modeled noise power.
  """
  return scale / (f ** alpha)

def std_without_outliers(data, sigma=2.0, max_iter=10):
    """Calculates the standard deviation of data after iteratively removing outliers."""
    if len(data) < 4:
        return np.nan
    data = np.copy(data)
    for _ in range(max_iter):
        if len(data) < 2: break
        mean, std = np.median(data), np.std(data)
        inliers_mask = np.abs(data - mean) < sigma * std
        if np.all(inliers_mask): break
        data = data[inliers_mask]
    return np.std(data)

# Handle NaNs and interpolate
def interpolate_std(b_std, b_cen, freq):
    valid = ~np.isnan(b_std)
    b_std_filled = np.interp(b_cen, b_cen[valid], b_std[valid])
    interp_std = np.interp(freq, b_cen, b_std_filled)
    interp_std[interp_std == 0] = np.nanmean(interp_std)
    return interp_std

@jit(nopython=True)
def running_std_with_filter(data, half_window):
    # Calculate the overall standard deviation
    std = np.std(data)
    
    # Initialize an array to store the running standard deviations
    running_std = np.empty(len(data))
    
    # Iterate over each element in the data
    for i in range(len(data)):
        # Define the start and end indices of the window
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        
        # Extract the window and filter values less than 3 * std
        window_data = data[start_idx:end_idx]
        filtered_data = window_data[window_data < 3 * std]
        
        # Calculate the standard deviation of the filtered window
        running_std[i] = np.std(filtered_data)
    
    return running_std
        
def nine_if_even(x):
    """Ensure x is odd, increment by 1 if even."""
    return x if (x % 2 == 1) else (x + 1)

def _estimate_medfilt_width(freqs, time, Mstar, Rstar, max_width_bins, alpha=8, min_width=5):
    """
    Estimate an approximately optimal median-filter kernel width (in bins)
    for whitening the BLS spectrum while preserving transit peaks.

    Rationale:
    - The characteristic frequency width of a transit peak is ~ q / T,
      where q is the duty cycle and T is the total observing baseline.
    - With a local grid spacing Δf, the peak spans about (q/T)/Δf bins.
    - A robust background estimate should use a window several times wider
      than the peak width; alpha≈6–10 works well in practice.

    We use stellar-informed duty cycle q(f) computed as in bls_kernel and
    aggregate over the search band to get a single odd kernel size.
    """
    try:
        # Handle edge case: very small frequency arrays
        if len(freqs) < 3:
            return nine_if_even(min_width)
        
        T = float(np.max(time) - np.min(time))
        if not np.isfinite(T) or T <= 0:
            return nine_if_even(min_width)

        # Duty cycle q(f) ~ (R*/a)/π ≈ const * f^(2/3)
        fsec = freqs / day2sec
        q = pifac * Rstar * Rsun / (G * Mstar * Msun) ** (1.0 / 3.0) * fsec ** (2.0 / 3.0)

        # Characteristic peak width in frequency units
        w_f = q / T

        # Local grid spacing Δf (use gradient to handle non-uniform grids)
        # Use edge_order=1 for robustness with small arrays
        df = np.abs(np.gradient(freqs, edge_order=1))
        df[~np.isfinite(df)] = np.nan
        df[df == 0] = np.nan

        # Peak width in bins
        bins = w_f / df
        bins = bins[np.isfinite(bins) & (bins > 0)]
        if bins.size == 0:
            return nine_if_even(min_width)

        median_bins = float(np.median(bins))
        # Scale by alpha to ensure the window is several peak-widths wide
        width = int(np.ceil(alpha * median_bins))

        # Enforce odd width and practical bounds
        if width < min_width:
            width = min_width
        if width % 2 == 0:
            width += 1
        if width > max_width_bins:
            # keep within available samples and odd
            width = max_width_bins if (max_width_bins % 2 == 1) else max_width_bins - 1
        return max(3, width)
    except Exception:
        # Fallback to a conservative small odd window on any error
        return nine_if_even(9)

def _rolling_percentile(x, k, pct):
    """Rolling percentile with constant window size k, shifted near edges."""
    k = int(k)
    if k < 3:
        return x.copy()
    if k % 2 == 0:
        k += 1
    half = k // 2
    out = np.empty_like(x)
    n = len(x)
    for i in range(n):
        start = max(0, min(i - half, n - k))
        end = start + k
        out[i] = np.percentile(x[start:end], pct)
    return out

def _rolling_median(x, k):
    """Rolling median with constant window size k, shifted near edges."""
    k = int(k)
    if k < 3:
        return x.copy()
    if k % 2 == 0:
        k += 1
    half = k // 2
    out = np.empty_like(x)
    n = len(x)
    for i in range(n):
        start = max(0, min(i - half, n - k))
        end = start + k
        out[i] = np.median(x[start:end])
    return out

def _rolling_mad(residual, k):
    """Compute rolling MAD (median absolute deviation) with constant window width k."""
    n = len(residual)
    if k > n:
        k = n
    half = k // 2
    mad = np.zeros(n)
    for i in range(n):
        start = max(0, min(i - half, n - k))
        end = start + k
        window = residual[start:end]
        med = np.median(window)
        mad[i] = np.median(np.abs(window - med))
    return mad

def _iterative_baseline(sqrtp, width, sigma_thresh=3.0, max_iter=3):
    """
    Estimate the continuum baseline using iterative sigma-clipping.
    
    This method is robust to strong positive outliers (transit peaks) by:
    1. Computing a rolling median baseline
    2. Identifying points significantly above the baseline (peaks)
    3. Excluding peaks and re-computing the baseline
    4. Iterating until convergence
    
    Parameters:
    - sqrtp: sqrt(BLS power) spectrum
    - width: rolling window width
    - sigma_thresh: threshold for clipping peaks (default 3.0)
    - max_iter: maximum iterations (default 3)
    
    Returns:
    - baseline: robust estimate of underlying continuum
    - mask: boolean mask of non-peak points used for final baseline
    """
    n = len(sqrtp)
    mask = np.ones(n, dtype=bool)  # Start with all points
    
    for iteration in range(max_iter):
        # Compute baseline using only non-masked points
        masked_sqrtp = sqrtp.copy()
        masked_sqrtp[~mask] = np.nan
        
        # Rolling median ignoring masked points
        baseline = np.zeros(n)
        half = width // 2
        for i in range(n):
            start = max(0, min(i - half, n - width))
            end = start + width
            window = masked_sqrtp[start:end]
            valid = window[np.isfinite(window)]
            if len(valid) > 0:
                baseline[i] = np.median(valid)
            else:
                baseline[i] = np.nan
        
        # Fill any NaN baselines with global median of valid points
        valid_baseline = baseline[np.isfinite(baseline)]
        if len(valid_baseline) > 0:
            baseline[~np.isfinite(baseline)] = np.median(valid_baseline)
        else:
            baseline[:] = np.nanmedian(sqrtp)
        
        # Compute residuals and noise estimate
        residual = sqrtp - baseline
        
        # Robust noise estimate using only current non-masked points
        valid_resid = residual[mask]
        if len(valid_resid) > 3:
            # Use MAD for robust noise estimate
            mad = np.median(np.abs(valid_resid - np.median(valid_resid)))
            noise = 1.4826 * mad
        else:
            noise = np.nanstd(residual)
        
        if noise <= 0 or not np.isfinite(noise):
            noise = 1.0
        
        # Update mask: exclude points significantly above baseline
        # (peaks are positive residuals > sigma_thresh * noise)
        new_mask = residual < (sigma_thresh * noise)
        
        # Check for convergence
        if np.array_equal(new_mask, mask):
            break
        
        mask = new_mask
    
    return baseline, mask

def _extrapolate_baseline(baseline, freqs, time, width, threshold=None):
    """
    Baseline extrapolation for long periods beyond the data baseline.
    
    For periods longer than the threshold (default: baseline length), 
    extrapolate baseline using statistics from the last 1/3 of the baseline 
    (well-sampled region) to avoid suppressing long-period signals.
    
    Parameters:
    - baseline: current baseline estimate from rolling median
    - freqs: frequency array (cycles/day)
    - time: time array (days)
    - width: rolling window width (not used, kept for compatibility)
    - threshold: period threshold in days. If None, uses baseline length.
    
    Returns:
    - baseline_corrected: baseline with extrapolated values at long periods
    """
    T_baseline = ( np.max(time) - np.min(time) ) 
    
    # Use provided threshold or default to baseline length
    if threshold is None:
        threshold = T_baseline 
    
    # Threshold: periods > threshold need extrapolation
    # In frequency space: freqs < 1/threshold
    freq_thresh = 1.0 / threshold
    
    long_period_mask = freqs < freq_thresh
    
    if not np.any(long_period_mask):
        # No long periods to extrapolate
        return baseline
    
    # Reference region: last 1/3 of baseline
    # For 60-day baseline: use 40-60 days (periods in that range)
    # In frequency space: 1/60 to 1/40 c/d
    freq_ref_min = 1.0 / T_baseline  # longest well-sampled period
    freq_ref_max = 1.0 / (T_baseline * 2.0/3.0)  # 2/3 of baseline
    reference_mask = (freqs >= freq_ref_min) & (freqs <= freq_ref_max)
    
    if not np.any(reference_mask):
        # Fallback: use all non-long-period data
        reference_mask = ~long_period_mask
    
    # Calculate median baseline from reference region
    reference_values = baseline[reference_mask]
    valid_mask = (reference_values > 1e-10) & np.isfinite(reference_values)
    
    if not np.any(valid_mask):
        # No valid values, return unchanged
        return baseline
    
    # Use median of reference region as extrapolated value
    extrapolated_baseline = np.median(reference_values[valid_mask])
    
    # Apply extrapolated baseline to long-period region
    baseline_corrected = baseline.copy()
    baseline_corrected[long_period_mask] = extrapolated_baseline
    
    return baseline_corrected

def _extrapolate_noise(noise, freqs, time, width, threshold=None):
    """
    Noise extrapolation for long periods beyond the data baseline.
    
    For periods longer than the threshold (default: baseline length), 
    extrapolate noise using statistics from the last 1/3 of the baseline 
    (well-sampled region) to avoid suppressing long-period signals.
    
    Parameters:
    - noise: current noise estimate from rolling median
    - freqs: frequency array (cycles/day)
    - time: time array (days)
    - width: rolling window width (not used, kept for compatibility)
    - threshold: period threshold in days. If None, uses baseline length.
    
    Returns:
    - noise_corrected: noise with extrapolated values at long periods
    """
    T_baseline = ( np.max(time) - np.min(time) ) 
    
    # Use provided threshold or default to baseline length / 2
    if threshold is None:
        threshold = T_baseline
    
    # Threshold: periods > threshold need extrapolation
    # In frequency space: freqs < 1/threshold
    freq_thresh = 1.0 / threshold
    
    long_period_mask = freqs < freq_thresh
    
    if not np.any(long_period_mask):
        # No long periods to extrapolate
        return noise
    
    # Reference region: last 1/3 of baseline
    # For 60-day baseline: use 40-60 days (periods in that range)
    # In frequency space: 1/60 to 1/40 c/d
    freq_ref_min = 1.0 / T_baseline  # longest well-sampled period
    freq_ref_max = 1.0 / (T_baseline * 2.0/3.0)  # 2/3 of baseline
    reference_mask = (freqs >= freq_ref_min) & (freqs <= freq_ref_max)
    
    if not np.any(reference_mask):
        # Fallback: use all non-long-period data
        reference_mask = ~long_period_mask
    
    # Calculate median noise from reference region
    reference_values = noise[reference_mask]
    valid_mask = (reference_values > 1e-10) & np.isfinite(reference_values)
    
    if not np.any(valid_mask):
        # No valid values, return unchanged
        return noise
    
    # Use median of reference region as extrapolated value
    extrapolated_noise = np.median(reference_values[valid_mask])
    
    # Apply extrapolated noise to long-period region
    noise_corrected = noise.copy()
    noise_corrected[long_period_mask] = extrapolated_noise
    
    return noise_corrected
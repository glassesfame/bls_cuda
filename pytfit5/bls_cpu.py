import os
import sys
from pathlib import Path
funcdir = str((Path(__file__).resolve()).parent)

import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import ticker
import matplotlib.gridspec as gridspec

from tqdm import tqdm

from numba import jit #We will attempt to precompile some routines
from numba import njit, prange

from math import floor

from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy import stats

import concurrent.futures
# Import tpy5_inputs_class from transitPy5
from pytfit5.transitPy5 import tpy5_inputs_class
from pytfit5 import period_validation as pval

# Importing tls functionality
sys.path.insert(0, f'{funcdir}')
from transitleastsquares import main as tlsm

# Backward compatibility alias (deprecated)
gbls_inputs_class = tpy5_inputs_class

#Constants
G       = 6.674e-11           # N m^2 kg^-2  Gravitation constant
Rsun    = 696265.0e0*1000.0e0 # m  radius of Sun
Msun    = 1.9891e30           # kg  mass of Sun
pifac   = 1.083852140278e0    # (2pi)^(2/3)/pi
day2sec = 86400.0             # seconds in a day
onehour = 1.0/24.0            # one hour convected to day.
kepfac = (2*np.pi)**2/G       # used below in the tls calculation

class gbls_ans_class:
    def __init__(self):
        self.epo      = 0.0
        self.bper     = 0.0
        self.rawper   = 0.0
        self.bpower   = -1
        self.snr      = -1
        self.tdur     = 8.0
        self.depth    = 1.0
        self.periods  = None  # Full period array (if return_spectrum=True)
        self.power    = None  # Full power array (if return_spectrum=True)
        self.freqs    = None  # Full frequency array (if return_spectrum=True)
        self.SR       = None # The signal residuals which apply for TLS

#generic routine to read in files
def readfile(filename):

    itime1 = 1765.5/86400.0 #Kepler integration time
    
    i=0
    #data=[] #Create an empty list
    time  = []
    flux  = []
    # ferr  = []
    # itime = []
    f=open(filename)
    for line in f:
        if i>0:
            line = line.strip() #removes line breaks 
            #columns = line.split() #break into columns based on a delimiter.  This example is for spaces
            columns = line.split() #if your data is seperated by commas, use this. 
            #data.append([float(r) for r in columns]) #Append 
            time.append(float(columns[0]))
            flux.append(float(columns[1]))
            # ferr.append(float(columns[2]))
            # itime.append(itime1)
        i+=1
    f.close()
    
    #data = np.array(data)
    time = np.array(time)
    flux = np.array(flux)
    # ferr = np.array(ferr)
    # itime = np.array(itime)
    
    return time, flux# , ferr, itime

@jit(nopython=True)
def nyquest_cpu(time):

    dt = np.zeros((time.shape[0]))
    
    for i in range(time.shape[0]-1):
        dt[i] = time[i+1] - time[i]

    mode = np.median(dt)
    nyq=1.0/(2.0*mode)

    return nyq

@jit(nopython=True)
def qfunc_cpu(f, Mstar, Rstar):
    '''Calculates optimal phase for circular orbit
    '''

    fsec = f/day2sec #Convert from c/d to c/sec 

    q = pifac * Rstar * Rsun / (G * Mstar * Msun)**(1.0/3.0) * fsec**(2.0/3.0)

    return q

@jit(nopython=True)
def calc_nsteps_cpu(Mstar, Rstar, freq1, freq2, nyq, ofac, npt, df0, nb, minbin):
    '''pre-calculate the number of steps to sweep the requested frequency range using optimum steping
    '''

    #Initialize counters
    f = freq1
    nstep = 0

    while(f<freq2):
        nstep+=1
        q = qfunc_cpu(f, Mstar, Rstar) #calculate shift to optimally sample based on expected transit duration
        df = q * df0 #estimate next frequency step based on optimum q
        f = f + df   #next frequency to scan

    return nstep

@jit(nopython=True)
def calc_freqs_cpu(nstep, Mstar, Rstar, freq1, freq2, nyq, ofac, npt, df0, nb, minbin):
    '''pre-calculate the number of steps to sweep the requested frequency range using optimum steping
    '''

    #Initialize arrays
    freqs = np.zeros((nstep))
    qs    = np.zeros((nstep))
    kmis  = np.zeros((3, nstep), dtype = np.int32) # kmi = 0, kma = 1, kkmi = 2
    
    #Initialize counters
    f = freq1
    i = 0
    # freqs[0] = f
    # qs[0]    = qfunc_cpu(f, Mstar, Rstar)

    while(f<freq2):
        qs[i] = qfunc_cpu(f, Mstar, Rstar) #calculate shift to optimally sample based on expected transit duration
        df = qs[i] * df0 #estimate next frequency step based on optimum q
        freqs[i] = f
        
        i += 1 #increase counter
        f = f + df   #next frequency to scan
    return freqs

@jit(nopython=True)
def bls_stats(epo, period, tdur, npt, time, flux):
    # Import necessary libraries if not already imported (e.g., NumPy)

    f   = np.zeros((npt))
    fin = np.zeros((npt))

    tdurfac = 0.5
    
    ph1 = 0.75 - tdurfac * tdur / period
    if ph1 < 0.5:
        ph1 = 0.5
    
    ph2 = 0.75 + tdurfac * tdur / period
    if ph2 > 1.0:
        ph2 = 1.0
    
    toff = 0.75 - (epo / period - int(epo /period))
    if toff < 0.0:
        toff += 1
    
    k = 0  # number of data points out of transit
    kin = 0  # number of data points in transit

    for i in range(npt):
        phase = time[i] / period - int(time[i] / period) + toff
        
        if phase < 0.0:
            phase += 1.0
        elif phase > 1.0:
            phase -= 1.0
        
        if (phase < ph1) or (phase > ph2):
            k += 1
            # Update the out-of-transit flux array with this value
            f[k] = flux[i]
        else:
            kin += 1
            fin[kin] = flux[i]
    
    if k > 3:
        # Calculate standard deviation of out-of-transit data
        std = np.std(f[:k])
        fmean = np.mean(f[:k])
    else:
        fmean = 0.0
        std = 0.0
    
    if kin > 1:
        # Sort in-transit flux array and get median
        sorted_fin = np.sort(fin[:kin])
        depth = -sorted_fin[kin // 2] + fmean
    else:
        depth = fmean
    
    snr = 0.0

    if k > 3 and kin > 1 and std > 0.0:
        snr = depth / std * np.sqrt(kin)
    
    # Output statements can be handled as needed
    
    return fmean, std, depth, snr

def makeplot(periods, power, time, flux, mintime, Keptime, epo, bper, bpower, snr, tdur, depth, filename, plots):

    rcParams.update({'font.size': 12}) #adjust font
    rcParams['axes.linewidth'] = 2.0
    rcParams['font.family'] = 'monospace'

    # Create a figure and three subplots
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

    # Create subplots
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    
    # Split the last row into two columns
    gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], width_ratios=[1, 1.5])
    
    ax4a = plt.subplot(gs4[0])
    ax4b = plt.subplot(gs4[1])
    

    for i in range(6):

        if i == 0:
            # Top panel : BLS spectrum
            xlabel = "Period (days)"
            ylabel = "Power"
        
            x = periods
            y = power
            
            xmin = np.min(x)
            xmax = np.max(x)
        
            ymin = 0.0 #np.min(y)
            ymax = np.max(y)
            ymax += 0.05*(ymax-ymin)
        
            xlog = 1
            ylog = 0
        
            pstyle = 1

            axn = ax1
            
        elif i == 1:
            # 2nd panel : Time series with transit markers
            xlabel = "Time (days)"
            ylabel = "Relative Flux"
        
            x = time+mintime-Keptime
            y = flux + 1
            
            xmin = np.min(x)
            xmax = np.max(x)
        
            ymin = np.min(y)
            ymax = np.max(y)
            dy = (ymax-ymin)
            ymax += 0.05 * dy
            ymin -= 0.05 * dy
        
            xlog = 0
            ylog = 0
        
            pstyle = 2
            psize = 1

            axn = ax2
            
            # Calculate and mark predicted transit times
            t_data = time + mintime - Keptime
            t_min = np.min(t_data)
            t_max = np.max(t_data)
            
            # Find all transit times within the data range
            n_transit_start = int(np.floor((t_min - epo) / bper))
            n_transit_end = int(np.ceil((t_max - epo) / bper))
            
            transit_times = []
            for n in range(n_transit_start, n_transit_end + 1):
                t_transit = epo + n * bper
                if t_min <= t_transit <= t_max:
                    transit_times.append(t_transit)

        elif i == 2:
            # 3rd panel : phase plot 

            xlabel = "Phase"
            ylabel = "Relative Flux"
            
            phase = (time+mintime-Keptime-epo)/bper - np.floor((time+mintime-Keptime-epo)/bper)
            phase[phase > 0.25] = phase[phase > 0.25] - 1.0
            # ph1 = 2.0 * tdur/bper

            x = phase
            y = flux + 1

            xmin = np.min(x)
            xmax = np.max(x)
        
            ymin = np.min(y)
            ymax = np.max(y)
            dy = (ymax-ymin)
            ymax += 0.05 * dy
            ymin -= 0.05 * dy

            xlog = 0
            ylog = 0 

            pstyle = 2
            psize = 1

            axn = ax3

        elif i == 3:
            # 4th panel, bottom right
            
            xlabel = "Phase (Hours)"
            ylabel = "Relative Flux"
            
            phase = (time+mintime-Keptime-epo)/bper - np.floor((time+mintime-Keptime-epo)/bper)
            phase[phase > 0.5] = phase[phase > 0.5] - 1.0
            ph1 = 2.0 * tdur/bper

            x = 24*bper*phase[(phase>-ph1)&(phase<ph1)]
            y = flux[(phase>-ph1)&(phase<ph1)]+1

            xmin = np.min(x)
            xmax = np.max(x)
        
            ymin = np.min(y)
            ymax = np.max(y)
            dy = (ymax-ymin)
            ymax += 0.05 * dy
            ymin -= 0.05 * dy

            xlog = 0
            ylog = 0 

            pstyle = 2
            psize = 1

            axn = ax4b

        elif i == 5:
            #Stats panel
            ax4a.set_xticks([])
            ax4a.set_yticks([])
            ax4a.set_frame_on(False)  # Remove the border
            ax4a.set_title("")  # Remove title

            # print("Per = ", bper, "(days)\nEPO = ", epo, "(days)\nPow = ", power, "\nSNR = ", \
            #       snr, "\nTdur = ", tdur*24, "(days)\nTdep = ", depth*1.0e6,"(ppm)")

            ln1 = f"Per  = {bper:.6f} days"
            ln2 = f"T0   = {epo:.6f} days"
            ln3 = f"Pow  = {bpower:.6f}"
            ln4 = f"SNR  = {snr:.1f}"
            ln5 = f"Tdur = {tdur*24:.1f} hours"
            ln6 = f"Tdep = {depth*1.0e6:.1f} ppm"
            lines_of_text = [ln1, ln2, ln3, ln4, ln5, ln6]
            for i, line in enumerate(lines_of_text):
                ax4a.text(0.0, 0.8 - (i/(2+len(lines_of_text))), line, ha='left')

            
    
        axn.tick_params(direction='in', which='major', bottom=True, top=True, \
                           left=True, right=True, length=10, width=2)
        axn.tick_params(direction='in', which='minor', bottom=True, top=True, \
                           left=True, right=True, length=4, width=2)
    
        
        axn.set_xlabel(xlabel)
        axn.set_ylabel(ylabel)
        
        axn.set_xlim(xmin,xmax)
        axn.set_ylim(ymin,ymax)
    
        if xlog == 1:
            axn.set_xscale('log')
        if ylog == 1:
            axn.set_yscale('log')
    
        axn.xaxis.set_major_formatter(ticker.ScalarFormatter())
        axn.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
        if pstyle == 2:
            axn.scatter(x, y, s = psize, color = 'black')
        else:
            axn.plot(x, y, color = 'black')
        
        # Add vertical lines for transit times on time series panel
        if i == 1 and len(transit_times) > 0:
            for t_transit in transit_times:
                axn.axvline(x=t_transit, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    fig.tight_layout()
    if plots == 2:        
        directory = os.path.dirname(filename)
        filename_without_extension, file_extension = os.path.splitext(os.path.basename(filename))

        if is_writable(directory):
            filename_without_extension = ".".join(filename.split(".")[:-1])
            fig.savefig(filename_without_extension+".png", dpi=150)
        else:
            print("Warning: Path to datafile not writable, attempting to make plot in current directory")
            fig.savefig(filename_without_extension+".png", dpi=150)
        plt.show()

    elif plots == 3:
        directory = os.path.dirname(filename)
        filename_without_extension, file_extension = os.path.splitext(os.path.basename(filename))

        if is_writable(directory):
            filename_without_extension = ".".join(filename.split(".")[:-1])
            fig.savefig(filename_without_extension+".png", dpi=150)
        else:
            print("Warning: Path to datafile not writable, attempting to make plot in current directory")
            fig.savefig(filename_without_extension+".png", dpi=150)
        plt.close(fig)

    else:
        plt.show()
        
def is_writable(directory):
    return os.access(directory, os.W_OK)

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


def calc_eph(p, jn1, jn2, npt, time, flux, freqs, ofac, nstep, nb, mintime, Keptime, Mstar, Rstar, normalize_mode="iterative_baseline", oneoverf_correction=False, oneoverf_extrapolate=True, oneoverf_threshold=None):

    periods = 1/freqs # periods (days)
    
    # Estimate a data-driven, duty-cycle informed median filter width.
    # Falls back to a small odd kernel if estimation fails.
    width = _estimate_medfilt_width(freqs, time, Mstar, Rstar, max_width_bins=nstep-1, \
                            alpha=4*ofac, min_width=nine_if_even(int(max(5, ofac*4))))

    sqrtp = np.sqrt(p)
    # Sanitize: replace NaN/Inf with very negative values so they don't dominate sort
    rawpower = np.where(np.isfinite(sqrtp), sqrtp, -1e30)
    rawsort = np.argsort(rawpower)
    rawper = periods[rawsort[-1]] 
    eps = 1e-12
    
    # Handle pure BLS case first (no processing needed)
    if normalize_mode == "none" and not oneoverf_correction:
        # Pure BLS spectrum with no baseline correction or normalization
        power = rawpower
        psort = rawsort
        bper = rawper

    else: # For all other cases, we need to compute baseline and/or normalization
        if normalize_mode == "none" and oneoverf_correction:
            # Baseline correction but no normalization
            # Use iterative baseline to avoid suppressing strong peaks
            baseline, _ = _iterative_baseline(sqrtp, width, sigma_thresh=3.0, max_iter=3)
            # Apply extrapolation if enabled
            if oneoverf_extrapolate:
                baseline = _extrapolate_baseline(baseline, freqs, time, width, threshold=oneoverf_threshold)
            residual = sqrtp - baseline
            power = residual
        elif normalize_mode == "iterative_baseline":
            # Iteratively identify continuum by clipping strong peaks
            # This is more robust for high-SNR signals that would bias standard methods
            if oneoverf_correction:
                # Apply baseline subtraction
                iter_baseline, peak_mask = _iterative_baseline(sqrtp, width, sigma_thresh=3.0, max_iter=3)
                # Extrapolate the iterative baseline for low frequencies (if enabled)
                if oneoverf_extrapolate:
                    iter_baseline = _extrapolate_baseline(iter_baseline, freqs, time, width, threshold=oneoverf_threshold)
                residual_iter = sqrtp - iter_baseline
            else:
                # No baseline subtraction, use raw sqrtp
                residual_iter = sqrtp
                # Still compute peak_mask for noise estimation
                _, peak_mask = _iterative_baseline(sqrtp, width, sigma_thresh=3.0, max_iter=3)
            
            # Compute noise from non-peak regions only
            if np.sum(peak_mask) > 10:
                noise_resid = residual_iter[peak_mask]
                mad_iter = np.median(np.abs(noise_resid - np.median(noise_resid)))
                noise_iter = 1.4826 * mad_iter
            else:
                noise_iter = np.nanstd(residual_iter)
            if not np.isfinite(noise_iter) or noise_iter <= 0:
                noise_iter = np.nanstd(residual_iter)
            # Create uniform noise array (no extrapolation needed - already global constant)
            noise_arr = np.full_like(residual_iter, noise_iter)
            power = residual_iter / (noise_arr + eps)
        elif normalize_mode == "mad":
            # Compute baseline for MAD normalization using iterative method
            if oneoverf_correction:
                baseline, _ = _iterative_baseline(sqrtp, width, sigma_thresh=3.0, max_iter=3)
                if oneoverf_extrapolate:
                    baseline = _extrapolate_baseline(baseline, freqs, time, width, threshold=oneoverf_threshold)
                residual = sqrtp - baseline
            else:
                residual = sqrtp
            # Rolling MAD around the baseline (constant window at edges)
            local_med = _rolling_median(residual, width)
            mad = _rolling_mad(residual - local_med, width)
            noise = 1.4826 * mad
            # Robust fallback: use global median for edge/low values
            global_noise = np.nanmedian(noise[noise > eps])
            if not np.isfinite(global_noise) or global_noise <= 0:
                global_noise = np.nanstd(residual)
            noise = np.maximum(noise, 0.1 * global_noise)  # floor at 10% of typical
            # Extrapolate noise for low frequencies (if enabled)
            if oneoverf_extrapolate:
                noise = _extrapolate_noise(noise, freqs, time, width, threshold=oneoverf_threshold)
            power = residual / (noise + eps)
        elif normalize_mode == "percentile_mad":
            # Use upper-percentile baseline and MAD of residuals (constant window at edges)
            if oneoverf_correction:
                perc_base = _rolling_percentile(sqrtp, width, 75)
                # Extrapolate percentile baseline for low frequencies (if enabled)
                if oneoverf_extrapolate:
                    perc_base = _extrapolate_baseline(perc_base, freqs, time, width, threshold=oneoverf_threshold)
                resid2 = sqrtp - perc_base
            else:
                resid2 = sqrtp
            local_med2 = _rolling_median(resid2, width)
            mad2 = _rolling_mad(resid2 - local_med2, width)
            noise2 = 1.4826 * mad2
            # Robust fallback
            global_noise2 = np.nanmedian(noise2[noise2 > eps])
            if not np.isfinite(global_noise2) or global_noise2 <= 0:
                global_noise2 = np.nanstd(resid2)
            noise2 = np.maximum(noise2, 0.1 * global_noise2)
            # Extrapolate noise for low frequencies (if enabled)
            if oneoverf_extrapolate:
                noise2 = _extrapolate_noise(noise2, freqs, time, width, threshold=oneoverf_threshold)
            power = resid2 / (noise2 + eps)
        else:  # "coverage_mad" (default)
            # Compute baseline for coverage_mad normalization using iterative method
            if oneoverf_correction:
                baseline, _ = _iterative_baseline(sqrtp, width, sigma_thresh=3.0, max_iter=3)
                if oneoverf_extrapolate:
                    baseline = _extrapolate_baseline(baseline, freqs, time, width, threshold=oneoverf_threshold)
                residual = sqrtp - baseline
            else:
                residual = sqrtp
            # Coverage proxy via kkmi(f)
            fsec = freqs / day2sec
            q = pifac * Rstar * Rsun / (G * Mstar * Msun) ** (1.0 / 3.0) * fsec ** (2.0 / 3.0)
            qmi = q / 2.0
            tiny = onehour * freqs / 2.0
            qmi = np.maximum(qmi, tiny)
            kkmi = np.maximum(5, np.floor(npt * qmi)).astype(float)
            coverage = np.clip(kkmi / float(npt), 0.01, 1.0)  # clip to [1%, 100%]
            # Rolling MAD (constant window at edges)
            local_med = _rolling_median(residual, width)
            mad = _rolling_mad(residual - local_med, width)
            noise = 1.4826 * mad
            # Robust fallback
            global_noise = np.nanmedian(noise[noise > eps])
            if not np.isfinite(global_noise) or global_noise <= 0:
                global_noise = np.nanstd(residual)
            noise = np.maximum(noise, 0.1 * global_noise)
            # Extrapolate noise for low frequencies (if enabled)
            if oneoverf_extrapolate:
                noise = _extrapolate_noise(noise, freqs, time, width, threshold=oneoverf_threshold)
            # Weighted normalization with floor
            denom = noise * coverage + eps
            power = residual / (denom + eps)

        # Sanitize: replace NaN/Inf with very negative values so they don't dominate sort
        power = np.where(np.isfinite(power), power, -1e30)
        psort = np.argsort(power) #Get sorted indicies to find best event
        bper   = periods[psort[-1]]   #Period of best event

    bpower = power[psort[-1]]     #Power of best event
    in1   = jn1[psort[-1]]        #start of ingress (based on BLS, so there is will be some error)
    in2   = jn2[psort[-1]]        #end of egress 

    # Calculate duration
    maxt = np.max(time)
    mint = np.min(time)
    duration = maxt - mint

    # Determine scaling factor: 
    # If standard BLS: scale = bper
    # If single event (bper > duration): scale = duration
    if bper > duration:
        scale_per = duration
    else:
        scale_per = bper

    # apply boundary fixes to in1, in2
    # pad in1,in2 by 1 and make sure we don't step outside [0,nb-1]
    # if in2 > nb:
    #     in2 = in2 - nb
    
    # pad in1,in2 by 1 and make sure we don't step outside [0,nb-1]
    # in1 = in1 - 1
    # if in1<0:
    #     in1=in1+nb
    # in2 = in2 + 1
    # if in2 > nb - 1: 
    #     in2 = in2 - nb + 1
    # if in1 == in2: 
    #     in2 = in1 + 1

    if in2 > nb: in2 = in2 - nb
    in1 = in1 - 1
    if in1 < 0: in1 = in1 + nb
    in2 = in2 + 1
    if in2 > nb - 1: in2 = in2 - nb + 1
    if in1 == in2: in2 = in1 + 1

    # Time of first Transit
    # if in1 < in2:
    #     epo = mintime + bper * ( (in1 + in2)/2.0 / nb)
    # else:
    #     epo = mintime + bper * ( (in1 + in2 - nb + 1) / 2.0 / nb )
    # epo = epo - Keptime

    # Time of first Transit
    if in1 < in2:
        epo = mintime + scale_per * ( (in1 + in2)/2.0 / nb)
    else:
        epo = mintime + scale_per * ( (in1 + in2 - nb + 1) / 2.0 / nb )
    epo = epo - Keptime

    # Transit duration.  (units = period = days)
    # if in1 <= in2:
    #     tdur = bper * (in2 - in1) / nb
    # else:
    #     tdur = bper * (in2 + nb - 1 - in1) / nb

    # Transit duration
    if in1 <= in2:
        tdur = scale_per * (in2 - in1) / nb
    else:
        tdur = scale_per * (in2 + nb - 1 - in1) / nb

    #Dirty translation of Fortran from transitfind5
    fmean, std, depth, snr = bls_stats(epo, bper, tdur, npt, time + mintime - Keptime, flux) 

    return periods, power, bper, rawper, epo, bpower, snr, tdur, depth

def bls(tpy5_inputs, time = np.array([0]), flux = np.array([0])):

    if tpy5_inputs.lcdir == "":
        filename = tpy5_inputs.filename
    else:
        filename = tpy5_inputs.lcdir + "/" + tpy5_inputs.filename
    Keptime  = tpy5_inputs.zerotime
    freq1    = tpy5_inputs.freq1
    freq2    = tpy5_inputs.freq2
    ofac     = tpy5_inputs.ofac
    Mstar    = tpy5_inputs.mstar
    Rstar    = tpy5_inputs.rstar
    minbin   = tpy5_inputs.minbin
    nper     = tpy5_inputs.nper
    multipro = tpy5_inputs.multipro
    
    if (time.shape[0] < 2) or (flux.shape[0] < 2):
        time, flux = readfile(filename)
        
    #Simple transform of time and flux arrays
    mintime = np.min(time)
    time = time - mintime #time starts at zero
    flux = flux - np.median(flux)  #flux is centred around 0

    npt   = time.shape[0]                     #number of data points
    nb    = np.min((2000, time.shape[0]-1.0)) #npt - 1, sets how the data in binned in phase.
    nyq   = nyquest_cpu(time)            #nyquest [c/d] -- CPU seems to be faster x2 for 1080ti vs 10700
    mint  = np.min(time)                      #minimum time [d]
    maxt  = np.max(time)                      #max time [d]

    if freq1 <= 0:
        freq1 = 2.0/(maxt-mint)                   #lowest frequency [c/d]
    if freq2 <= 0:
        freq2 = 2.0                               #highest frequency [c/d]
    
    # Validate frequency range
    if freq1 >= freq2:
        raise ValueError(
            f"Invalid frequency range: freq1={freq1:.6f} must be < freq2={freq2:.6f} c/d\n"
            f"Remember: frequency = 1/period, so longer periods = lower frequencies.\n"
            f"For period range [{1/freq2:.2f}, {1/freq1:.2f}] days, use:\n"
            f"  freq1 = {1/(maxt-mint):.6f} c/d (period = {maxt-mint:.2f} days)\n"
            f"  freq2 = {freq2:.6f} c/d (period = {1/freq2:.2f} days)"
        )

    #Compute some quantities 
    steps = int(ofac * (freq2 - freq1) * npt/nyq) #naive number of frequencies to scan
    df0   = (freq2 - freq1) / steps
    
    #Calculate the number of steps needed to scan the frequency range 
    nstep = calc_nsteps_cpu(Mstar, Rstar, freq1, freq2, nyq, ofac, npt, df0, nb, minbin)
    # print("freqs: ", freq1, freq2)
    # print("nstep: ", nstep)
    
    # Validate minimum nstep to ensure meaningful BLS search
    if nstep < 3:
        print(f"Warning: nstep={nstep} is too small for meaningful BLS search.")
        print(f"Frequency range [{freq1:.6f}, {freq2:.6f}] c/d may be too narrow.")
        print(f"Consider adjusting freq1, freq2, or ofac parameters.")
        # Return a dummy result
        gbls_ans = gbls_ans_class()
        gbls_ans.epo = 0.0
        gbls_ans.bper = 1.0 / freq1 if freq1 > 0 else 1.0
        gbls_ans.bpower = 0.0
        gbls_ans.snr = 0.0
        gbls_ans.tdur = 0.1
        gbls_ans.depth = 0.0
        return gbls_ans

    #no need to break up runs
    nper = nstep
    
    #Calculate the frequencies that we will be folding at
    freqs = calc_freqs_cpu(nstep, Mstar, Rstar, freq1, freq2, nyq, ofac, npt, df0, nb, minbin)

    if multipro == 0:
        #Store our answers -- this is for single threads (make this an option!)
        p = np.zeros_like(freqs)
        jn1   = np.zeros_like(freqs, dtype=np.int32)
        jn2   = np.zeros_like(freqs, dtype=np.int32)
        
        # for i in tqdm(range(nstep)):
        #     p[i], jn1[i], jn2[i] = bls_kernel(freqs[i], time, flux, nb, Rstar, Mstar)

        duration = np.max(time) - np.min(time) # Should be ~ max(time) as min is 0

        for i in tqdm(range(nstep)):
            if (1.0 / freqs[i]) > duration:
                 p[i], jn1[i], jn2[i] = bls_kernel_single(freqs[i], time, flux, nb, Rstar, Mstar)
            else:
                 p[i], jn1[i], jn2[i] = bls_kernel(freqs[i], time, flux, nb, Rstar, Mstar)

    else: 
        # Store our answers
        max_processes = os.cpu_count()
        ndiv = nstep // (max_processes - 1) + 1
        
        p   = np.zeros((max_processes, ndiv))
        jn1 = np.zeros((max_processes, ndiv), dtype=np.int32)
        jn2 = np.zeros((max_processes, ndiv), dtype=np.int32)

        iarg = np.zeros((max_processes, ndiv), dtype = np.int32)
        for i in range(0, max_processes):
            for k,j in enumerate(range(i, nstep, max_processes)):
                iarg[i, k] = j
    
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
            futures = {executor.submit(compute_bls_kernel, iarg[i], freqs, time, flux, \
                                       nb, Rstar, Mstar, ndiv): i for i in range(iarg.shape[0])}
            
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    p[i], jn1[i], jn2[i] = result
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
    
        p = p.T.ravel()[0:nstep]
        jn1 = jn1.T.ravel()[0:nstep]
        jn2 = jn2.T.ravel()[0:nstep]
    
    normalize_mode = getattr(tpy5_inputs, "normalize", "coverage_mad")
    oneoverf_correction = getattr(tpy5_inputs, "oneoverf_correction", False)
    oneoverf_extrapolate = getattr(tpy5_inputs, "oneoverf_extrapolate", True)
    oneoverf_threshold = getattr(tpy5_inputs, "oneoverf_threshold", None)
    periods, power, bper, rawper, epo, bpower, snr, tdur, depth = \
        calc_eph(p, jn1, jn2, npt, time, flux, freqs, ofac, nstep, nb, mintime, Keptime, Mstar, Rstar, normalize_mode, oneoverf_correction, oneoverf_extrapolate, oneoverf_threshold)
    if tpy5_inputs.plots > 0:
        makeplot(periods, power, time, flux, mintime, Keptime, epo, bper, bpower, snr, tdur, depth, \
                filename, tpy5_inputs.plots)

    gbls_ans = gbls_ans_class()
    gbls_ans.epo    = epo
    gbls_ans.bper   = bper
    gbls_ans.rawper = rawper
    gbls_ans.bpower = bpower
    gbls_ans.snr    = snr
    gbls_ans.tdur   = tdur
    gbls_ans.depth  = depth
    
    # Optionally store full spectrum
    if tpy5_inputs.return_spectrum:
        gbls_ans.periods = periods
        gbls_ans.power = power
        gbls_ans.freqs = freqs

    return gbls_ans

# def compute_bls_kernel(iarg, freqs, time, flux, nb, Rstar, Mstar, ndiv):
    
#     p   = np.zeros((ndiv))
#     jn1 = np.zeros((ndiv), dtype = np.int32)
#     jn2 = np.zeros((ndiv), dtype = np.int32)

#     j = 0
#     for i in iarg:
#         p[j], jn1[j], jn2[j] = bls_kernel(freqs[i], time, flux, nb, Rstar, Mstar)
#         j += 1
    
#     return p, jn1, jn2

@jit(nopython=True)
def bls_pulse_kernel(flux_binned, counts_binned, widths, sigma):
    """
    Scans for single pulses. Returns the Signal-to-Noise Ratio (SNR).
    Enforces DIPS ONLY (flux sum < 0).
    """
    nb = len(flux_binned)
    n_widths = len(widths)
    
    best_snr = np.zeros(n_widths)
    best_starts = np.zeros(n_widths, dtype=np.int32)
    best_depths = np.zeros(n_widths)
    
    # Precompute Integral Images
    cum_y = np.zeros(nb + 1)
    cum_c = np.zeros(nb + 1, dtype=np.int32)
    
    for i in range(nb):
        cum_y[i+1] = cum_y[i] + flux_binned[i]
        cum_c[i+1] = cum_c[i] + counts_binned[i]
        
    # Iterate over durations
    for w_idx in range(n_widths):
        width = widths[w_idx]
        
        max_snr = 0.0 # Start at 0, ignore positive peaks
        best_idx = -1
        depth = 0.0
        
        # Scan
        for i in range(nb - width + 1):
            s = cum_y[i + width] - cum_y[i] # Sum of flux
            r = cum_c[i + width] - cum_c[i] # Count of points
            
            # Check for valid bin and TRANSIT (s < 0)
            if r > 0 and s < 0:
                # SNR calculation
                # SNR = |Signal_Sum| / (Sigma_per_point * sqrt(Points))
                # This assumes white noise for the normalization
                current_snr = np.abs(s) / (sigma * np.sqrt(r))
                
                if current_snr > max_snr:
                    max_snr = current_snr
                    best_idx = i
                    depth = s / r 

        best_snr[w_idx] = max_snr
        best_starts[w_idx] = best_idx
        best_depths[w_idx] = depth
        
    return best_snr, best_starts, best_depths

def compute_pulse_search(time, flux, min_duration_hours=1.0, max_duration_hours=12.0, bin_duration_days=None):
    
    # 1. Detrend / Zero-Center
    # Crucial step: Remove median to prevent wide boxes from grabbing offsets
    # For very long datasets, consider a high-pass filter or running median here.
    med_flux = np.median(flux)
    flux_norm = flux - med_flux
    
    # Calculate robust sigma (MAD) for SNR scaling
    mad = np.median(np.abs(flux_norm))
    sigma = mad * 1.4826
    
    t_start = time[0]
    duration = time[-1] - t_start
    
    # 2. Binning
    if bin_duration_days is None:
        bin_duration_days = 10.0 / 1440.0 # ~10 mins
        
    nb = int(duration / bin_duration_days) + 1
    
    indices = ((time - t_start) / bin_duration_days).astype(int)
    mask = (indices >= 0) & (indices < nb)
    indices = indices[mask]
    flux_subset = flux_norm[mask] # Use zero-centered flux
    
    c_binned = np.bincount(indices, minlength=nb)
    y_binned = np.bincount(indices, weights=flux_subset, minlength=nb)
    
    # 3. Duration Grid
    min_width = int((min_duration_hours / 24.0) / bin_duration_days)
    max_width = int((max_duration_hours / 24.0) / bin_duration_days)
    if min_width < 1: min_width = 1
    if max_width > nb: max_width = nb
    
    widths = np.arange(min_width, max_width + 1, dtype=np.int32)
    
    if len(widths) == 0:
        return None
        
    # 4. Run Kernel (Pass sigma)
    snrs, starts, depths = bls_pulse_kernel(y_binned, c_binned, widths, sigma)
    
    # 5. Extract Best Result
    best_idx = np.argmax(snrs)
    best_snr = snrs[best_idx]
    
    best_width_bins = widths[best_idx]
    best_start_bin  = starts[best_idx]
    best_depth = depths[best_idx]
    
    best_duration = best_width_bins * bin_duration_days
    best_t0 = t_start + (best_start_bin + 0.5 * best_width_bins) * bin_duration_days
    
    return {
        'snr': best_snr,              # Now in SNR units
        't0': best_t0,
        'duration': best_duration,
        'depth': best_depth,
        'spectrum_duration': widths * bin_duration_days * 24.0,
        'spectrum_snr': snrs
    }



@jit(nopython=True)
def bls_kernel_single(freq, time, flux, nb, Rstar, Mstar):
    """
    BLS kernel optimized for single event searches (Period > Duration).
    """
    # Explicitly cast nb to int for array creation
    nb = int(nb)
    npt = time.shape[0]
    
    # Duration of the dataset (Assuming time starts at 0, or is shifted)
    duration = time[npt-1]
    
    if duration <= 0:
        return 0.0, 0, 0

    nbm1 = int(nb - 1)
    
    # No buffer for wrapping needed
    ibi = np.zeros(nb, dtype=np.int32)
    y = np.zeros(nb, dtype=np.float64)
    
    # Bin the data: Map 0..duration to 0..nb
    # scale factor to convert time to bin index
    scale = nbm1 / duration
    
    for i in range(npt):
        # Map time[i] directly to a bin
        j = int(time[i] * scale)
        if j < nb:
            ibi[j] += 1
            y[j] += flux[i]
            
    # Calculate search parameters based on physics
    fsec = freq / day2sec
    q = pifac * Rstar * Rsun / (G * Mstar * Msun)**(1.0/3.0) * fsec**(2.0/3.0)
    
    # Calculate ratio of Period to Duration: P / T_dur = (1/f) / duration
    # This scales the duty cycle 'q' (fraction of Period) to fraction of the Array (Duration)
    pd_ratio = (1.0 / freq) / duration
    
    # Define limits for transit duration
    qmi = q / 2.0
    if qmi < onehour*freq/2:
        qmi = onehour*freq/2
        
    qma = q * 2.0
    if qma > 0.25:
        qma = 0.25
        
    if qmi > qma:
        qma = qmi * 2.0
        if qma > 0.4:
            qma = 0.4
            
    # Scale q limits (fraction of Period) to bin counts over the Duration
    # bins = q * (Period/Duration) * nb
    kmi = int(qmi * pd_ratio * nb)
    if kmi < 1: kmi = 1
    
    kma = int(qma * pd_ratio * nb) + 1
    if kma > nb: kma = nb
    
    # Minimum points in transit check
    kkmi = int(npt * qmi * pd_ratio)
    if kkmi < 5: kkmi = 5
    
    p = 0.0
    jn1 = 0
    jn2 = 0
    
    # Search loop (Linear scan, no wrapping)
    # i is start bin, j is end bin
    for i in range(nb):
        s = 0.0
        k = 0
        kk = 0
        
        # Determine max end index (clamped to nb)
        jmax = i + kma + 1
        if jmax > nb: jmax = nb
        
        for j in range(i, jmax):
            k += 1          # width in bins
            kk += ibi[j]    # count of points in bin j
            s += y[j]       # sum of flux in bin j
            
            # Check conditions
            if k > kmi and kk > kkmi:
                dfac = kk * (npt - kk)
                if dfac > 0:
                    pow1 = s*s / dfac
                    if pow1 > p:
                        p = pow1
                        jn1 = i
                        jn2 = j
                        
    return p, jn1, jn2

def compute_bls_kernel(iarg, freqs, time, flux, nb, Rstar, Mstar, ndiv):
    
    p = np.zeros((ndiv))
    jn1 = np.zeros((ndiv), dtype = np.int32)
    jn2 = np.zeros((ndiv), dtype = np.int32)
    
    # Calculate duration once
    duration = time[-1] - time[0] 

    j = 0
    for i in iarg:
        # Check if Period > Duration
        if (1.0 / freqs[i]) > duration:
            p[j], jn1[j], jn2[j] = bls_kernel_single(freqs[i], time, flux, nb, Rstar, Mstar)
        else:
            p[j], jn1[j], jn2[j] = bls_kernel(freqs[i], time, flux, nb, Rstar, Mstar)
        j += 1
        
    return p, jn1, jn2

@jit(nopython=True)
def bls_kernel(freq, time, flux, nb, Rstar, Mstar):

    npt = time.shape[0]

    nb2  = int(2 * nb)
    nbm1 = int(nb - 1)
    nb   = int(nb)

    ibi = np.zeros((nb2), dtype=np.int32)
    y   = np.zeros((nb2), dtype=np.float64)
    
    # Initialize return values to safe defaults
    p = 0.0
    jn1 = 0
    jn2 = 0

    for i in range(npt):

        phase = time[i]*freq - int(time[i]*freq)
        j = int(phase * nbm1)

        ibi[j] += 1
        y[j]   += flux[i]

    #duplicate array
    for i in range(nb):
        ibi[i+nb] = ibi[i]
        y[i+nb]   = y[i]

    fsec = freq / day2sec
    q = pifac * Rstar*Rsun / (G*Mstar*Msun)**(1.0/3.0) * fsec**(2.0/3.0)

    qmi = q / 2.0
    if qmi < onehour*freq/2:
        qmi = onehour*freq/2

    qma = q * 2.0
    if qma > 0.25:
        qma = 0.25

    if qmi > qma:
        qma = qmi * 2.0
        if qma > 0.4:
            qma = 0.4

    kmi = int(qmi * nb)
    if kmi < 1:
        kmi = 1
    kma = int(qma * nb) + 1
    kkmi = int(npt * qmi)
    if kkmi < 5: #minbin
        kkmi = 5
    nbkma = nb + kma

    for i in range(nb):
        s = 0.0
        k = 0
        kk = 0
        nb2 = i + kma + 1

        for j in range(i, nb2):
            k  = k + 1
            kk = kk + ibi[j]
            s  = s  + y[j]

            if k > kmi and kk > kkmi:
                dfac = (kk * (npt - kk)) #const[1] = npt
                if dfac > 0 :
                    pow1 = s*s / dfac

                    if pow1 > p:
                        p = pow1
                        jn1 = i
                        jn2 = j        
    
    return p, jn1, jn2

def makeplot_pulse(time, flux, ans, mintime, Keptime, filename, plots):
    """
    Plotting routine for single pulse results.
    """
    rcParams.update({'font.size': 12})
    rcParams['axes.linewidth'] = 2.0
    rcParams['font.family'] = 'monospace'

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # Panel 1: Full Time Series
    ax1 = plt.subplot(gs[0])
    
    # X-axis: Time + mintime - Keptime (BJD/BKJD)
    x = time + mintime - Keptime
    y = flux + 1.0
    
    ax1.plot(x, y, 'k.', markersize=1)
    ax1.set_ylabel("Relative Flux")
    ax1.set_xlabel("Time (days)")
    ax1.set_title("Single Pulse Detection")
    
    # Mark the event
    # ans.epo is in 'time' frame (0-based)
    t0_plot = ans.epo + mintime - Keptime
    dur = ans.tdur
    
    ymin, ymax = ax1.get_ylim()
    ax1.axvline(t0_plot, color='r', alpha=0.5)
    ax1.axvspan(t0_plot - dur/2, t0_plot + dur/2, color='r', alpha=0.2)
    
    # Bottom Row Split
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[1, 2])
    
    # Panel 3: Stats (Bottom Left)
    ax3 = plt.subplot(gs_bottom[0])
    ax3.axis('off')
    
    stats_text = [
        f"Type: Single Pulse",
        f"T0   = {t0_plot:.6f} days",
        f"Dur  = {ans.tdur*24:.2f} hours",
        f"Depth= {ans.depth*1e6:.0f} ppm",
        f"SNR  = {ans.snr:.1f}"
    ]
    
    for i, line in enumerate(stats_text):
        ax3.text(0.1, 0.8 - i*0.15, line, fontsize=14)
    
    # Panel 2: Zoomed View (Bottom Right)
    ax2 = plt.subplot(gs_bottom[1])
    
    # Zoom window: +/- 3 durations or fixed width
    window = max(3 * dur, 0.5) # at least 0.5 days
    
    # Mask based on x (plot coordinates)
    mask = np.abs(x - t0_plot) < window
    
    if np.sum(mask) > 0:
        ax2.plot(x[mask], y[mask], 'k.', markersize=3)
        ax2.set_xlim(t0_plot - window, t0_plot + window)
        
        # Plot box model
        t_model = np.linspace(t0_plot - window, t0_plot + window, 1000)
        f_model = np.ones_like(t_model)
        in_transit = np.abs(t_model - t0_plot) < dur/2
        f_model[in_transit] = 1.0 - ans.depth
        
        ax2.plot(t_model, f_model, 'r-', alpha=0.7, linewidth=2)
        
    ax2.set_ylabel("Relative Flux")
    ax2.set_xlabel("Time (days)")
    ax2.set_title("Zoomed Event")
    
    plt.tight_layout()
    
    if plots == 2:        
        directory = os.path.dirname(filename)
        if is_writable(directory):
            filename_without_extension = ".".join(filename.split(".")[:-1])
            fig.savefig(filename_without_extension+"_pulse.png", dpi=150)
        else:
            print("Warning: Path to datafile not writable, attempting to make plot in current directory")
            filename_base = os.path.basename(filename)
            filename_without_extension = ".".join(filename_base.split(".")[:-1])
            fig.savefig(filename_without_extension+"_pulse.png", dpi=150)
        plt.show()

    elif plots == 3:
        directory = os.path.dirname(filename)
        if is_writable(directory):
            filename_without_extension = ".".join(filename.split(".")[:-1])
            fig.savefig(filename_without_extension+"_pulse.png", dpi=150)
        else:
            print("Warning: Path to datafile not writable, attempting to make plot in current directory")
            filename_base = os.path.basename(filename)
            filename_without_extension = ".".join(filename_base.split(".")[:-1])
            fig.savefig(filename_without_extension+"_pulse.png", dpi=150)
        plt.close(fig)

    else:
        plt.show()

def bls_pulse(tpy5_inputs, time=np.array([0]), flux=np.array([0])):
    """
    Combined Periodic and Single Pulse Search.
    
    Runs both standard BLS (periodic) and Pulse Search (single event).
    Returns the result with the highest SNR.
    
    If Pulse Search wins:
        - Returns a gbls_ans_class object with:
            - bper = 0 (indicating single event)
            - epo = pulse t0
            - tdur = pulse duration
            - depth = pulse depth
            - snr = pulse snr
    
    If BLS wins:
        - Returns the standard gbls_ans_class object from bls()
        
    Generates a consolidated plot if inputs.plots > 0.
    """
    
    # 1. Run Periodic BLS
    # Temporarily disable plotting in bls() to handle it here
    # Force return_spectrum=True so we have data for plotting if needed
    original_plots = tpy5_inputs.plots
    original_return_spectrum = getattr(tpy5_inputs, 'return_spectrum', False)
    
    tpy5_inputs.plots = 0 
    tpy5_inputs.return_spectrum = True
    
    # Ensure time and flux are loaded if not provided
    if (time.shape[0] < 2) or (flux.shape[0] < 2):
        if tpy5_inputs.lcdir == "":
            filename = tpy5_inputs.filename
        else:
            filename = tpy5_inputs.lcdir + "/" + tpy5_inputs.filename
        time, flux = readfile(filename)
        # Pre-process like bls() does
        mintime = np.min(time)
        time = time - mintime
        flux = flux - np.median(flux)
    else:
        # If provided, we still need mintime for plotting if we want to match bls() behavior exactly
        # But bls() calculates mintime internally. 
        # If we pass time/flux to bls(), it uses them as is (assuming they are already processed or bls processes them).
        # bls() does: mintime = np.min(time); time = time - mintime; flux = flux - np.median(flux)
        # So we should let bls() do its thing, but for pulse search we need the same processed data?
        # Actually, bls() modifies 'time' and 'flux' local variables, not the passed arrays (unless in-place).
        # Python arguments are passed by assignment. 'time = time - mintime' creates a new local array.
        # So the passed 'time' is unchanged.
        # We should process time/flux here so we pass the SAME data to both searches.
        mintime = np.min(time)
        time_proc = time - mintime
        flux_proc = flux - np.median(flux)
        
        # Update: bls() re-calculates mintime and re-centers time/flux. 
        # If we pass already centered data, mintime=0.
        # To be safe, let's pass the original data to bls() and let it handle it.
        # But for pulse search, we need to do the same processing.
        pass

    # Run BLS
    bls_ans = bls(tpy5_inputs, time, flux)
    
    # Restore inputs settings
    tpy5_inputs.plots = original_plots
    tpy5_inputs.return_spectrum = original_return_spectrum
    
    # 2. Run Pulse Search
    # We need to process data for pulse search as bls() does internally
    # bls() does:
    mintime = np.min(time)
    time_proc = time - mintime
    flux_proc = flux - np.median(flux)
    
    # Use parameters from inputs class
    pulse_res = compute_pulse_search(
        time_proc, 
        flux_proc, 
        min_duration_hours=tpy5_inputs.pulse_min_duration_hours, 
        max_duration_hours=tpy5_inputs.pulse_max_duration_hours,
        bin_duration_days=tpy5_inputs.pulse_bin_duration_days
    )
    
    # 3. Compare Results
    pulse_snr = pulse_res['snr'] if pulse_res is not None else 0.0
    
    # Determine winner
    if bls_ans.snr >= pulse_snr:
        final_ans = bls_ans
        is_periodic = True
    else:
        # Create answer object for pulse
        final_ans = gbls_ans_class()
        final_ans.epo = pulse_res['t0'] # This t0 is relative to time_proc start (0) + mintime?
        # compute_pulse_search uses time_proc which starts at 0.
        # Its returned t0 is in that time frame.
        # bls_ans.epo is also in the shifted time frame (0 to duration).
        # When plotting, makeplot adds mintime back.
        
        final_ans.epo = pulse_res['t0']
        
        # Estimate period for single transit
        rho_star = pval.get_stellar_density(tpy5_inputs.mstar, tpy5_inputs.rstar)
        P_est, _, _, _ = pval.estimate_single_transit_period(
            time=time_proc,
            tdur=pulse_res['duration'],
            rho_star=rho_star,
            margin_factor=1.5
        )
        final_ans.bper = P_est
        
        final_ans.snr = pulse_res['snr']
        final_ans.tdur = pulse_res['duration']
        final_ans.depth = -pulse_res['depth']
        final_ans.bpower = pulse_res['snr'] # Use SNR as power proxy
        is_periodic = False
        
    # 4. Plotting
    if tpy5_inputs.plots > 0:
        if tpy5_inputs.lcdir == "":
            filename = tpy5_inputs.filename
        else:
            filename = tpy5_inputs.lcdir + "/" + tpy5_inputs.filename
            
        Keptime = tpy5_inputs.zerotime
        
        if is_periodic:
            # Use standard makeplot
            # We need to ensure we have the spectrum data
            if bls_ans.periods is not None:
                makeplot(
                    bls_ans.periods, 
                    bls_ans.power, 
                    time_proc, 
                    flux_proc, 
                    mintime, 
                    Keptime, 
                    bls_ans.epo, 
                    bls_ans.bper, 
                    bls_ans.bpower, 
                    bls_ans.snr, 
                    bls_ans.tdur, 
                    bls_ans.depth, 
                    filename, 
                    tpy5_inputs.plots
                )
        else:
            # Use new pulse plot
            makeplot_pulse(
                time_proc, 
                flux_proc, 
                final_ans, 
                mintime,
                Keptime,
                filename, 
                tpy5_inputs.plots
            )
            
    # Cleanup spectrum if not requested
    if is_periodic and not original_return_spectrum:
        bls_ans.periods = None
        bls_ans.power = None
        bls_ans.freqs = None
        
    return final_ans

## ADDING THE TLS FUNCTIONALITIES
def minstpars(runpar, starr):
    '''
    Imposing a minimum threshold on our values to prevent a duration grid
    that goes to values which are too small. Determined by pipeline parameters.
    '''
    minarr = (1-runpar.lowbuff) * starr
    minarr[minarr < runpar.stthres] = (1-runpar.stbuff) * starr[minarr < runpar.stthres]
    return tuple(minarr)

def setgridbounds(runpar, t):
    '''
    Considering if there is already a minimum and maximum frequency set 
    or if we are using the default parameters. Then, sanity check with a/rstar.
    Setting the lower frequency for TLS by the number of transits. 
    '''
    # Setting the minimum period if the frequency is set.
    pars = (runpar.pars).copy()
    if runpar.freq2 <= 0:
        pars['period_min'] = 0.5
        # runpar.freq2 = 2.0
    else:
        pars['period_min'] = 1/runpar.freq2
        
    # Checking to see if pmin > than the minimum arstarmin
    T2 = kepfac/(runpar.mstar*Msun) * (runpar.arstarmin*runpar.rstar*Rsun)**3
    pmin = np.sqrt(T2)/day2sec
    if pmin > pars['period_min']:
        pars['period_min'] = pmin
        # runpar.freq2 = 1/pmin
        
    # Setting the upper bound for the transit search
    if runpar.freq1 <= 0:
        pars['n_transits_min'] = 2
        # runpar.freq1 = 2.0/(np.max(t)-np.min(t))
    else:
        pars['n_transits_min'] = int((np.max(t)-np.min(t))*runpar.freq1)
        if pars['n_transits_min'] < 1:
            pars['n_transits_min'] = 1
            
    return pars
    
def getTLSDict(runpar, t):
    '''
    We take the general parameters for TLS and curate it for the specific RIC.
    This is primarily with the stellar parameters, and then we check for the
    minimum period threshold as well as consider if we need to plot.
    '''

    pars = setgridbounds(runpar, t)
    starr = np.array([runpar.rstar, runpar.mstar])
    pars['R_star'], pars['M_star'], pars['u'] = *starr, runpar.u    
    pars['R_star_min'], pars['M_star_min'] = minstpars(runpar, starr)
    pars['R_star_max'], pars['M_star_max'] = starr*(1+runpar.upbuff)
    
    return pars

def tls(tpy5_inputs, t=np.array([0]), f=np.array([0]), dy=None):
    '''
    Running the TLS module and setting its parameters in the dictionary.
    This is with the limb-darkening included, and no plot will be shown.
    Zero-point is subtracted off.

    verbose bug in tls means the keyword argument is given in the power function.
    '''
    if tpy5_inputs.lcdir == "":
        filename = tpy5_inputs.filename
    else:
        filename = tpy5_inputs.lcdir + "/" + tpy5_inputs.filename
    ## This branch hasn't yet been tested! Depends where this ends up.
    if (t.shape[0] < 2) or (f.shape[0] < 2):
        t, f, dy = readfile(filename)
    
    defaultpars = getTLSDict(tpy5_inputs, t)
    # The direct user input of parameters will override defaults.
    pars = {**defaultpars, **tpy5_inputs.pars}
    if tpy5_inputs.search_verbose:
        pprint.pprint(pars)
    t0 = t - tpy5_inputs.zerotime
    model = tlsm.transitleastsquares(t0, f, dy=dy)
    tlsob = model.power(**pars, verbose=tpy5_inputs.search_verbose) # running TLS
    srU, srC = np.unique(tlsob.SR, return_counts=True)
    srflag = (srU[0] < 10**(-3)) & (srC[0] > len(tlsob.SR)/100)

    tlsans = gbls_ans_class()
    tlsans.epo      = tlsob.T0
    tlsans.bper     = tlsob.period
    tlsans.rawper   = tlsob.periods[np.argmax(tlsob.power_raw)]    
    tlsans.bpower   = tlsob.SDE
    tlsans.snr      = tlsob.snr
    tlsans.tdur     = tlsob.duration
    tlsans.depth    = (1-tlsob.depth)
    tlsans.SR       = srflag
    if tpy5_inputs.return_spectrum:
        tlsans.periods  = tlsob.periods
        tlsans.power    = tlsob.power  
        # Rather than the frequencies, we return the signal residuals for TLS
        tlsans.SR       = tlsob.SR

    if tpy5_inputs.plots > 0:
        timezeroed = t - np.min(t)
        if np.isnan(tlsob.period) or tlsob.transit_count <= 1:
            makeplot_pulse(timezeroed, f, tlsans, np.min(t), tpy5_inputs.zerotime, filename, tpy5_inputs.plots)
        else:
            makeplot(tlsob.periods, tlsob.power, timezeroed, f, np.min(t), tpy5_inputs.zerotime, \
            tlsob.T0, tlsob.period, tlsob.SDE, tlsob.snr, tlsob.duration, (1-tlsob.depth), filename, tpy5_inputs.plots)
        
    return tlsans
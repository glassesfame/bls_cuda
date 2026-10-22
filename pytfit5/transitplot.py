import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pytfit5.transitmodel import transitModel
from pytfit5.keplerian import transitDuration
from pytfit5.effects import ttv_lininterp

def plotTransit(phot, sol, pl_to_plot=1, nintg=41, ntt=-1, tobs=-1, omc=-1,
                filename=None, y1=None, y2=None, xdur_scale=1.5):
    """
    Plots a transit model. Assuming time is in days. Set flux=0 for no scatterplot.

    phot: Phot object from reading data file
    sol: Transit model object with parameters
    nintg: Number of points inside the integration time
    pl_to_plot: Index of planet to plot. 1 being the first planet
    y1, y2: Optional y-axis limits. If not provided, limits are computed from the data.
    xdur_scale: Multiplier applied to the transit duration when setting x-axis limits.
    """

    # Read phot class
    time = phot.time[(phot.icut == 0)]
    if np.isclose(np.median(phot.flux_f), 0, atol=0.2):
        flux = phot.flux_f[(phot.icut == 0)] + 1
    else:
        flux = phot.flux_f[(phot.icut == 0)]
    itime = phot.itime[(phot.icut == 0)]

    pl_to_plot -= 1 # Shift indexing to start at 0

    t0 = sol.t0[pl_to_plot]
    per = sol.per[pl_to_plot]
    zpt = sol.zpt

    # Copy the original Rp/R* before modifying it
    rdr = sol.rdr.copy()

    # Remove the other planets from the model
    for i in range(sol.npl):
        if i != pl_to_plot:
            sol.rdr[i] = 0

    tmodel = transitModel(sol, time, itime, nintg, ntt, tobs, omc) - zpt
    flux = flux # - zpt # Remove the zero point to always plot around 1

    # Second model with only the other planets to substract
    sol.rdr = rdr.copy()
    sol.rdr[pl_to_plot] = 0
    tmodel2 = transitModel(sol, time, itime, nintg, ntt, tobs, omc)

    # Restore the original Rp/R*
    sol.rdr = rdr

    tdur = transitDuration(sol, pl_to_plot)*24
    if tdur < 0.01 or np.isnan(tdur):
        tdur = 2

    # Fold the time array and sort it. Handle TTVs
    ph1 = t0/per - np.floor(t0/per)
    phase = np.empty(len(time))
    for i, x in enumerate(time):
        if type(ntt) is not int and ntt[pl_to_plot] > 0:
            ttcor = ttv_lininterp(tobs, omc, ntt, x, pl_to_plot)
        else:
            ttcor = 0
        t = x - ttcor
        phase[i] = (t/per - np.floor(t/per) - ph1) * per*24
        if phase[i] >  0.5 * per*24:
            phase[i] -= per*24
        if phase[i] < -0.5 * per*24:
            phase[i] += per*24

    i_sort = np.argsort(phase)
    phase_sorted = phase[i_sort]
    model_sorted = tmodel[i_sort]

    stdev = np.std(flux - tmodel)

    # Remove the other planets
    fplot = flux - tmodel2 + 1

    # Find bounds of plot
    i1, i2 = np.searchsorted(phase_sorted, (-tdur, tdur))
    if i1 == i2:
        i1 = 0
        i2 = len(model_sorted)
    ymin = min(model_sorted[i1:i2])
    ymax = max(model_sorted[i1:i2])
    y1_auto = ymin - 0.1*(ymax-ymin) - 3.0*stdev
    y2_auto = ymax + 0.1*(ymax-ymin) + 3.0*stdev
    if np.abs(y2_auto - y1_auto) < 1.0e-10:
        y1_auto = min(flux)
        y2_auto = max(flux)

    #Make sure data outliers are seen.
    y1_auto = np.min([y1_auto, min(flux)])
    y2_auto = np.max([y2_auto, max(flux)])

    if y1 is None:
        y1 = y1_auto
    if y2 is None:
        y2 = y2_auto

    mpl.rcParams.update({'font.size': 22}) # Adjust font

    # Generate smooth model for plotting
    phase_smooth = np.linspace(-xdur_scale*tdur, xdur_scale*tdur, 1000)
    # Convert phase (hours) to time (days) relative to t0
    # phase = (t - t0) * 24
    # t = phase/24 + t0
    time_smooth = t0 + phase_smooth/24.0
    itime_smooth = float(np.median(itime))

    # Save original radii
    rdr_orig = sol.rdr.copy()
    # Remove other planets
    for i in range(sol.npl):
        if i != pl_to_plot:
            sol.rdr[i] = 0
    
    # Calculate model with no TTVs
    smooth_model = transitModel(sol, time_smooth, itime=itime_smooth, nintg=nintg, ntt=-1) - zpt
    
    # Restore radii
    sol.rdr = rdr_orig

    plt.figure(figsize=(12,6)) # Adjust size of figure
    plt.scatter(phase, fplot, c="blue", s=100.0, alpha=0.35, edgecolors="none") #scatter plot
    plt.plot(phase_smooth, smooth_model, c="red", lw=3.0)
    plt.xlabel('Phase (hours)') #x-label
    plt.ylabel('Relative Flux') #y-label
    plt.axis((-xdur_scale*tdur, xdur_scale*tdur, y1, y2))
    plt.tick_params(direction="in")
    if (filename==None):
        plt.show()
    else:
        plt.savefig(filename, dpi=150)


def plotTransitMCMC(chain, burnin, sol, params_to_fit, phot,
                    pl_to_plot=1, n_samples=300, nintg=41,
                    ntt=-1, tobs=-1, omc=-1, filename=None):
    """
    Plots MCMC posterior confidence intervals for a transit model.

    Draws n_samples rows uniformly from chain[burnin:], reconstructs the transit
    model for each draw, and overlays 1-sigma and 2-sigma shaded envelopes plus
    the posterior median on the phase-folded photometry.

    chain        : 2-D MCMC chain array, shape (n_steps, n_params)
    burnin       : Number of leading rows to discard as burn-in
    sol          : Transit model object; non-fitted parameters are taken from here
    params_to_fit: Parameter name list, same as passed to genmcmcInput
    phot         : Phot object containing the photometry
    pl_to_plot   : Planet to plot, 1-based (default 1)
    n_samples    : Number of posterior draws used to build the envelope (default 300)
    nintg        : Oversampling points for time-convolution (default 41)
    ntt, tobs, omc : TTV arrays forwarded to transitModel when folding the data
    filename     : If given, saves figure to this path; otherwise displays interactively.

    Returns the matplotlib Figure.
    """
    from pytfit5.transitmcmc import _get_fit_parameter_layout
    from pytfit5.transitmodel import nb_st_param, nb_pl_param, transit_model_class

    pl_idx = pl_to_plot - 1
    _rdr_off = 3  # rdr is the 4th entry (index 3) in the planet parameter block

    # --- Decode chain samples back to full sol arrays ---
    id_to_fit, log_space_params, _ = _get_fit_parameter_layout(sol, params_to_fit)
    sol_base = sol.to_array()

    post_chain = chain[burnin:, :]
    n_post = post_chain.shape[0]
    rng = np.random.default_rng()
    draw_idx = rng.choice(n_post, size=min(n_samples, n_post), replace=False)
    samples = post_chain[draw_idx]

    # Median posterior solution — used to subtract other planets from the data scatter
    mm = np.median(post_chain, axis=0)
    sol_med = sol_base.copy()
    for i, ind in enumerate(id_to_fit):
        sol_med[ind] = np.exp(mm[i]) if ind in log_space_params else mm[i]

    # --- Phase grid for smooth model ---
    t0_ref  = sol_med[nb_st_param + pl_idx * nb_pl_param + 0]  # t0
    per_ref = sol_med[nb_st_param + pl_idx * nb_pl_param + 1]  # per
    zpt_ref = sol_med[7]                                         # zpt

    sol_tmp = transit_model_class()
    sol_tmp.from_array(sol_med)
    tdur_ref = transitDuration(sol_tmp, pl_idx) * 24  # hours
    if tdur_ref < 0.01 or np.isnan(tdur_ref):
        tdur_ref = 2.0

    phase_smooth = np.linspace(-1.5 * tdur_ref, 1.5 * tdur_ref, 500)
    time_smooth  = t0_ref + phase_smooth / 24.0
    itime_smooth = float(np.median(phot.itime))

    # --- Evaluate model for each posterior sample ---
    model_curves = np.empty((len(samples), len(phase_smooth)))
    for s_i, row in enumerate(samples):
        sol_s = sol_base.copy()
        for i, ind in enumerate(id_to_fit):
            sol_s[ind] = np.exp(row[i]) if ind in log_space_params else row[i]
        # Zero out other planets so we only see the target transit
        for pi in range(sol.npl):
            if pi != pl_idx:
                sol_s[nb_st_param + pi * nb_pl_param + _rdr_off] = 0.0
        zpt_s = sol_s[7]
        model_curves[s_i] = transitModel(sol_s, time_smooth, itime=itime_smooth,
                                         nintg=nintg, ntt=-1) - zpt_s

    # Percentile envelopes: 2.5, 16, 50, 84, 97.5
    p = np.percentile(model_curves, [2.5, 16, 50, 84, 97.5], axis=0)

    # --- Phase-fold the data (mirrors plotTransit) ---
    sel  = phot.icut == 0
    time  = phot.time[sel]
    flux  = phot.flux_f[sel]
    if np.isclose(np.median(flux), 0, atol=0.2):
        flux = flux + 1
    itime = phot.itime[sel]

    # Subtract the other planets from the data using the median sol
    sol_other = sol_med.copy()
    sol_other[nb_st_param + pl_idx * nb_pl_param + _rdr_off] = 0.0
    tmodel_other = transitModel(sol_other, time, itime, nintg, ntt, tobs, omc)
    fplot = flux - tmodel_other + 1

    ph1 = t0_ref / per_ref - np.floor(t0_ref / per_ref)
    phase_data = np.empty(len(time))
    for i, x in enumerate(time):
        ttcor = 0.0
        if type(ntt) is not int and ntt[pl_idx] > 0:
            ttcor = ttv_lininterp(tobs, omc, ntt, x, pl_idx)
        t = x - ttcor
        phase_data[i] = (t / per_ref - np.floor(t / per_ref) - ph1) * per_ref * 24
        if phase_data[i] >  0.5 * per_ref * 24:
            phase_data[i] -= per_ref * 24
        if phase_data[i] < -0.5 * per_ref * 24:
            phase_data[i] += per_ref * 24

    # --- Axis limits ---
    i1, i2 = np.searchsorted(phase_smooth, (-tdur_ref, tdur_ref))
    if i1 == i2:
        i1, i2 = 0, len(phase_smooth)
    ymin_m = p[0, i1:i2].min()
    ymax_m = p[4, i1:i2].max()
    stdev  = np.std(fplot)
    y1 = ymin_m - 0.1 * (ymax_m - ymin_m) - 3.0 * stdev
    y2 = ymax_m + 0.1 * (ymax_m - ymin_m) + 3.0 * stdev
    y1 = min(y1, fplot.min())
    y2 = max(y2, fplot.max())

    mpl.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(phase_data, fplot, c="blue", s=100, alpha=0.35,
               edgecolors="none", zorder=2, label="Data")
    ax.fill_between(phase_smooth, p[0], p[4], color="orange", alpha=0.25, label="2σ")
    ax.fill_between(phase_smooth, p[1], p[3], color="orange", alpha=0.50, label="1σ")
    ax.plot(phase_smooth, p[2], color="red", lw=2.5, label="Median", zorder=3)

    ax.set_xlabel("Phase (hours)")
    ax.set_ylabel("Relative Flux")
    ax.set_xlim(-1.5 * tdur_ref, 1.5 * tdur_ref)
    ax.set_ylim(y1, y2)
    ax.tick_params(direction="in")
    ax.legend(fontsize=14, loc="lower right")

    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename, dpi=150)

    return fig


def printParams(sol):
    """
    Prints the parameters in a nice way.

    sol: Transit model object containing the parameters to print
    """

    stellarDict = {
        "ρ* (g/cm³)": "rho", "c1": "nl1", "c2": "nl2", "q1": "nl3", "q2": "nl4",
        "Dilution": "dil", "Velocity Offset": "vof", "Photometric zero point": "zpt"
    }

    if hasattr(sol, "dscale"):
        stellarDict["Error scale factor"] = "dscale"

    planetDict = {
        "t0 (days)": "t0", "Period (days)": "per", "Impact parameter": "bb", "Rp/R*": "rdr",
        "sqrt(e)cos(w)": "ecw", "sqrt(e)sin(w)": "esw", "RV Amplitude (m/s)": "krv",
        "Thermal eclipse depth (ppm)": "ted", "Ellipsoidal variations (ppm)": "ell", "Albedo amplitude (ppm)": "alb"
    }

    # Stellar params
    for key in stellarDict:
        var_name = stellarDict[key]
        val = getattr(sol, var_name)
        try:
            err = getattr(sol, "d" + var_name)
        except:
            err = 0

        if val != 0:
            exponent = np.floor(np.log10(abs(val)))
        else:
            exponent = 1

        if abs(exponent) > 2:
            print(f"{key + ':':<30} {val:>10.3e} ± {err:.3e}")
        elif len(str(val)) > 7:
            print(f"{key + ':':<30} {val:>10.7f} ± {err:.7f}")
        else:
            print(f"{key + ':':<30} {val:>10} ± {err}")

    # Planet params
    for j in range(sol.npl):
        if sol.npl > 1:
            print(f"\nPlanet #{j + 1}:")
        for key in planetDict:
            var_name = planetDict[key]
            val = getattr(sol, var_name)
            try:
                err = getattr(sol, "d" + var_name)
            except:
                err = 0

            p_val = val[j]
            p_err = err[j]
            if p_val != 0:
                exponent = np.floor(np.log10(abs(p_val)))
            else:
                exponent = 1

            if abs(exponent) > 2:
                print(f"{key + ':':<30} {p_val:>10.3e} ± {p_err:.3e}")
            elif len(str(p_val)) > 7:
                print(f"{key + ':':<30} {p_val:>10.7f} ± {p_err:.7f}")
            else:
                print(f"{key + ':':<30} {p_val:>10} ± {p_err}")


def plot_lightcurve_summary(phot, bls_ans=None, tpy5_inputs=None, figsize=(14, 10)):
    """
    Plot a comprehensive lightcurve summary showing raw and detrended flux.
    Optionally include BLS results with phased lightcurve and transit closeup.
    
    Parameters
    ----------
    phot : phot_class
        Photometry object with time, flux, flux_f, icut arrays
    bls_ans : gbls_ans_class, optional
        BLS results object containing epo, bper, tdur, depth, bpower, snr
        If provided, adds phased lightcurve and transit closeup panels
    tpy5_inputs : tpy5_inputs_class, optional
        Input parameters (used for zerotime if provided)
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Determine zerotime offset
    zerotime = 0.0
    if tpy5_inputs is not None:
        zerotime = tpy5_inputs.zerotime
    
    # Filter good data
    good_mask = (phot.icut == 0)
    time_plot = phot.time[good_mask] - zerotime
    flux_raw = phot.flux[good_mask]
    flux_detrended = phot.flux_f[good_mask]
    
    # Determine if we have BLS results
    has_bls = (bls_ans is not None and 
               hasattr(bls_ans, 'bper') and 
               hasattr(bls_ans, 'epo'))
    
    # Create figure with appropriate layout
    if has_bls:
        # 4 rows: raw flux, detrended flux, phased, transit closeup + stats
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)
        
        ax1 = plt.subplot(gs[0])  # Raw flux
        ax2 = plt.subplot(gs[1])  # Detrended flux
        ax3 = plt.subplot(gs[2])  # Phased lightcurve
        
        # Split bottom row for transit closeup and statistics
        gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], 
                                               width_ratios=[1, 1.5], wspace=0.3)
        ax4_stats = plt.subplot(gs4[0])  # Statistics panel
        ax4_transit = plt.subplot(gs4[1])  # Transit closeup
        
    else:
        # 2 rows only: raw flux and detrended flux
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*0.5), 
                                       sharex=True)
    
    # Common y-axis calculations
    def get_ylim(data):
        ymin, ymax = np.min(data), np.max(data)
        dy = ymax - ymin
        return ymin - 0.05*dy, ymax + 0.05*dy
    
    # Panel 1: Raw flux
    ax1.scatter(time_plot, flux_raw, s=1, color='black', alpha=0.6)
    ax1.set_ylabel('Raw Flux', fontsize=11)
    ax1.set_ylim(get_ylim(flux_raw))
    ax1.tick_params(direction='in', which='major', bottom=True, top=True,
                   left=True, right=True, length=8, width=1.5)
    ax1.tick_params(direction='in', which='minor', bottom=True, top=True,
                   left=True, right=True, length=4, width=1)
    ax1.grid(True, alpha=0.3)
    
    # Mark transit locations on raw flux if BLS available
    if has_bls:
        epo = bls_ans.epo
        bper = bls_ans.bper
        tdur = bls_ans.tdur if hasattr(bls_ans, 'tdur') else 0.1
        
        # Find all transit times in the data range
        t_min, t_max = np.min(time_plot), np.max(time_plot)
        n_transit_start = int(np.floor((t_min - epo + zerotime) / bper))
        n_transit_end = int(np.ceil((t_max - epo + zerotime) / bper))
        
        for n in range(n_transit_start, n_transit_end + 1):
            t_center = epo - zerotime + n * bper
            if t_min <= t_center <= t_max:
                ax1.axvline(t_center, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Panel 2: Detrended flux
    ax2.scatter(time_plot, flux_detrended, s=1, color='black', alpha=0.6)
    ax2.set_ylabel('Detrended Flux', fontsize=11)
    ax2.set_xlabel('Time (days)' if not has_bls else '', fontsize=11)
    ax2.set_ylim(get_ylim(flux_detrended))
    ax2.tick_params(direction='in', which='major', bottom=True, top=True,
                   left=True, right=True, length=8, width=1.5)
    ax2.tick_params(direction='in', which='minor', bottom=True, top=True,
                   left=True, right=True, length=4, width=1)
    ax2.grid(True, alpha=0.3)
    
    # Mark transit locations on detrended flux if BLS available
    if has_bls:
        for n in range(n_transit_start, n_transit_end + 1):
            t_center = epo - zerotime + n * bper
            if t_min <= t_center <= t_max:
                ax2.axvline(t_center, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    if not has_bls:
        # No BLS results, just show the two panels
        plt.tight_layout()
        return fig
    
    # Panel 3: Phased lightcurve (only if BLS available)
    phase = ((time_plot + zerotime - epo) / bper) % 1.0
    phase[phase > 0.25] = phase[phase > 0.25] - 1.0
    
    ax3.scatter(phase, flux_detrended, s=1, color='black', alpha=0.6)
    ax3.set_ylabel('Relative Flux', fontsize=11)
    ax3.set_xlabel('Phase', fontsize=11)
    ax3.set_xlim(np.min(phase), np.max(phase))
    ax3.set_ylim(get_ylim(flux_detrended))
    ax3.tick_params(direction='in', which='major', bottom=True, top=True,
                   left=True, right=True, length=8, width=1.5)
    ax3.tick_params(direction='in', which='minor', bottom=True, top=True,
                   left=True, right=True, length=4, width=1)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f'Phased to P={bper:.6f} days', fontsize=11)
    
    # Panel 4a: Statistics panel
    ax4_stats.set_xticks([])
    ax4_stats.set_yticks([])
    ax4_stats.set_frame_on(False)
    
    # Prepare statistics text
    stats_lines = []
    stats_lines.append(f"Period  = {bper:.6f} days")
    stats_lines.append(f"T0      = {epo:.6f} days")
    if hasattr(bls_ans, 'bpower'):
        stats_lines.append(f"Power   = {bls_ans.bpower:.6f}")
    if hasattr(bls_ans, 'snr'):
        stats_lines.append(f"SNR     = {bls_ans.snr:.1f}")
    if hasattr(bls_ans, 'tdur'):
        stats_lines.append(f"Duration = {tdur*24:.2f} hours")
    if hasattr(bls_ans, 'depth'):
        stats_lines.append(f"Depth   = {bls_ans.depth*1e6:.1f} ppm")
    
    # Display statistics
    for i, line in enumerate(stats_lines):
        ax4_stats.text(0.05, 0.85 - i*0.15, line, ha='left', va='top',
                      fontsize=11, family='monospace',
                      transform=ax4_stats.transAxes)
    
    ax4_stats.set_title('BLS Detection', fontsize=11, pad=10)
    
    # Panel 4b: Transit closeup (±1 transit duration)
    phase_hours = phase * bper * 24  # Convert to hours
    tdur_hours = tdur * 24
    
    # Select data within ±1 duration
    transit_mask = np.abs(phase_hours) < tdur_hours
    if np.any(transit_mask):
        phase_transit = phase_hours[transit_mask]
        flux_transit = flux_detrended[transit_mask]
        
        ax4_transit.scatter(phase_transit, flux_transit, s=3, color='black', alpha=0.8)
        ax4_transit.set_ylabel('Relative Flux', fontsize=11)
        ax4_transit.set_xlabel('Phase (Hours)', fontsize=11)
        ax4_transit.set_xlim(-tdur_hours, tdur_hours)
        ax4_transit.set_ylim(get_ylim(flux_transit))
        ax4_transit.tick_params(direction='in', which='major', bottom=True, top=True,
                               left=True, right=True, length=8, width=1.5)
        ax4_transit.tick_params(direction='in', which='minor', bottom=True, top=True,
                               left=True, right=True, length=4, width=1)
        ax4_transit.grid(True, alpha=0.3)
        ax4_transit.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax4_transit.set_title('Transit Closeup (±1 Duration)', fontsize=11)
    
    plt.tight_layout()
    return fig

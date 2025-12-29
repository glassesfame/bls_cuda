import sys
import os
from urllib.error import HTTPError

import pandas as pd

import numpy as np
import copy 
from astroquery.mast import Observations, Catalogs
from astropy.io import fits

import pytfit5.transitmodel as transitm
import pytfit5.keplerian as kep
import pytfit5.transitfit as transitf
import matplotlib.pyplot as plt  #MatPlotLib for some simple plots 

from scipy import stats #For Kernel Density Estimation

# Nice for keeping an eye on progress.
from tqdm import trange

import concurrent.futures

from urllib.parse import quote as urlencode
from urllib.request import urlretrieve

import http.client as httplib

import json

def fit_ttvs(phot, sol_fit, ntt=-1, tobs=-1, omc=-1, pflag = 0, pstart = 0):

    if isinstance(pflag, int):
        pflag = np.ones((sol_fit.npl),dtype=int)*pflag

    if isinstance(pstart, int):
        pstart = np.ones((sol_fit.npl),dtype=int)*pstart
    
    flux_f_copy = np.copy(phot.flux_f)
    
    Tmin = np.min(phot.time[(phot.icut == 0)])
    Tmax = np.max(phot.time[(phot.icut == 0)])
    
    tt_list      = []
    for nplanet in trange(sol_fit.npl):
    
        #Zero out the current planet 
        rdr_copy = sol_fit.rdr[nplanet]
        sol_fit.rdr[nplanet] = 0
        #model with only other planets
        tmodel = transitm.transitModel(sol_fit, phot.time, itime=phot.itime, ntt=ntt, tobs=tobs, omc=omc)
        sol_fit.rdr[nplanet] = rdr_copy
    
        #Make model for single planet 
        sol_c     = transitm.transit_model_class()
        # Parameters that define the star/scene for the transit model
        sol_c.rho = sol_fit.rho    
        sol_c.nl1 = sol_fit.nl1   
        sol_c.nl2 = sol_fit.nl2
        sol_c.nl3 = sol_fit.nl3   
        sol_c.nl4 = sol_fit.nl4  
        sol_c.t0  = [sol_fit.t0[nplanet]]             # Center of transit time (days)
        sol_c.per = [sol_fit.per[nplanet]]            # Orbital Period (days)
        sol_c.bb  = [sol_fit.bb[nplanet]]                      # Impact parameter
        sol_c.rdr = [sol_fit.rdr[nplanet]]  # Rp/R*
        sol_c.ecw = [0.0]                      # sqrt(e)cos(w)
        sol_c.esw = [0.0]                      # sqrt(e)sin(w)
        sol_c.krv = [0.0]                      # RV amplitude (m/s)
        sol_c.ted = [0.0]                     # thermal eclipse depth (ppm)
        sol_c.ell = [0.0]                      # Ellipsodial variations (ppm)
        sol_c.alb = [0.0]                      # Albedo amplitude (ppm)
        sol_c.npl = 1
        
        #Get duration of the current planet
        tdur = kep.transitDuration(sol_fit, nplanet)
        
        phot.flux_f = flux_f_copy - tmodel + 1
    
        # T0=sol(9)                +int((     Tmin-sol(9)                 )/sol(10)               +0.0d0)*sol(10)
        T0 = sol_fit.t0[nplanet]+np.floor((Tmin-sol_fit.t0[nplanet])/sol_fit.per[nplanet]+0.0  )*sol_fit.per[nplanet]

        cal_omc     = pstart[nplanet]
        cal_omc_old = 0.0
        d_cal_omc   = 0.0
    
        tt      = []
        # print(pflag[nplanet])
        while(T0 < Tmax):

            Ts = T0 - 2.0*tdur + cal_omc         + d_cal_omc
            Te = T0 + 2.0*tdur + cal_omc         + d_cal_omc
            Ts2= T0 - 0.5*tdur + cal_omc - 0.021 + d_cal_omc  # add 30-mins
            Te2= T0 + 0.5*tdur + cal_omc + 0.021 + d_cal_omc  
            sol_c.t0[0] = T0   + cal_omc         + d_cal_omc
    
            params_to_fit = ["t0"]
            phot.tflag = np.zeros((phot.time.shape[0]))
            phot.tflag[(phot.time >= Ts) & (phot.time <= Te)] = 1
            k =  len(phot.time[(phot.time >= Ts2) & (phot.time <= Te2)])
            # print(T0, k)
            if k > 3:
                
                cal_omc_old = cal_omc 
                
                sol_c_fit = transitf.fitTransitModel(sol_c, params_to_fit, phot)

                cal_omc = sol_c_fit.t0[0] - T0
                d_cal_omc = cal_omc - cal_omc_old
    
                tt.append([T0, cal_omc, sol_c_fit.dt0[0]])
    
            else:
                cal_omc += d_cal_omc
    
            T0 = T0 + sol_fit.per[nplanet]
    
            if pflag[nplanet] == 0:
                cal_omc   = 0.0  # check if we are using predictive
                d_cal_omc = 0.0

            # print(cal_omc, cal_omc_old, d_cal_omc)
            # input()
    
        tt_list.append(np.array(tt))
    
    #Restore photometry
    phot.flux_f = np.copy(flux_f_copy)
    
    tobs_list = []
    omc_list = []
    omc_err_list = []
    ntt_list = []
    for tt1 in tt_list:
        tobs_list.append(tt1[:,0])
        omc_list.append(tt1[:,1])
        omc_err_list.append(tt1[:,2])
        ntt_list.append(len(tt1[:,0]))
    
    ntt_new     = np.array(ntt_list)
    tobs_new    = pad_list_of_arrays(tobs_list)
    omc_new     = pad_list_of_arrays(omc_list)
    omc_err_new = pad_list_of_arrays(omc_err_list)

    return ntt_new, tobs_new, omc_new, omc_err_new


def plotTTVs(ntt, tobs, omc, omc_err, KOI, koi_df):
    # --- THE CORRECTED PLOT CODE ---
    num_planets = len(ntt)
    
    # Create a figure and a set of subplots.
    # sharex=True is the key to linking the x-axes.
    # We make the figure taller based on the number of planets.
    fig, axes = plt.subplots(
        nrows=num_planets,
        ncols=1,
        figsize=(12, 2.5 * num_planets),
        sharex=True
    )
    plt.rcParams.update({'font.size': 12})
    
    # If there's only one planet, axes is not a list, so we make it one
    if num_planets == 1:
        axes = [axes]
    
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, num_planets))
    
    # Loop through each planet and plot on its dedicated axis
    for i, ntt_1 in enumerate(ntt):
        # Select the correct axis for this planet
        ax = axes[i]
        
        # Get the data slice for the current planet
        time_data = tobs[i, 0:ntt_1]
        omc_data = omc[i, 0:ntt_1] * 24 * 60  # Convert to minutes
        omc_error_data = omc_err[i, 0:ntt_1] * 24 * 60 # Convert to minutes
    
        ax.errorbar(
            time_data[omc_error_data > 0],
            omc_data[omc_error_data > 0],
            yerr=omc_error_data[omc_error_data > 0],
            fmt='o',
            linestyle='none',
            capsize=4.0,
            label=str(koi_df["KOI"].values[i]),
            color=colors[i]
        )
        
        # Set the y-label for each subplot
        ax.set_ylabel('O-C (mins)')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # The x-axis label only needs to be set for the bottom-most plot
    axes[-1].set_xlabel('Time (BJD-2454900)')
    
    # Add a title for the entire figure
    fig.suptitle(f'Transit Timing Variations for KOI-{str(KOI)} System', fontsize=16, y=0.9)
    
    fig.subplots_adjust(hspace=0)
    
    plt.show()

def find_koi_rows(koi_identifier, Kepler_cat):
  """
  Finds rows based on the KOI identifier.
  - If koi_identifier is a float, it finds an exact match.
  - If koi_identifier is an int, it finds all rows where the integer part matches.
  """
  if isinstance(koi_identifier, float):
    # Case 1: Exact match for a float input (e.g., 377.01)
    print(f"Searching for exact KOI value: {koi_identifier}\n")
    return Kepler_cat[Kepler_cat['KOI'] == koi_identifier]
  elif isinstance(koi_identifier, int):
    # Case 2: Match the integer part (e.g., 377)
    # This selects rows where KOI is >= 377 and < 378
    print(f"Searching for KOI values starting with: {koi_identifier}\n")
    lower_bound = koi_identifier
    upper_bound = koi_identifier + 1
    return Kepler_cat[(Kepler_cat['KOI'] >= lower_bound) & (Kepler_cat['KOI'] < upper_bound)]
  else:
    # Handle cases where input is neither int nor float
    return "Invalid input. Please provide an integer or a float."

def get_photometry(koi_df, raw = 0, ztime = 54899.5):

    kic_id = int(koi_df["KIC"].to_numpy()[0])
    formatted_kic = str(kic_id).zfill(8)

    koi_int = int(koi_df["KOI"].to_numpy()[0])

    if (raw==1):
        file_url = f"https://kona.ubishops.ca/Kepler_n/koi{koi_int}.n/klc{formatted_kic}.dat" #Raw data file
    else:
        file_url = f"https://kona.ubishops.ca/Kepler_n/koi{koi_int}.n/klc{formatted_kic}.dc.dat" #Processed 
    
    # print(f"Constructed URL: {file_url}")

    try:
        # Read the space-delimited file with no header
        photometry_df = pd.read_csv(
            file_url,
            sep=r'\s+',
            header=None,
        )

        if len(photometry_df.columns) == 4:
            photometry_df.columns = ['time', 'flux', 'flux_err', 'itime']
        elif len(photometry_df.columns) == 3:
            photometry_df.columns = ['time', 'flux', 'flux_err']
        else:
            print(f"Warning: The data file at {file_url} contains an unexpected "
                  f"number of columns ({len(photometry_df.columns)}).")
            return None
        
    except HTTPError:
        print(f"Error: Could not retrieve data file at {file_url}. It may not exist (404 Not Found).")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None

    phot = phot_class()
    phot.time = photometry_df['time'].to_numpy() - ztime 
    phot.flux = photometry_df['flux'].to_numpy() + 1.0 #Offset to 1.
    phot.ferr = photometry_df['flux_err'].to_numpy()

    if len(photometry_df.columns) == 3:
        phot.itime = np.ones((phot.time.shape[0])) * 1765.5/86400.0 # Long cadence integration time
    else:
        phot.itime = photometry_df['itime'].to_numpy()

    phot.tflag = np.zeros(len(phot.time))  # Flag for in-transit data
    phot.icut  = np.zeros(len(phot.time))  # Flag for data cuts 
    phot.flux_f = np.copy(phot.flux)       # Space to store processed data

    phot.tflag = np.zeros((phot.time.shape[0]))

    ## Set inputs of the tpy5_inputs_class
    tpy5_inputs = tpy5_inputs_class() #This sets up parameters for data processing
    
    ## Star ID
    tpy5_inputs.kid   = int(koi_df["KIC"].to_numpy()[0])  #KIC ID
    tpy5_inputs.mstar = koi_df["mstar"].to_numpy()[0]
    tpy5_inputs.rstar = koi_df["rstar"].to_numpy()[0]
    tpy5_inputs.teff  = koi_df["teff"].to_numpy()[0] 
    tpy5_inputs.logg  = koi_df["logg"].to_numpy()[0] 
    tpy5_inputs.feh   = koi_df["[M/H]"].to_numpy()[0]
    
    ## and assign stellar errors (Assuming symmetric for now)
    tpy5_inputs.e_mstar = (koi_df["mstar_ep"].to_numpy()[0] + np.abs(koi_df["mstar_em"].to_numpy()[0])) / 2.0
    tpy5_inputs.e_rstar = (koi_df["rstar_ep"].to_numpy()[0] + np.abs(koi_df["rstar_em"].to_numpy()[0])) / 2.0
    tpy5_inputs.e_logg =  (koi_df["logg_ep"].to_numpy()[0]  + np.abs(koi_df["logg_em"].to_numpy()[0])) / 2.0
    tpy5_inputs.e_teff =  koi_df["teff_e"].to_numpy()[0]
    tpy5_inputs.e_metal = koi_df["[M/H]_e"].to_numpy()[0]

    return phot, tpy5_inputs

def populate_transit_model(main_df, koi_identifier):
    """
    Fetches and parses the n0.dat file for a KOI system, populating a
    transit model object.

    Args:
        main_df (pd.DataFrame): The main DataFrame with KOI and Disposition data.
        koi_identifier (int or float): The KOI system number (e.g., 377 or 377.01).

    Returns:
        TransitModelClass: A populated instance of the class, or None if an error occurs.
    """
    sol = transitm.transit_model_class()
    sol.t0 = []
    sol.per = []
    sol.bb = []
    sol.rdr = []
    sol.ecw = []
    sol.esw = []
    sol.krv = []
    sol.ted = []
    sol.ell = []
    sol.alb = []
    
    koi_int = int(koi_identifier)

    # --- Step 1: Find all confirmed planets in the system ---
    lower_bound = koi_int
    upper_bound = koi_int + 1
    
    # Filter for the system and for dispositions that DO NOT start with 'F'
    planets_in_system = main_df[
        (main_df['KOI'] >= lower_bound) &
        (main_df['KOI'] < upper_bound) &
        (~main_df['Disposition'].str.startswith('F', na=False))
    ].sort_values(by='KOI') # Sort to ensure planets are in order (377.01, .02, etc.)

    if planets_in_system.empty:
        print(f"No confirmed planets found for KOI system {koi_int}.")
        return None
    
    sol.npl = len(planets_in_system)
    print(f"Found {sol.npl} confirmed planets in system {koi_int}.")

    # --- Step 2: Fetch and parse the n0.dat file ---
    file_url = f"https://kona.ubishops.ca/Kepler_n/koi{koi_int}.n/n0.dat"
    print(f"Fetching model solution from: {file_url}")

    try:
        # Read the first two columns of the space-delimited file
        model_params_df = pd.read_csv(
            file_url,
            sep=r'\s+',
            engine='python',
            header=None,
            usecols=[0, 1],
            names=['param_name', 'param_value']
        )
        # Convert to a Series for easy key-value lookup (e.g., params_map['RHO'])
        params_map = pd.Series(
            model_params_df.param_value.values,
            index=model_params_df.param_name
        )
    except HTTPError:
        print(f"ERROR: Could not find n0.dat file at {file_url}.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    # --- Step 3: Map system-wide parameters ---
    system_param_map = {
        'RHO': 'rho', 'NL1': 'nl1', 'NL2': 'nl2', 'NL3': 'nl3',
        'NL4': 'nl4', 'DIL': 'dil', 'VOF': 'vof', 'ZPT': 'zpt'
    }
    for file_name, attr_name in system_param_map.items():
        if file_name in params_map:
            setattr(sol, attr_name, params_map[file_name])

    # --- Step 4: Map per-planet parameters for confirmed planets ---
    planet_param_map = {
        'EP': sol.t0, 'PE': sol.per, 'BB': sol.bb, 'RD': sol.rdr, 'EC': sol.ecw,
        'ES': sol.esw, 'KR': sol.krv, 'TE': sol.ted, 'EL': sol.ell, 'AL': sol.alb
    }

    # Iterate through each confirmed planet we found in Step 1
    for index, planet_row in planets_in_system.iterrows():
        # Get the planet number (e.g., from 377.01, we get 1)
        planet_num = int(str(planet_row['KOI']).split('.')[1])

        # For each type of parameter ('EP', 'PE', etc.), find the corresponding
        # value in the file (e.g., 'EP1') and append it to the correct list in 'sol'.
        for base_name, target_list in planet_param_map.items():
            param_to_find = f"{base_name}{planet_num}"
            if param_to_find in params_map:
                target_list.append(params_map[param_to_find])
            else:
                target_list.append(None) # Append None if param is missing

    return sol

def pad_list_of_arrays(list_of_arrays):
    """
    Converts a list of 1D NumPy arrays of varying lengths into a
    single 2D NumPy array by padding shorter arrays with zeros.

    Args:
        list_of_arrays (list): A list where each element is a 1D NumPy array.

    Returns:
        np.ndarray: A 2D NumPy array.
    """
    # Handle the edge case of an empty list
    if not list_of_arrays:
        return np.array([[]]) # Return an empty 2D array

    # 1. Find the length of the longest array in the list
    # The `max` function with a generator expression is efficient for this.
    max_length = max(len(arr) for arr in list_of_arrays)

    # 2. Create a 2D array of zeros with the correct dimensions
    num_planets = len(list_of_arrays)
    padded_array = np.zeros((num_planets, max_length))

    # 3. Copy the data from the original arrays into the new 2D array
    for i, arr in enumerate(list_of_arrays):
        # Use slicing to place the data from the original array
        # into the corresponding row of the new array.
        padded_array[i, :len(arr)] = arr

    return padded_array

def get_timing_data(main_df, koi_identifier):
    """
    Fetches transit timing data for all confirmed planets in a KOI system.

    For each confirmed planet, it attempts to download and parse a .tt file.

    Args:
        main_df (pd.DataFrame): The main DataFrame with KOI and Disposition data.
        koi_identifier (int): The KOI system number (e.g., 377).

    Returns:
        tuple: A tuple containing:
            - ntt (np.ndarray): 1D array with the count of timing points per planet.
            - tobs_arrays (list): A list of 1D NumPy arrays, one for each planet's tobs.
            - omc_arrays (list): A list of 1D NumPy arrays for omc values.
            - omc_err_arrays (list): A list of 1D NumPy arrays for omc_err values.
        Returns (None, None, None, None) if no confirmed planets are found.
    """
    koi_int = int(koi_identifier)

    # --- Step 1: Find all confirmed planets in the system ---
    lower_bound = koi_int
    upper_bound = koi_int + 1
    
    # Filter for the system and for dispositions that DO NOT start with 'F'
    confirmed_planets = main_df[
        (main_df['KOI'] >= lower_bound) &
        (main_df['KOI'] < upper_bound) &
        (~main_df['Disposition'].str.startswith('F', na=False))
    ].sort_values(by='KOI')

    if confirmed_planets.empty:
        print(f"No confirmed planets found for KOI system {koi_int}.")
        return None, None, None, None
    
    # print(f"Found {len(confirmed_planets)} confirmed planets in system {koi_int}.")

    # --- Step 2: Initialize lists to hold the data for each planet ---
    ntt_list = []
    tobs_list = []
    omc_list = []
    omc_err_list = []

    # --- Step 3: Loop through each confirmed planet and fetch its data ---
    for index, planet_row in confirmed_planets.iterrows():
        # Format the KOI to have 2 decimal places for the filename (e.g., 377.01)
        # The format {value:07.2f} pads with zeros to a total width of 7 (e.g., 0377.01)
        koi_full_str = f"{planet_row['KOI']:07.2f}"
        
        # file_url = f"https://kona.ubishops.ca/Kepler_n/koi{koi_int}.n/koi{koi_full_str}.tt"
        file_url = f"https://kona.ubishops.ca/Kepler_n/timing_DR25/koi{koi_full_str}.tt"
        # print(f"Attempting to fetch: {file_url}")

        try:
            # Read the three space-delimited columns from the URL
            timing_df = pd.read_csv(
                file_url,
                sep=r'\s+',
                engine='python',
                header=None,
                names=['tobs', 'omc', 'omc_err']
            )
            
            # Append the count and the data arrays to our lists
            ntt_list.append(len(timing_df))
            tobs_list.append(timing_df['tobs'].to_numpy())
            omc_list.append(timing_df['omc'].to_numpy())
            omc_err_list.append(timing_df['omc_err'].to_numpy())
            print(f" -> Success! Found {len(timing_df)} timing points.")

        except HTTPError:
            # This block runs if the file was not found (404 error)
            print(f" -> File not found. Setting timing points to 0.")
            ntt_list.append(0)
            # Append empty arrays to maintain structure
            tobs_list.append(np.array([]))
            omc_list.append(np.array([]))
            omc_err_list.append(np.array([]))
        except Exception as e:
            print(f" -> An unexpected error occurred: {e}")
            ntt_list.append(0)
            tobs_list.append(np.array([]))
            omc_list.append(np.array([]))
            omc_err_list.append(np.array([]))

    # --- Step 4: Convert the count list to a final NumPy array ---
    ntt = np.array(ntt_list)

    if ntt is not None:
        # print("--- Original Data (Lists of Arrays) ---")
        # print(f"Number of timing points per planet: {ntt}")
        # # Print the shape of each array in the list to show they are different
        # for i, arr in enumerate(tobs_list):
        #     print(f"  Shape of tobs array for planet {i+1}: {arr.shape}")
    
    
        # --- 2. Convert the lists to padded 2D NumPy arrays ---
        # print("\n--- Converting to Padded 2D NumPy Arrays ---")
        tobs_array_2d = pad_list_of_arrays(tobs_list)
        omc_array_2d = pad_list_of_arrays(omc_list)
        omc_err_array_2d = pad_list_of_arrays(omc_err_list)
    
        # print(f"\nShape of the final padded 'tobs_array_2d': {tobs_array_2d.shape}")
        # print(f"Shape of the final padded 'omc_array_2d': {omc_array_2d.shape}")
        # print(f"Shape of the final padded 'omc_err_array_2d': {omc_err_array_2d.shape}")
    
    return ntt, tobs_array_2d, omc_array_2d, omc_err_array_2d

# Data processing parameters  
class tpy5_inputs_class:
    def __init__(self):
        self.photfile  = "filename.txt"
        self.photdir   = "/path/to/photometry/"
        self.roi       = 210.01 # Roman ID
        self.boxbin    = 2.0    # Detrending window
        self.gapsize   = 0.5    # Detection of gaps in the data -- we do not detrend over gaps
        self.nfitp     = 2      # Order of polynomial for detrending.  2 = quadratic
        self.dsigclip  = 3.0    # Sigma clipping for derivative 
        self.sigclip   = 3.0    # Sigma clipping for simple clipping
        self.boxwindow = 2.0    # Box size for simple clipping [days]
        self.boxsigma  = 3.0    # Box size for simple clipping [days]
        self.nsampmax  = 6      # Sample size for derivative routine
        self.detrended = 0      # Track if detrended data is used/created
        self.dataclip  = 0      # Track if clipped data is used/created 
        self.fstd_cut  = 5      # Simple Sigma-clipping
        self.rjd       = 2461345.5 # time offset for the roman JD used in the simulations
        self.mstar     = 1.0    # Mass of star   [Msun]
        self.rstar     = 1.0    # Radius of star [Rsun]
        self.teff      = 5777   # Temperature of star [K]
        self.logg      = 4.5    # Star gravity [cgs]
        self.feh       = 0.0    # star metalicticity 
        self.e_mstar   = 0.3    # Error in stellar mass [Msun]
        self.e_rstar   = 0.3    # Error in stellar radius [Rsun]
        self.e_teff    = 500    # Error in Teff [K]
        self.e_logg    = 0.2    # Error in log(g) [cgs]
        self.e_feh     = 0.3    # Error in FeH 
        # Iterative detrending parameters
        self.iter_max_iters      = 5    # Maximum iterations for iterative detrending
        self.iter_sigma_threshold = 3.0  # Sigma threshold for transit detection
        self.iter_min_duration    = 0.05 # Minimum transit duration in days (~1 hour)
        self.iter_verbose        = False  # Print iteration progress and statistics

        # Search parameters!
        self.filename = "filename.txt"  # Lightcurve filename
        self.lcdir    = ""              # Lightcurve directory
        self.savedir  = ""              # Save directory (for plots as well)
        self.zerotime = 0.0             # Time offset (e.g., for Kepler/TESS)
        self.freq1    = -1              # Minimum frequency to search [c/d] (auto if -1)
        self.freq2    = -1              # Maximum frequency to search [c/d] (auto if -1)
        self.plots    = 1               # 0 = no plots, 1 = X11, 2 = PNG+X11, 3 = PNG
        # BLS parameters (consolidated from gbls_inputs_class)
        self.ofac     = 8.0             # Oversampling factor
        self.nper     = 50000           # Number of periods (deprecated, auto-calculated)
        self.minbin   = 5               # Minimum number of bins in transit
        self.multipro = 1               # 0 = single thread, 1 = multiprocessing
        self.normalize = "iterative_baseline"  # BLS normalization: none, mad, percentile_mad, coverage_mad, iterative_baseline
        self.return_spectrum = False    # If True, return full BLS spectrum (periods, power, freqs)
        self.oneoverf_correction = True # If True, apply baseline subtraction (else leave raw)
        self.oneoverf_extrapolate = True # If True, extrapolate baseline/noise for periods > threshold
        self.oneoverf_threshold = None   # Period threshold for extrapolation (days). None = use baseline length 
        # Pulse search parameters
        self.pulse_min_duration_hours = 1.0
        self.pulse_max_duration_hours = 12.0
        self.pulse_bin_duration_days = None
        # TLS search parameters
        self.search_verbose = False        # from the transitleastsquares module plus pars check
        self.pars = {}              # if any additional parameters are needed for tls
        self.u = [0.4804, 0.1867]   # the default limb-darkening from transitleastsquares
        self.arstarmin = 1.1        # truncate periods violating this bound
        self.upbuff = 0.2           # maximum mass/radius is 20% of the input
        self.lowbuff = 0.8          # minimum mass/radius is 80% of the input
        self.stthres = 0.095        # if values are lower than this, then:
        self.stbuff = 0.2           # resets the lower/upper buffer to 20%

class exocat_class:
    def __init__(self):
        self.ticid=[]
        self.toiid=[]
        self.toiid_str=[]
        self.ra=[]
        self.dec=[]
        self.tmag=[]
        self.t0=[]
        self.t0err=[]
        self.per=[]
        self.pererr=[]
        self.tdur=[]
        self.tdurerr=[]
        self.tdep=[]
        self.tdeperr=[]

class phot_class:
    def __init__(self):
        self.time=[]   #initialize arrays
        self.flux=[]
        self.ferr=[]
        self.itime=[]
        self.tflag=[]  #in-transit flag
        self.icut=[]   #data cut flag
        self.flux_f=[] #processed flux

    def __getitem__(self, index):
        # 1. Create a fresh instance
        new_obj = phot_class()
        
        # 2. Iterate over all attributes in the current object
        for key, value in self.__dict__.items():
            
            # 3. Check if it is a numpy array that should be sliced
            # (You might want to add a check for length to avoid slicing mismatched arrays)
            if isinstance(value, np.ndarray) and len(value) == len(self.time):
                # Apply the slice/mask
                setattr(new_obj, key, value[index])
            else:
                # Pass through non-array attributes (like names/IDs)
                setattr(new_obj, key, copy.deepcopy(value))
                
        return new_obj

class catalogue_class:
    def __init__(self):
        #IDs
        self.tid=[]  #Catalogue ID [0]
        self.toi=[]  #KOI [1]
        self.planetid=[] #Confirmed planet name (e.g., Kepler-20b)
        #model parameters
        self.rhostarm=[] #rhostar model [31]
        self.rhostarmep=[] #+error in rhostar [32]
        self.rhostarmem=[] #-error in rhostar [33]
        self.t0=[] #model T0 [4]
        self.t0err=[] #error in T0 [5]
        self.per=[] #period [2]
        self.pererr=[] #error in period [3]
        self.b=[] #impact parameter [9]
        self.bep=[] #+error in b [10]
        self.bem=[] #-error in b [11]
        self.rdrs=[] #model r/R* [6]
        self.rdrsep=[] #+error in r/R* [7]
        self.rdrsem=[] #-error in r/R* [8]
        #stellar parameters
        self.rstar=[] #stellar radius [39]
        self.rstar_ep=[] #stellar radius +err [40]
        self.rstar_em=[] #stellar radius -err [41]
        self.teff=[] #Teff [37]
        self.teff_e=[] #Teff error [38]
        self.rhostar=[] #rhostar [34]
        self.rhostar_ep=[] #rhostar +err [35]
        self.rhostar_em=[] #rhostar -err [36]
        self.logg=[] #stellar radius [45]
        self.logg_ep=[] #stellar radius +err [46]
        self.logg_em=[] #stellar radius -err [47]
        self.feh=[] #metallicity [48]
        self.feh_e=[] #metallicity error [49]
        self.q1=[] #limb-darkening
        self.q1_e =[]
        self.q2=[] #limb-darkening
        self.q2_e =[]
        #disposition
        self.statusflag=[]

def safe_float_conversion(s, default=0.0):
    if s is None or s == '':
        #print(f"Input is None or empty. Using default value {default}.")
        return default
    try:
        return float(s)
    except ValueError as e:
        print(f"Error converting to float: {e}")
        return default

# Read 'toi_file' (from NASA EA -- new table)
def readtoicsv(toi_file):
    exocat=exocat_class() #set up class
    f=open(toi_file)

    icount=0
    for line in f:
        line = line.strip()
        row = line.split(',') #break into columns
        if row[0][0]!='#':
            #skip comments
            icount+=1
            if icount>1:
                #skip header
                exocat.ticid.append(int(float(row[2])))
                exocat.toiid.append(float(row[0]))
                exocat.toiid_str.append(row[0])
                exocat.ra.append(safe_float_conversion(row[7]))
                exocat.dec.append(safe_float_conversion(row[11]))
                exocat.tmag.append(float(row[59]))

                if row[25]=='':
                    exocat.t0.append(-1.0)
                else:
                    try:
                        exocat.t0.append(float(row[24]) - 2.457E6) # Planet Transit Midpoint Value [BJD]
                    except:
                        print(row[25])

                if row[26]=='': exocat.t0err.append(-1.0)
                else: exocat.t0err.append(float(row[25])) # Planet Transit Midpoint Upper Unc [BJD]

                if row[30]=='': exocat.per.append(-1.0)
                else: exocat.per.append(float(row[29])) # Planet Orbital Period Value [days]

                if row[31]=='': exocat.pererr.append(-1.0)
                else: exocat.pererr.append(float(row[30])) # Planet Orbital Period Upper Unc [days]

                if row[35]=='': exocat.tdur.append(-1.0)
                else: exocat.tdur.append(float(row[34])) # Planet Transit Duration Value [hours]

                if row[36]=='': exocat.tdurerr.append(-1.0)
                else: exocat.tdurerr.append(float(row[35])) # Planet Transit Duration Upper Unc [hours]

                if row[40]=='': exocat.tdep.append(-1.0)
                else: exocat.tdep.append(float(row[39])) # Planet Transit Depth Value [ppm]

                if row[41]=='': exocat.tdeperr.append(-1.0)
                else: exocat.tdeperr.append(float(row[40])) # Planet Transit Depth Upper Unc [ppm]
    f.close()

    return exocat

def get_tess_data(u_ticid,max_flag=16,out_dir='./download'):
    """Given a TIC-ID, return time,flux,ferr,itime
    u_ticid : (int) TIC ID

    returns lc_time,flux,ferr,int_time
    """
    tic_str='TIC' + str(u_ticid)
    #out_dir='/data/rowe/TESS/download/'

    # Search MAST for TIC ID
    print('Searching MAST for TIC ID (this can be slow)')
    obs_table=Observations.query_object(tic_str,radius=".002 deg")

    # Identify TESS timeseries data sets (i.e. ignore FFIs)
    print("Identifying TESS timeseries data sets")
    oti=(obs_table["obs_collection"] == "TESS") & \
            (obs_table["dataproduct_type"] == "timeseries")
    if oti.any() == True:
        data_products=Observations.get_product_list(obs_table[oti])
        dpi=[j for j, s in enumerate(data_products["productFilename"]) if "lc.fits" in s]
        manifest=Observations.download_products(data_products[dpi],download_dir=out_dir)
    else:
        manifest=[]

    lc_time=[]
    flux=[]
    ferr=[]
    int_time=[]
    for j in range(0,len(manifest)):
        fits_fname=str(manifest["Local Path"][j])
        #print(fits_fname)
        hdu=fits.open(fits_fname)
        tmp_bjd=hdu[1].data['TIME']
        tmp_flux=hdu[1].data['PDCSAP_FLUX']
        tmp_ferr=hdu[1].data['PDCSAP_FLUX_ERR']
        tmp_int_time=hdu[1].header['INT_TIME'] + np.zeros(len(tmp_bjd))
        tmp_flag=hdu[1].data['QUALITY']

        ii=(tmp_flag <= max_flag) & (~np.isnan(tmp_flux))
        tmp_bjd=tmp_bjd[ii]
        tmp_flux=tmp_flux[ii]
        tmp_ferr=tmp_ferr[ii]
        tmp_int_time=tmp_int_time[ii]
        # Shift flux measurements
        median_flux=np.median(tmp_flux)
        tmp_flux=tmp_flux / median_flux
        tmp_ferr=tmp_ferr / median_flux
        # Append to output columns
        lc_time=np.append(lc_time,tmp_bjd)
        flux=np.append(flux,tmp_flux)
        ferr=np.append(ferr,tmp_ferr)
        int_time=np.append(int_time,tmp_int_time)

        hdu.close()

    # Sort by time
    si=np.argsort(lc_time)
    lc_time=np.asarray(lc_time)[si]
    flux=np.asarray(flux)[si]
    ferr=np.asarray(ferr)[si]
    int_time=np.asarray(int_time)[si]

    phot=phot_class()
    phot.time=np.copy(lc_time)
    phot.flux=np.copy(flux)
    phot.ferr=np.copy(ferr)
    phot.itime=np.copy(int_time)    

    phot.itime=phot.itime/(60*24) #convert minutes to days

    phot.tflag = np.zeros((phot.time.shape[0])) # pre-populate array to mark transit data (=1 when in transit)
    phot.flux_f = []                            # placeholder for detrended data
    phot.icut = np.zeros((phot.time.shape[0]))  # cutting data (0==keep, 1==toss)

    return phot

def mastQuery(request):

    server='mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
        "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head,content

def ticAdvancedSearch(id):
    request = {"service":"Mast.Catalogs.Filtered.Tic",
                "format":"json",
                "params":{
                "columns":"*",
                "filters":[
                    {"paramName":"id",
                        "values":[{"min":id,"max":id}]}]
                        #"values":[{261136679}]}]
                }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData

def populate_catalogue(tic_output, exocat, toi_index):

    koicat = catalogue_class()
    
    koicat.tid.append(exocat.ticid[toi_index])
    koicat.toi.append(exocat.toiid[toi_index])
    koicat.planetid.append(" ")
    
    koicat.t0.append(exocat.t0[toi_index])
    koicat.t0err.append(exocat.t0err[toi_index])
    
    koicat.per.append(exocat.per[toi_index])
    koicat.pererr.append(exocat.pererr[toi_index])
    
    koicat.b.append(0.4)   #This is a guess because it is not populated.
    koicat.bep.append(0.1)
    koicat.bem.append(-0.1)
    
    koicat.rdrs.append(np.sqrt(exocat.tdep[toi_index]/1.0e6))
    koicat.rdrsep.append(0.001)
    koicat.rdrsem.append(-0.001)
    
    if tic_output['data'][0]['rad'] == None:
        print('Warning: No R* Available, using Sun')
        koicat.rstar.append(1)
        koicat.rstar_ep.append(0.5)
        koicat.rstar_em.append(0.5)
    else:
        koicat.rstar.append(tic_output['data'][0]['rad'])
        e_rad = tic_output['data'][0]['e_rad'] if tic_output['data'][0]['e_rad'] is not None else 0
        koicat.rstar_ep.append(e_rad)
        koicat.rstar_em.append(-e_rad)
    
    if tic_output['data'][0]['Teff'] == None:
        print('Warning: No Teff Available, using Sun')
        koicat.teff.append(5777)
        koicat.teff_e.append(500)
    else:
        koicat.teff.append(tic_output['data'][0]['Teff'])
        koicat.teff_e.append(tic_output['data'][0]['e_Teff'])
    
    if tic_output['data'][0]['rho'] == None:
        print('Warning: No rhostar Available, using 1.0')
        koicat.rhostar.append(1.0)
        koicat.rhostar_ep.append(0.5)
        koicat.rhostar_em.append(-0.5)
        koicat.rhostarm.append(1.0)
        koicat.rhostarmep.append(0.1)
        koicat.rhostarmem.append(-0.1)
    else:
        koicat.rhostar.append(tic_output['data'][0]['rho'])
        e_rho = tic_output['data'][0]['e_rho'] if tic_output['data'][0]['e_rho'] is not None else 0
        koicat.rhostar_ep.append(e_rho)
        koicat.rhostar_em.append(-e_rho)
        koicat.rhostarm.append(tic_output['data'][0]['rho'])
        koicat.rhostarmep.append(0.1)
        koicat.rhostarmem.append(-0.1)
    
    if tic_output['data'][0]['logg'] == None:
        koicat.logg.append(4.5)
        koicat.logg_ep.append(0.5)
        koicat.logg_em.append(-0.5)
        print('Warning: No log(g) Available, using 4.5')
    else:
        koicat.logg.append(tic_output['data'][0]['logg'])
        e_logg = tic_output['data'][0]['e_logg'] if tic_output['data'][0]['e_logg'] is not None else 0
        koicat.logg_ep.append(e_logg)
        koicat.logg_em.append(-e_logg)
    
    koicat.feh.append(0.0)
    koicat.feh_e.append(1.0)
    
    koicat.q1.append(0.5)
    koicat.q1_e.append(0.5)
    
    koicat.q2.append(0.5)
    koicat.q2_e.append(0.5)
    
    koicat.statusflag.append('P')

    return koicat

def get_data_and_catalogues(tpy5_inputs):

    exocat=readtoicsv(tpy5_inputs.toifile)
    toi_index = [j for j, x in enumerate(exocat.toiid) if x == tpy5_inputs.toi][0]
    print('TIC ID: ',exocat.ticid[toi_index],'| TOI ID: ',exocat.toiid[toi_index])
    
    #Get SC Lightcurve for MAST
    #Each Sector/Quarter of data is median corrected independently, then concatentated together.
    phot_SC=get_tess_data(exocat.ticid[toi_index])  #give the TIC_ID and return SC data.

    #Get Stellar parameters from MAST
    tic_output = ticAdvancedSearch(exocat.ticid[toi_index])

    toicat = populate_catalogue(tic_output, exocat, toi_index)

    return toi_index, phot_SC, toicat

def time_window_clip(phot, tpy5_inputs, max_iters=5):
    """
    Performs iterative sigma clipping using a rolling time window.
    
    Parameters
    ----------
    phot : phot_class
        The object containing .time, .flux_f, and .icut
    window_width : float
        The width of the window in the same units as phot.time (usually days).
    threshold : float
        Number of standard deviations to clip.
    max_iters : int
        Maximum number of clipping iterations to run.
        
    Returns
    -------
    phot : phot_class
        The object with updated .icut (modified in-place).
    """

    window_width = tpy5_inputs.boxwindow
    threshold = tpy5_inputs.boxsigma

    # 1. Create a DataFrame for easy time handling
    df = pd.DataFrame({
        'time': phot.time,
        'flux': phot.flux_f,
        'icut': phot.icut
    })

    # 2. Convert time to a DatetimeIndex
    # Pandas needs datetime objects to do time-based rolling. 
    # We treat the float time as "days" relative to a dummy epoch.
    df.index = pd.to_datetime(df['time'], unit='D', origin='unix')

    # Convert window_width (float days) to a string format Pandas understands
    window_str = f"{window_width}D"

    print(f"Starting clipping: Window={window_width} days, Sigma={threshold}")

    for i in range(max_iters):
        # Count currently rejected points
        n_bad_start = df['icut'].sum()

        # 3. Mask bad data for the calculation
        # valid_flux is a Series where icut=1 values are set to NaN
        valid_flux = df['flux'].where(df['icut'] == 0)

        # 4. Calculate Rolling Statistics
        roller = valid_flux.rolling(window=window_str, center=True, min_periods=1)
        
        roll_median = roller.median()
        roll_std = roller.std()

        # 5. Identify Outliers
        # FIX: Use .bfill() and .ffill() instead of deprecated fillna(method=...)
        roll_std = roll_std.bfill().ffill()

        # Calculate deviation from local median
        deviation = np.abs(df['flux'] - roll_median)
        
        # Find points that exceed threshold
        # fillna(False) is still valid because we are providing a value, not a method
        is_outlier = (deviation > (threshold * roll_std)).fillna(False)

        # 6. Update the Mask
        # We perform a bitwise OR to keep previously flagged bad points
        new_mask = df['icut'] | is_outlier.astype(int)
        
        # Check if we found anything new
        n_bad_end = new_mask.sum()
        diff = n_bad_end - n_bad_start
        
        df['icut'] = new_mask
        
        print(f"  Iteration {i+1}: flagged {diff} new outliers.")
        
        if diff == 0:
            break

    # 7. Write results back to the class object
    phot.icut = df['icut'].values

def calc_meddiff(npt, x):
    """
    Used by cutoutliers to calculation distribution of derivatives
    """

    dd = np.zeros(npt - 1)

    for i in range(npt - 1):
        dd[i] = np.abs(x[i] - x[i+1])

    meddiff = np.median(dd)
    # p = np.argsort(dd)
    # meddiff =dd[p[int((npt-1)/2)]]

    return meddiff

def run_sigclip_global(phot, tpy5_inputs):
    """
    Simple sigma-clipping of data
    """

    med = np.median(phot.flux_f[(phot.icut==0)])
    std = np.std(phot.flux_f[(phot.icut==0)])
    phot.icut[np.abs(phot.flux_f - med) > tpy5_inputs.sigclip * std] = 1
    tpy5_inputs.dataclip = 1

def run_cutoutliers(phot, tpy5_inputs):

    phot.icut = np.maximum(cutoutliers(phot.time, phot.flux, tpy5_inputs.nsampmax, tpy5_inputs.dsigclip), phot.icut)
    tpy5_inputs.dataclip = 1 
    
def cutoutliers(x, y, nsampmax = 3, sigma = 3.0):
    """
    Uses derivatives to cut outliers
    """

    threshold = 0.0005 #Fixed threshold, if sigma <= 0
    # nsampmax  = 3      #number of +/- nearby samples to use for stats
    # sigma     = 3.0    #sigma cut level

    npt   = x.shape[0]
    icut  = np.zeros((npt), dtype = np.int32)
    samps = np.zeros((nsampmax * 2 + 1))

    for i in range(1,npt-1):

        i1 = np.max((0,       i - nsampmax))
        i2 = np.min((npt - 1, i + nsampmax))

        nsamp = i2 - i1 + 1
        samps[0:nsamp] = y[i1:i2+1]
        # print(i1, i2, samps)

        std = np.median(np.abs(np.diff(samps)))

        if sigma > 0:
            threshold = std * sigma

        vp = y[i] - y[i+1]
        vm = y[i] - y[i-1]

        if  (np.abs(vp) > threshold) and (np.abs(vm) > threshold) and (vp/vm > 0):
            icut[i] = 1 #cut

    return icut

def run_polyfilter(phot_SC, tpy5_inputs):

    phot_SC.flux_f = polyfilter(phot_SC.time, phot_SC.flux, phot_SC.ferr, phot_SC.tflag, \
                                tpy5_inputs.boxbin, tpy5_inputs.nfitp, tpy5_inputs.gapsize)

    tpy5_inputs.detrended = 1 # Mark that we have detrended data


def run_polyfilter_iterative(phot, tpy5_inputs):
    """
    Iterative polynomial detrending that automatically masks transit-like features.
    
    This function performs iterative detrending to avoid distorting transit signals:
    1. Run initial detrending with current tflag mask
    2. Identify significant negative deviations (potential transits)
    3. Update tflag to mask detected transit-like features
    4. Re-run detrending with updated mask
    5. Iterate until convergence or max iterations reached
    
    Parameters
    ----------
    phot : phot_class
        Photometry object with time, flux, ferr, tflag
    tpy5_inputs : tpy5_inputs_class
        Input parameters (boxbin, nfitp, gapsize, iter_max_iters, iter_sigma_threshold, 
        iter_min_duration, iter_verbose)
        
    Returns
    -------
    None (modifies phot.flux_f and phot.tflag in place)
    """
    
    # Get parameters from tpy5_inputs class
    max_iters = tpy5_inputs.iter_max_iters
    sigma_threshold = tpy5_inputs.iter_sigma_threshold
    min_duration = tpy5_inputs.iter_min_duration
    verbose = tpy5_inputs.iter_verbose
    
    # Initialize tflag if not already set
    if not hasattr(phot, 'tflag') or phot.tflag is None or len(phot.tflag) == 0:
        phot.tflag = np.zeros(len(phot.time))
    
    # Store original tflag to preserve user-specified transit masks
    tflag_original = np.copy(phot.tflag)
    
    if verbose:
        print(f"Starting iterative detrending (max {max_iters} iterations)")
        print(f"  Sigma threshold: {sigma_threshold:.1f}")
        print(f"  Min duration: {min_duration*24:.1f} hours")
        print(f"  Initially masked points: {np.sum(tflag_original)}")
    
    for iteration in range(max_iters):
        # Run detrending with current mask
        phot.flux_f = polyfilter(phot.time, phot.flux, phot.ferr, phot.tflag, 
                                 tpy5_inputs.boxbin, tpy5_inputs.nfitp, tpy5_inputs.gapsize)
        
        # Calculate residuals
        residuals = phot.flux_f - np.median(phot.flux_f)
        
        # Calculate local noise level (use only non-masked points)
        unmasked = (phot.tflag == 0)
        if np.sum(unmasked) < 10:
            if verbose:
                print(f"  Iteration {iteration+1}: Too few unmasked points, stopping")
            break
            
        sigma = np.std(residuals[unmasked])
        
        # Identify significant negative deviations (potential transits)
        # Only look at currently unmasked points
        is_negative_outlier = (residuals < -sigma_threshold * sigma) & unmasked
        
        if np.sum(is_negative_outlier) == 0:
            if verbose:
                print(f"  Iteration {iteration+1}: No new transit-like features detected, converged")
            break
        
        # Group consecutive outlier points to identify transit events
        # This helps distinguish real transits from single-point outliers
        transit_mask = np.zeros(len(phot.time), dtype=bool)
        
        # Sort by time to identify consecutive groups
        time_sorted_idx = np.argsort(phot.time)
        outlier_sorted = is_negative_outlier[time_sorted_idx]
        time_sorted = phot.time[time_sorted_idx]
        
        # Find groups of consecutive outliers
        in_group = False
        group_start = 0
        
        for i in range(len(outlier_sorted)):
            if outlier_sorted[i] and not in_group:
                # Start of new group
                in_group = True
                group_start = i
            elif not outlier_sorted[i] and in_group:
                # End of group
                group_end = i - 1
                group_duration = time_sorted[group_end] - time_sorted[group_start]
                
                # Only mask if duration exceeds minimum
                if group_duration >= min_duration:
                    # Mask this group plus a small buffer
                    buffer_time = min_duration * 0.5
                    t_start = time_sorted[group_start] - buffer_time
                    t_end = time_sorted[group_end] + buffer_time
                    transit_mask = transit_mask | ((phot.time >= t_start) & (phot.time <= t_end))
                
                in_group = False
        
        # Handle case where group extends to end of data
        if in_group:
            group_end = len(outlier_sorted) - 1
            group_duration = time_sorted[group_end] - time_sorted[group_start]
            if group_duration >= min_duration:
                buffer_time = min_duration * 0.5
                t_start = time_sorted[group_start] - buffer_time
                t_end = time_sorted[group_end] + buffer_time
                transit_mask = transit_mask | ((phot.time >= t_start) & (phot.time <= t_end))
        
        # Check if we found new transit-like features
        new_masked = np.sum(transit_mask & ~phot.tflag.astype(bool))
        
        if new_masked == 0:
            if verbose:
                print(f"  Iteration {iteration+1}: No new transit events with sufficient duration, converged")
            break
        
        # Update tflag with newly detected transits
        phot.tflag = (phot.tflag.astype(bool) | transit_mask).astype(int)
        
        if verbose:
            print(f"  Iteration {iteration+1}: Masked {new_masked} new points (total masked: {np.sum(phot.tflag)})")
    
    # Final detrending pass with complete mask
    phot.flux_f = polyfilter(phot.time, phot.flux, phot.ferr, phot.tflag, 
                             tpy5_inputs.boxbin, tpy5_inputs.nfitp, tpy5_inputs.gapsize)
    
    tpy5_inputs.detrended = 1
    
    if verbose:
        print(f"Iterative detrending complete")
        print(f"  Final masked points: {np.sum(phot.tflag)} (original: {np.sum(tflag_original)})")
        print(f"  Newly detected transit points: {np.sum(phot.tflag) - np.sum(tflag_original)}")


def polyfilter(time, flux, ferr, tflag, boxbin, nfitp, gapsize, multipro = 1):
    """
    Computes detended data based on Kepler TRANSITFIT5 routines

    Parameters:
    time (np.array) : times series, usually in days
    flux (np.array) : relative flux
    ferr (np.array) : uncertanity on relative flux
    boxbin (float)  : full width of detrending window.  Units should match time.
    nfitp  (int)    : order of polynomial fit
    gapsize (float) : identifying boundaries to not detrend across.  Units should match time.
    multipro (int)  : if == 0, single thread operation.  if == 1, multithreading.

    Returns:
    flux_f (np.array) : detrended time-series
    """

    bbd2=boxbin/2.0  #pre-calculate the half-box width

    ts = np.argsort(time) #get indicies for data (we don't assume it's sorted) 

    ngap = 0
    gaps = [] 
    npt = time.shape[0]
    
    for i in range(npt-1):
        if time[ts[i+1]] - time[ts[i]] > gapsize:
            ngap += 1
            g1 = (time[ts[i+1]] + time[ts[i]])/2.0
            gaps.append(g1)
    gaps = np.array(gaps)

    # print("Number of gaps detected: ", ngap)

    if multipro == 0:
        # Single thread version
        offset = np.zeros((npt))
        for i in range(npt):
            offset[ts[i]] = polyrange(ts[i], time, flux, ferr, tflag, nfitp, ngap, gaps, bbd2)

    else:
    
        max_processes = os.cpu_count()
        ndiv = npt // (max_processes - 1) + 1
        offset = np.zeros((max_processes, ndiv))
    
        iarg = np.zeros((max_processes, ndiv), dtype = int)
        for i in range(0, max_processes):
            for k,j in enumerate(range(i, npt, max_processes)):
                iarg[i, k] = j
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
                futures = {executor.submit(compute_polyrange, iarg[i], time, flux, ferr, tflag, nfitp, \
                                           ngap, gaps, bbd2, ndiv): i for i in range(iarg.shape[0])}
                
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        result = future.result()
                        offset[i] = result
                    except Exception as exc:
                        print(f'Generated an exception: {exc}')
        
        offset = offset.T.ravel()[0:npt]
    
    flux_f = flux - offset + np.median(flux)

    return flux_f

def compute_polyrange(iarg, time, flux, ferr, tflag, nfitp, ngap, gaps, bbd2, ndiv):

    offset = np.zeros((ndiv))

    j = 0
    for i in iarg:
        offset[j] = polyrange(i, time, flux, ferr, tflag, nfitp, ngap, gaps, bbd2)
        j += 1

    return offset

def polyrange(i, time, flux, ferr, tflag, nfitp, ngap, gaps, bbd2):
    """
    Computes mask for time series
    """

    tzero = time[i]
    t1 = tzero - bbd2
    t2 = tzero + bbd2

    #Check if there is a gap in our time-span
    for j in range(ngap):
        if (gaps[j] > t1) and (gaps[j] < tzero):
            t1 = gaps[j]
        if (gaps[j] < t2) and (gaps[j] > tzero):
            t2 = gaps[j]

    #Get data inside time-span that is not a transit.
    tmask = (time > t1) & (time < t2) & (tflag == 0)

    x = time[tmask]
    y = flux[tmask]
    z = ferr[tmask]

    npt2 = x.shape[0]
    # print(t1, t2, npt2, tzero)

    if npt2 > nfitp + 1:
        offset1 = polydetrend(x, y, z, nfitp, tzero)
    elif npt2 == 0:
        offset1 = 0.0
    else:
        offset1 = np.mean(y)
    
    
    return offset1

def polydetrend(x, y, z, nfitp, x_c):

    # Make these command-line parameters
    maxiter = 10
    sigcut = 4.0

    x_centred = x - x_c
    
    ans = np.polyfit(x_centred, y, nfitp , w=1/z)
    poly_func = np.poly1d(ans)
    y_pred = poly_func(x_centred)
    chisq = np.sum(((y - y_pred) / z) ** 2)

    offset = poly_func(0.0)

    # Iteratively remove outliers and refit
    ii = 0       #count the number of iterations
    dchi = 1     #we will interate to reduce chi-sq
    ochi = chisq 
    
    while (ii < maxiter) and (dchi > 0.1):
        model = poly_func(x_centred)
        residuals = y - model
        
        std = np.std(residuals)
        mask = np.abs(residuals) < sigcut * std
        x_centred_filtered = x_centred[mask]
        y_filtered = y[mask]
        z_filtered = z[mask]
        
        coeffs = np.polyfit(x_centred_filtered, y_filtered, nfitp, w=1/z_filtered)

        # Create a polynomial function from the coefficients
        poly_func = np.poly1d(coeffs)
        
        # Calculate predicted values
        y_pred = poly_func(x_centred_filtered)

        # Calculate offset (this is removed from the data)
        offset = poly_func(0.0)
        
        # Calculate chi-squared
        chisq = np.sum(((y_filtered - y_pred) / z_filtered) ** 2)
        
        dchi = abs(chisq - ochi)
        ochi = chisq
        
        ii += 1

    return offset

#generic routine to read in photometry
def readphot(filename):

    itime1 = 1765.5/86400.0 #Kepler integration time
    
    i=0

    phot = phot_class()
    
    f=open(filename)
    for line in f:
        if i>0:
            line = line.strip() #removes line breaks 
            columns = line.split() #break into columns based on a delimiter.  This example is for spaces
            phot.time.append(float(columns[0]))
            phot.flux.append(float(columns[1]))
            if len(columns) >= 3:
                phot.ferr.append(float(columns[2]))
            if len(columns) >= 4:
                phot.itime.append(float(columns[3]))
        i+=1
    f.close()
    
    phot.time = np.array(phot.time)
    phot.flux = np.array(phot.flux)
    if len(phot.ferr) > 0:
        phot.ferr = np.array(phot.ferr)
    else:
        phot.ferr = np.ones((phot.flux.shape[0]))*np.std(phot.flux)
    if len(phot.itime) > 0:
        phot.itime = np.array(phot.itime)
    else:
        phot.itime = np.median(np.diff(phot.time))*np.ones((phot.time.shape[0]))

    phot.tflag = np.zeros((phot.time.shape[0])) # pre-populate array to mark transit data (=1 when in transit)
    phot.flux_f = []                            # placeholder for detrended data
    phot.icut = np.zeros((phot.time.shape[0]))  # cutting data (0==keep, 1==toss)
    
    return phot

#generic routine to read in files
def readbaddata(filename):
    
    i=0
    #data=[] #Create an empty list
    bad_cadence  = []
    bad_time  = []
    # ferr  = []
    # itime = []
    f=open(filename)
    for line in f:
        if i>0:
            line = line.strip() #removes line breaks 
            columns = line.split() #break into columns based on a delimiter.  This example is for spaces
            bad_cadence.append(int(columns[0]))
            bad_time.append(float(columns[1]))

        i+=1
    f.close()

    bad_cadence = np.array(bad_cadence)
    bad_time = np.array(bad_time)
    
    return bad_cadence, bad_time

# Find the closest values
def find_closest(array_list, array_values):
    # Reshape array_list to a column vector for broadcasting
    diff_matrix = np.abs(array_list[:, None] - array_values)
    
    # Get indices of minimum differences along axis 1 (columns)
    closest_indices = np.argmin(diff_matrix, axis=1)
    bad_diffs = array_list - array_values[closest_indices]

    return closest_indices, bad_diffs


def checkperT0(samples,burninfrac,nthin,sol,serr):
    
    #get indices of Period and T0 from chain
    nparsol=len(sol)
    j=-1
    iT0=-1
    iPer=-1
    for i in range(nparsol):
        if np.abs(serr[i])>1.0e-10:
            j+=1
            if i==8:
                iT0=j
            if i==9:
                iPer=j
                
    #print('ii',iT0,iPer)
    
    nburnin=len(samples)*burninfrac
    burnin=int(nburnin/nthin) #correct burnin using nthin.

    chain=np.array(samples)
    chain=chain[::nthin,:] #Thin out chain.
    #print('Thin size: ',len(chain))
    #print('Burnin: ',burnin)

    if burnin > 0:
        burnin_bak=np.copy(burnin)
    else:
        burnin=np.copy(burnin_bak)
    sigcut=4
    niter=3
    for k in range(niter):
        npars=chain.shape[1]
        test=np.copy(chain[burnin:,:])
        for i in range(npars):
            nch=test.shape[0]
            #print(nch)
            mean=np.mean(test[:,i])
            std=np.std(test[:,i])
            #print(mean,std)
            if std>0.0:
                test2=[]
                for j in range(nch):
                    #print(test[j,i], np.abs(test[j,i]-mean),std*sigcut)
                    #input()
                    if np.abs(test[j,i]-mean) < sigcut*std:
                        test2.append(test[j,:])
                test=np.array(test2)
        nch=test.shape[0]
        #print("nchains:",nch)
        chain=np.copy(test)
        burnin=0
    
    
    if iT0>-1:
        mode,x_eval,kde1=modekdestimate(chain[:,iT0],burnin)
        perc1 = intperc(mode,x_eval,kde1)
        t0_ep=np.abs(perc1[1]-mode)
        t0_em=np.abs(mode-perc1[0])
    else:
        t0_ep=0.0
        t0_em=0.0
    
    if iPer>-1:
        mode,x_eval,kde1=modekdestimate(chain[:,iPer],burnin)
        perc1 = intperc(mode,x_eval,kde1)
        per_ep=np.abs(perc1[1]-mode)
        per_em=np.abs(mode-perc1[0])
    else:
        per_ep=0.0
        per_em=0.0
        
    return t0_ep,t0_em,per_ep,per_em

def modekdestimate(chain,burnin):
    'Estimate Mode with KDE and return KDE'
    #range of data
    minx=np.min(chain[burnin:])
    maxx=np.max(chain[burnin:])
    x_eval = np.linspace(minx, maxx, num=1000)
    kde1 = stats.gaussian_kde(chain[burnin:])#,0.3)
    modeval=[]
    modekde=0
    for x in x_eval:
        if kde1(x) > modekde:
            modekde=kde1(x)
            modeval=x
    return modeval,x_eval,kde1

def intperc(x,x_eval,kde1,perc=0.6827):
    'find error bounds'
    idx = (np.abs(x_eval-x)).argmin()
    kdea=np.array(kde1(x_eval))

    n=len(x_eval)

    #print(x,idx)

    i1=1
    i2=1
    intval=0.0

    j1=np.copy(idx)
    j2=np.copy(idx)
    j1old=np.copy(j1)
    j2old=np.copy(j2)
    while intval < perc:
        j1test=np.max((0,idx-i1-1))
        j2test=np.min((n-1,idx+i2+1))
        if kdea[j1test] > kdea[j2test]:
            if j1test>0:
                j1=np.copy(j1test)
                i1=i1+1
            else:
                j1=np.copy(j1test)
                j2=np.copy(j2test)
                i2=i2+1
            #print('case1')
        else:
            if j2test<n-1:
                j2=np.copy(j2test)
                i2=i2+1
            else:
                j2=np.copy(j2test)
                j1=np.copy(j1test)
                i1=i1+1
            #print('case2')

        intval=np.trapz(kdea[j1:j2],x_eval[j1:j2])
        #print(j1,j2,intval,kdea[j1test],kdea[j2test])

        #make sure we can break from loop
        if (j1 == 0) and (j2 == n-1):  #break we reach boundaries of array
            #print('break1')
            intval=1.0
        if (j1 == j1old) and (j2 == j2old): #break if stuck in loop.
            #print('break2')
            intval=1.0

        #Update old values to check we are making progress.
        j1old=np.copy(j1)
        j2old=np.copy(j2)

    #print(x_eval[j1],x_eval[j2])
    return x_eval[j1],x_eval[j2];




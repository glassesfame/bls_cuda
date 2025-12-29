import sys
from pathlib import Path
funcdir = str((Path(__file__).resolve()).parent)
homedir = str(Path(funcdir).parent.absolute())
datadir = lambda file: f'{homedir}/data/{file}'

import h5py
import timeit
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, f'{homedir}')
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5
import pytfit5.transitmodel as transitm

## CONSTANTS
ROMANOFF = 2461450
SAVEDIR = f'{homedir}/data/output'
LCDIR = f'{homedir}/data/rlc' # lcdir only contains a few sample lightcurves!
COLS = np.array(['RIC', 'Period', 'T0', 'TDur', 'TDepth', \
                 'RawPer', 'SR', 'Power', 'SNR', 'Time'])
starID = np.loadtxt(datadir('rlc/rand42ric.txt')).astype(int)
cat = pd.read_csv(datadir('trunccat.csv')) # trimmed planet catalogue!

def read2Phot(filepath, phot=None):
    '''
    Load in the data from the saved directory as defined by lcDir.
    And then, read the data into the phot class bc this is the format
    required to run the data processing functions.
    '''
    if phot == None:
        # Initialising the photometry class required for data-processing.
        phot = tpy5.phot_class()
        # Load in our time, flux, error arrays from the h5 file. 
        with h5py.File(filepath, 'r') as file:
            phot.time = file['time'][:]
            phot.flux = file['flux'][:]
            phot.ferr = file['ferr'][:]
            
    npt = len(phot.time)
    # Parameters required to store the processing arrays in.
    phot.itime = np.ones(npt) * 0.00063333 # Convert from seconds to days 
    phot.qflag = np.ones(npt)
    phot.tflag = np.zeros(npt)  # Flag for in-transit data
    phot.icut  = np.zeros(npt)  # Flat for data cuts
    
    return phot

def processData(phot, tpy5_inputs):
    '''
    Running the data-processing. Note that the detrending will always happen
    but the clipping is optional based on the previously assigned sigma-clipping.
    '''
    tpy5.run_polyfilter_iterative(phot, tpy5_inputs) # data-processing
    if tpy5_inputs.dsigclip != 0:  # data (sigma) clipping, phot.icut will flag bad data
        tpy5.run_cutoutliers(phot, tpy5_inputs)
    return (phot.icut == 0) & (phot.tflag == 0)

def getRICRow(ric, df):
    '''
    For a given star id (or roman id), we find the number of entries.
    '''
    mask = np.isin(df.star_id, ric)
    ricIdx = np.where(mask)[0][0]
    # take one row for clarity as all stellar parameters are the same.
    ricDF = df.iloc[ricIdx]
        
    return df.iloc[ricIdx]

def getROIRow(roi, df):
    '''
    Getting the roi row, which works because we only expect one row for
    each ROI.
    '''
    roidf = df[df.planet_id == roi]
    return roidf.iloc[0]

# To be able to swiss-cheese in the loop function.
getPhase1 = lambda time, stats: (time - stats[1])/stats[0]
def getTransArr(time, stats, buff=0.2):
    '''
    Where the planet should be transiting, the transiting array indicates
    this with a 1.0. If it is not transiting, then transiting is 0.0.
    '''
    transiting = np.zeros(len(time))
    phase1 = getPhase1(time, stats)
    phase = phase1 - np.floor(phase1)
    phase[phase > 0.5] -= 1.0
    phase[phase < -0.5] += 1.0
    transThres = (stats[2]/stats[0]/2.0) * (1+buff)
    transMask = np.logical_and(phase < transThres, phase > -transThres)
    transiting[transMask] = 1.0

    return transiting

def swissCheese(t, f, stats, buff):
    '''
    Getting the data swiss-cheesed with the statistics of:
    period, t0 (centre of transit) and the transit duration!
    '''
    t0 = t-ROMANOFF 
    mask = getTransArr(t0, stats, buff=buff)
    return ~np.array(mask).astype(bool)

def loop(t5puts, t, f, err, func, buff):
    '''
    Controls the while loop of tls or bls and returns an array of all the values.
    We get the stellar parameters 
    '''
    it = 0
    ricRun = [] # Unknown length as we begin
    stats = [10] # Arbitrarily setting the SNR to proceed.

    while np.abs(stats[-1]) > 6 and it < 10:
    # Looping until we do not see a signal or the baseline of 10.
        t1 = timeit.default_timer()

        if func == gbls.tls:
            ansob = func(t5puts, t, f, err)
        else: # we can run bls_pulse or bls
            ansob = func(t5puts, t, f)
        t2 = timeit.default_timer()
        stats = np.array([ansob.bper, ansob.epo, ansob.tdur, ansob.depth, \
                          ansob.rawper, ansob.SR, ansob.bpower, ansob.snr])
        ricRun.append(np.append(stats, t2-t1))
        # Require the time, the flux and the error be trimmed.
        scm = swissCheese(t, f, stats[:3], buff=buff)
        t, f, err = t[scm], f[scm], err[scm]
        it += 1

    return np.array(ricRun)

def writedf(ric, arr):
    '''
    Writing the ric run output into a dataframe. This is appended to the frames list
    and all these dataframes are then all put together.
    '''
    stdata = np.tile([int(ric)], (len(arr), 1))
    data = np.append(stdata, arr, axis=1)
    return pd.DataFrame(columns=COLS, data=data)

def save2csv(savename, frame):
    '''
    If the frame is empty, then we simply return it. If it is not, then we concat all
    the specific ric runs and save everything together. 
    '''
    if not frame: return frame 
    df = pd.concat(frame, ignore_index=True)
    df.to_csv(savename, index=False)
    return []

def main(t5puts, rics, df, runB=True, runT=True, tbuff=0.25, bbuff=0.05, saveIt=150):
    '''
    '''
    # Configure the save mechanism and the light-curve directory.
    if not t5puts.lcdir:
        t5puts.lcdir = LCDIR
    if not t5puts.savedir:
        t5puts.savedir = lambda method, num: f'{SAVEDIR}/{t5puts.filename}{method}{num}.csv'
        
    if isinstance(t5puts.savedir, str): # Ensuring nothing will go wrong with saving.
        savedir = t5puts.savedir.strip('/')
        t5puts.savedir = lambda file: f'/{savedir}/{file}'
        
    tlsFrames, blsFrames = [], []
    saveNum = 0
    
    for i, ric in enumerate(rics):
        print(f'Starting RIC: {ric}, which is number: {i}!')

        # Loading in the photometric data and data processing.
        phot = read2Phot(f'{t5puts.lcdir}/{int(ric)}raw.h5') # getting the photometric data
        m = processData(phot, t5puts) # detrending/clipping
        t, f, err = phot.time[m], phot.flux_f[m], phot.ferr[m]
        ## The stellar parameters which are unique to each RIC.
        starRow = getRICRow(ric, df)
        t5puts.rstar = float(starRow['star_radius'])
        t5puts.mstar = float(starRow['star_mass'])
        t5puts.u = starRow[['transit_limb1_F146', 'transit_limb2_F146']].values
        
        if runT: ## Running TLS
            tarr = loop(t5puts, np.copy(t), np.copy(f), np.copy(err), gbls.tls, tbuff)
            tlsFrames.append(writedf(ric, tarr))
        if runB: ## Running BLS
            barr = loop(t5puts, np.copy(t), np.copy(f), np.copy(err), gbls.bls_pulse, bbuff)
            blsFrames.append(writedf(ric, barr))

        ## Saving if necessary!
        if max(len(tlsFrames), len(blsFrames)) >= saveIt:
            # Checking with max because len(tlsFrame) == len(blsFrame) if both are run
            print(f'This is save no. {saveNum}.')
            tlsFrames = save2csv(t5puts.savedir('tls', saveNum), tlsFrames)
            blsFrames = save2csv(t5puts.savedir('bls', saveNum), blsFrames)
            saveNum += 1 # resetting the frames to zero so increment count.

    # To account for the last save if it is not a round number          
    print(f'Finished! Saving (just in case) again.')
    tlsFrames = save2csv(t5puts.savedir('tls', saveNum), tlsFrames)
    blsFrames = save2csv(t5puts.savedir('bls', saveNum), blsFrames)

## PLOTTING FUNCTIONALITIES
def plotSpecs(ax, logscale):
    '''
    Specifications for the tick parameters and for the logscale.
    logscale: can be an integer or a list.
    '''
    ax.tick_params(direction='in', which='major', bottom=True, top=True, \
                   left=True, right=True, length=10, width=2)
    ax.tick_params(direction='in', which='minor', bottom=True, top=True,  \
                   left=True, right=True, length=4, width=2)

    if logscale==1:
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif logscale==2:
        ax.set_xscale('log')
    elif logscale==3:
        ax.set_yscale('log')
        
    return ax
    
def setupPlot(sizeTuple, logscale=1, fontsize=16, scalar=True):
    '''
    To format single plots uniformly. The sizeTuple is a tuple.
    logscale: should be an integer if trying to get axis in logscale.
    scalar: allows us to format the x and y axis as scalars.
    '''
    matplotlib.rcParams.update({'font.size': fontsize}) #adjust font
    matplotlib.rcParams['axes.linewidth'] = 2.0
    
    fig = plt.figure(figsize=sizeTuple) #adjust size of figure
    ax = plt.axes()
    ax = plotSpecs(ax, logscale)

    if scalar:
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    return fig,ax

def setupPlots(sizeTuple, row=1, col=2, logscale=1, fontsize=16, scalar=True, **kwargs):
    '''
    Uses the same style as setupPlot but allows for multiple plots.
    logscale can be an integer or it can be an iterable, i.e. list.
    scalar: allows us to format the x and y axis as scalars.
    '''
    
    matplotlib.rcParams.update({'font.size': fontsize}) # adjust font
    matplotlib.rcParams['axes.linewidth'] = 2.0

    # If all the plots should be on the same scale, then the user can input an integer.
    if isinstance(logscale, int):
        logscale = np.repeat(logscale, row*col)
    
    fig, axs = plt.subplots(nrows=row, ncols=col, figsize=sizeTuple, **kwargs) # adjust size of figure
    for i, ax in enumerate(axs.flatten()):
        ax = plotSpecs(ax, logscale[i])
        if scalar:
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    return fig, axs

# roput = tpy5.tpy5_inputs_class()
# roput.zerotime = ROMANOFF
# roput.plots = 0
# roput.boxbin, roput.dsigclip = 3.0, 0
# roput.filename = 'end'
# main(roput, starID[300:], cat)
import sys
from pathlib import Path
# Can have compatability issues with imports so redefine it.
funcdir = str((Path(__file__).resolve()).parent)
homedir = str(Path(funcdir).parent.absolute())
# funcdir -> folder (parent removes the file); homedir -> bls_cuda
basedir = str(Path(homedir).parent.absolute())
datadir = lambda file: f'{basedir}/scratch/{file}'
# rlcdir = lambda file: f'/home/sliu/digitalliance/data/{file}'

import h5py
import copy
import timeit
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.optimize as sopt
from scipy.stats import gamma

import matplotlib.pyplot as plt
sys.path.insert(0, f'{homedir}')
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5
import pytfit5.transitmodel as transitm
from pytfit5.synthetic import compare_bls_injection

## CONSTANTS
ROMANOFF = 2461450
# SAVEDIR = f'{basedir}/scratch/output'
SAVEDIR = f'{basedir}/scratch'
LCDIR = datadir('rlc')
COLS = np.array(['RIC', 'Period', 'T0', 'TDur', 'TDepth', \
                 'RawPer', 'SR', 'Power', 'SNR', 'Time'])
RECCOLS = ['overlap_fraction', 'precision', 'true_positive', \
          'false_positive', 'false_negative', 'period_factor', \
          'period_factor_type', 'is_recovered']
starID = np.loadtxt(datadir('starID.txt')).astype(int)
cat = pd.read_csv(datadir('trunccat.csv')) # trimmed planet catalogue!

# inherit from this class, so we can not deal with compatability issues and just put this into our functionalities?
class pipelineIns(tpy5.tpy5_inputs_class):

    def __init__(self):
        super().__init__() 
        
        self.rics = starID
        self.df = cat
        self.blsfunc = gbls.bls
        self.tlsfunc = gbls.tls
        self.tbuff = 0.23
        self.bbuff = 0.19
        self.itmax = 10 # To set a single run, itmax = 1.
        self.saveIt = 500
        self.savedir = lambda method, num: f'{SAVEDIR}/{self.filename}{method}{num}.csv'
        
    def t5Compat(self):
        '''
        Changing the directionaries for compatability bc savedir is a function,
        and also configuring the lcdir so that it points to an absolute path.
        Changes the saving mechanism (see save2csv and main) so we do not save.
        '''
        if not self.lcdir: # Change to default if nothing is given.
            self.lcdir = LCDIR
        # Changes the configuration so that the dataframes are not saved.
        if self.savedir is None:
            self.savedir = lambda method, num: None
        if isinstance(self.savedir, str): # if savedir is a string.
            if self.savedir != '':
                savedir = self.savedir.strip('/')
                self.savedir = lambda method, num: f'/{savedir}/{self.filename}{method}{num}.csv'
            else: # just in case an empty string is entered here.
                self.savedir = lambda method, num: f'{SAVEDIR}/{self.filename}{method}{num}.csv'
            

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
    return phot.icut == 0

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

def prepInputs(ric, t5puts):
    '''
    Loading in the photometric data and data processing.
    '''
    phot = read2Phot(f'{t5puts.lcdir}/{int(ric)}/raw.h5') # getting the photometric data
    m = processData(phot, t5puts) # detrending/clipping
    ## The stellar parameters which are unique to each RIC.
    starRow = getRICRow(ric, t5puts.df)
    t5puts.rstar = float(starRow['star_radius'])
    t5puts.mstar = float(starRow['star_mass'])
    t5puts.u = starRow[['transit_limb1_F146', 'transit_limb2_F146']].values

    return t5puts, phot.time[m], phot.flux_f[m], phot.ferr[m]

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

def singleIt(t5puts, t, f, err, func):
    '''
    Running a single iteration
    '''
    t1 = timeit.default_timer()
    ansob = func(t5puts, t, f, err)
    t2 = timeit.default_timer()
    stats = np.array([ansob.bper, ansob.epo, ansob.tdur, ansob.depth, \
                      ansob.rawper, ansob.SR, ansob.bpower, ansob.snr])
    return stats, t2-t1

def loop(t5puts, t, f, err, func, buff):
    '''
    Controls the while loop of tls or bls and returns an array of all the values.
    We get the stellar parameters 
    '''

    ricRun = [] # Unknown length as we begin
    # run one iteration first, and then break if we only need to run once.
    stats, tdiff = singleIt(t5puts, t, f, err, func)
    ricRun.append(np.append(stats, tdiff))

    it = 1  # Do we loop or do we not?
    # Looping until we do not see a signal or the baseline of 10.
    while stats[-1] > 6 and it < t5puts.itmax:
        # Require the time, the flux and the error be trimmed.
        scm = swissCheese(t, f, stats[:3], buff=buff)
        t, f, err = t[scm], f[scm], err[scm]
        stats, tdiff = singleIt(t5puts, t, f, err, func)
        ricRun.append(np.append(stats, tdiff))
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
    if savename is not None:
        df = pd.concat(frame, ignore_index=True)
        df.to_csv(savename, index=False)
    return []

def main(t5puts):
    '''
    Note that t5puts are the pipeline inputs which inherits
    from tpy5_input_class. 
    '''
    tlsFrames, blsFrames = [], []
    saveNum = 0
    # Handling compatability issues if tpy5_input_class is given.
    if not isinstance(t5puts, pipelineIns):
        t5puts = pipelineIns(t5puts)
    t5puts.t5Compat() # ensuring lcdir and savedir compatability
    
    for i, ric in enumerate(t5puts.rics):
        print(f'Starting RIC: {ric}, which is number: {i}!')
        t5puts, t, f, err = prepInputs(ric, t5puts)

        if t5puts.tlsfunc is not None:
            tarr = loop(t5puts, np.copy(t), np.copy(f), np.copy(err),\
                        t5puts.tlsfunc, t5puts.tbuff)
            tlsFrames.append(writedf(ric, tarr))
    
        if t5puts.blsfunc is not None:
            barr = loop(t5puts, np.copy(t), np.copy(f), np.copy(err),\
                        t5puts.blsfunc, t5puts.bbuff)
            blsFrames.append(writedf(ric, barr))

        ## Saving if necessary!
        if (max(len(tlsFrames), len(blsFrames)) >= t5puts.saveIt):
            # Checking with max because len(tlsFrame) == len(blsFrame) if both are run
            print(f'This is checkpoint no. {saveNum}.')
            tlsFrames = save2csv(t5puts.savedir('tls', saveNum), tlsFrames)
            blsFrames = save2csv(t5puts.savedir('bls', saveNum), blsFrames)
            saveNum += 1 # resetting the frames to zero so increment count.

    # To account for the last save if it is not a round number  
    if t5puts.savedir is not None:
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
    
    return fig, ax

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

def rho(mass, radius):
    '''
    Getting the density parameter!
    '''
    radius_cm = radius*gbls.Rsun*100
    mass_g = mass*gbls.Msun*10**3
    return mass_g / (4/3 * np.pi * radius_cm**3)

def getROIRow(roi, df):
    '''
    Getting the row of values for the specified ROI.
    (works because we only expect one row for each ROI)
    '''
    roidf = df[df.planet_id == roi]
    return roidf.iloc[0]

def makesol(roirow):
    '''
    Formatting the dataframe information into a class, so it 
    can be brought into the function accordingly.
    ''' 
    if 'star_rho' not in roirow.index:
        roirow['star_rho'] = rho(roirow.star_mass, roirow.star_radius)
        
    solr = transitm.transit_model_class()
    solr.npl = 1.0
    solr.t0 = [roirow.transit_t0_BRJD + (0.5 - 105)]
    solr.per = [roirow.transit_period]
    solr.rdr = [roirow.transit_rp_rstar]
    solr.bb = [roirow.planet_impact]
    solr.rho = roirow.star_rho

    return solr

def getbestrec(df):
    '''
    Identify the run with the best recovery. The hierarchy of:
    recovered -> the number of true positives -> SNR strength
    '''
    if df.empty:
        return df

    cons = [lambda df: df.is_recovered, \
            lambda df: np.array(df.true_positive) == df.true_positive.max(), \
            lambda df: np.array(df.SNR) == (df.SNR).max()]
    df['best'] = True
    counter = 0
    # counter < len(cons) prevents an error thrown up if duplicate entries
    while np.sum(df.best) > 1 and (counter < len(cons)):
        df.loc[df.best, 'best'] = cons[counter](df[df.best])
        counter += 1
    return df
    
def roirec(roirow, rundf, best=True, rlcdir=LCDIR):
    '''
    For each ROI, we must iterate through all the runs for that specific RIC.
    '''
    roirow = roirow.iloc[0]
    roi = float(roirow.planet_id)
    
    ricrundf = rundf[np.isin(rundf.RIC, int(roi))]
    recdf = pd.DataFrame(columns=RECCOLS)
    # The solution and time array are constant for all runs.
    roisol = makesol(roirow.copy())
    phot = read2Phot(f'{rlcdir}/{int(roi)}/raw.h5')
    phot.time = phot.time - ROMANOFF

    for runrow in ricrundf.itertuples():
        runsol = copy.deepcopy(roisol)
        runsol.t0 = [runrow.T0]
        runsol.per = [runrow.Period]
        runsol.rdr = [np.sqrt(runrow.TDepth)]
        runrec = compare_bls_injection(phot, sol_injected=roisol,\
                sol_bls=runsol, verbose=False)
        # This unpacks easily because runrec is in a dictionary form.
        recdf.loc[len(recdf)] = runrec

    # Convert the index from ricrundf to the recovery dataframe.
    recdf.set_index(ricrundf.index, inplace=True)
    adddf = ricrundf[['Period', 'T0', 'TDur', 'RawPer', 'TDepth', 'SNR']]
    df = pd.concat([adddf, recdf], axis=1)
    # Filtering out the noisy/non-recovered entries based on type.
    perdf = df[df.RawPer != 0] # emphasis on a matching period.
    perm = np.logical_or(~np.isin(perdf.period_factor_type, ['mismatch', 'none']), \
                      perdf.is_recovered)
    puldf = df[df.RawPer == 0] # emphasis on some form of overlap
    pulm = np.all([puldf.period_factor_type != 'none', \
        puldf.overlap_fraction > 0.2, puldf.precision > 0.2], axis=0)
    dfs = pd.concat([perdf[perm], puldf[pulm]])

    if best: return getbestrec(dfs)
    return dfs

def pulseRICs(bdf):
    '''
    If the raw period of BLS is zero, then we know the pulse signal strength
    has 'won' over the periodic signal strength. Therefore, we look for cases
    where all the runs for a certain ric have raw power of zero.
    '''
    rics, counts = np.unique(bdf.RIC, return_counts=True)
    # Count the number of runs per RIC
    rics0, counts0 = np.unique(bdf.RIC[bdf.RawPer == 0], return_counts=True)
    # Count the number of prundfulse runs
    pulsem = np.isin(rics, rics0)
    # If a RIC only has pulse runs then a periodic signal was never searched for.
    return rics0[counts[pulsem] == counts0]

def getrec(rundf, cat, verbose=True, rlcdir=None):
    '''
    Create a catalogue with only the RICs of the rundf.
    Then, add the density column to that dataframe. This
    allows us to run the recovery between the two dfs.
    '''

    scat = cat.copy()[np.isin(cat.star_id, rundf.RIC)]
    scat['star_rho'] = rho(scat.star_mass, scat.star_radius)
    scatgrp = scat.groupby('planet_id')[scat.columns]
    tqdm.pandas(desc='health bar <3')
    rec = scatgrp.progress_apply(roirec, rundf=rundf, rlcdir=rlcdir)
    rec.set_index(rec.index.get_level_values(0), inplace=True)

    reccedm = np.logical_and(rec.best, rec.is_recovered)
    # To avoid double-counting if there is an issue with best.
    reccedrois = np.unique(rec.index[reccedm])
    if verbose: print(f'The number of recoverable ROIs is {len(scat)}; ' \
                f'the number of recovered ROIs is {len(reccedrois)}.')
    return rec
    
def checkrec(recdf):
    '''
    If our best filter has missed a recovered ROI, then add it back it.
    WARNING: this may add duplicates in. 

    If pulse: we are more flexible on under/overestimations as well!
        Underestimation: there are no false positives
        Overestimations: there are no false negatives.
    If periodic: we want to check for the period_factor_type:
        fraction: i.e. overlap is high, precision is low
        multiple: i.e. precision is high, overlap is low
        close: only consider this if the overlap and precision are high enough    
    '''
    
    bestrec = recdf[np.array(recdf.best).astype(bool)]
    otherrec = recdf[~np.array(recdf.best).astype(bool)]
    notbestcheck = np.setdiff1d(otherrec.index[otherrec.is_recovered], \
                                bestrec.index[bestrec.is_recovered])


    if np.sum(recdf.best) > np.unique(recdf[recdf.best].index).shape[0]:
        rois, counts = np.unique(recdf[recdf.best].index, return_counts=True)
        doubleroi = rois[counts > 1]
        print(f'Warning! There has been double-counting for best {len(doubleroi)} ROIs: {doubleroi}')
        
    if notbestcheck.size > 0: # i.e. if this is not an empty list.
        print(f'Recovered ROIs not marked as best are: {notbestcheck}.')
        adddf = otherrec[np.isin(otherrec.index, notbestcheck)]
        bestrec = pd.concat([bestrec, adddf[adddf.is_recovered]])

    recced = bestrec[bestrec.is_recovered]
    # We aim to maxmimise ROIs recovered so only bestrec is considered.
    puldf = bestrec[np.logical_and(~bestrec.is_recovered, bestrec.RawPer == 0)]
    perdf = bestrec[np.logical_and(~bestrec.is_recovered, bestrec.RawPer != 0)]
    
    if not puldf.empty: # Pulse Exceptions for Over/Underestimates
        underest = np.logical_and(puldf.false_positive == 0, puldf.overlap_fraction > 0.3)
        overest = np.logical_and(puldf.false_negative == 0, puldf.precision > 0.3)
        puladd = puldf[np.logical_or(overest, underest)]
    else: puladd = None
        
    if not perdf.empty: # Periodic Exceptions
        mult = np.all([perdf.period_factor_type == 'multiple', \
                    perdf.overlap_fraction > (1/perdf.period_factor)*0.75, \
                    perdf.precision > 0.5], axis=0)
        frac = np.all([perdf.period_factor_type == 'fraction', \
                    perdf.precision > (1/perdf.period_factor)*0.75, \
                    perdf.overlap_fraction > 0.5], axis=0)
        close = np.all([perdf.period_factor_type == 'close', \
                perdf.precision > 0.3, perdf.overlap_fraction > 0.3], axis=0)
        exact = perdf.period_factor_type == 'exact'
        peradd = perdf[np.any([mult, frac, close, exact], axis=0)]
    else: peradd = None

    if peradd is None and puladd is None:
        return recced, pd.DataFrame()

    # probable = pd.concat([puladd, peradd])
    # ## We do not want duplicates in the probable dataframe
    # np.isin(probable.index, recced.index)
    return recced, pd.concat([puladd, peradd])

def persnr(rois, cat):
    '''
    Getting the period and SNR from the catalogue!
    Mainly for plotting purposes.
    '''
    m = np.isin(cat.planet_id, rois)
    return cat.transit_period[m], cat.planet_transit_snr[m]

# Primarily used for the histogram of the recovery statistic.
def getLims(arr):
    '''
    To return where the bins should begin by the minimum value
    and where the bin range should end by the maximum value.
    '''
    return np.min(arr), np.max(arr)

def logBins(dMin, dMax, binNum):
    '''
    Return the bin edges in logarithmic intervals. 
    '''
    return np.logspace(np.log10(dMin), np.log10(dMax), binNum+1)

def linBins(dMin, dMax, binNum):
    '''
    Return the bin edges in linear intervals. 
    '''
    return np.linspace(dMin, dMax, binNum+1)

def venn(a, b):
    '''
    Returns intersection, a unique values, b unique values.
    '''
    return np.intersect1d(a, b), np.setdiff1d(a, b), np.setdiff1d(b, a)

def vennscatter(axs, adf, bdf, cat, formatdict=[{}, {}, {}]):
    '''
    axs (list): does not neccessarily have to be unique
    adf & bdf (dataframe): should be output from checkrec
    cat (dataframe): original catalogue to extract period + snr
    formatdict (list of dict): for matplotlib formatting.
    
    '''
    vennOut = venn(np.unique(adf.index), np.unique(bdf.index))
    for i, roiarr in enumerate(vennOut):
        axs[i].scatter(*persnr(roiarr, cat), **formatdict[i])
    return axs

# Constructing and running the histogram plotting functionalities. 
def getHist(data, binNum, log, xMin=0, xMax=0):
    '''
    Obtain the histogram and the bin edges. Can be used for both logarithmic and linear plots. 
    '''

    if xMin == xMax:
        xMin, xMax = getLims(data[~np.isnan(data)])
    if log:
        binE = logBins(xMin, xMax, binNum)
    else:
        binE = linBins(xMin, xMax, binNum)
    
    return np.histogram(data, bins=binE)

def stepFul(h, x):
    '''
    So the step function begins at zero instead of starting midair.
    '''
    return np.append(np.array([0]), h), np.append(np.array([0]), x)

def zeroB(x):
    '''
    To ensure the zero bin is not overly large and allows the x array to increase in order.
    '''
    if x[0] < x[1]-x[0]:
        return x[0]
    return x[1]-x[0]

def histPlot(bData, tData, ax, labels=['TLS', 'TLS Mod'], binNum=25, log=True, rec=True):
    '''
    Allows for common formatting of the histogram. 
    '''
    if rec:
        tH, tX = getHist(tData[tData > 0], binNum, log)
        # bH, bX = np.histogram(bData[bData > 0], bins=tX)
        bH, bX = getHist(bData[bData > 0], binNum, log)
        
        tX = np.append([0, zeroB(tX)], tX)
        tH = np.append([len(tData[tData <= 0]), 0], tH)
        bX = np.append([0, zeroB(bX)], bX)
        bH = np.append([len(bData[bData <= 0]), 0], bH)
    else:
        tH, tX = getHist(tData, binNum, log)
        bH, bX = np.histogram(bData, bins=tX)
        
    ax.hist(tX[:-1], tX, weights=tH, alpha=0.3, color='orange')
    ax.hist(bX[:-1], bX, weights=bH, edgecolor='white', alpha=0.8, label=labels[0])
    tH, tX = stepFul(tH, tX)
    ax.step(tX, np.append(tH, np.array([0])), where='post', linewidth=3, label=labels[1])

    ax.grid(True, linestyle='--', alpha=0.75)
    ax.set_ylabel('Frequency')
    ax.legend()

    return ax
    

def getrecfrac(recroi, cat, binE):
    '''
    Getting the fraction of ROIs recovered based on SNR.
    recroi: the array of rois which were recovered
    cat: the catalogue for the rois which were searched! must be specific!!
         i.e. allcat[np.isin(allcat.star_id, rundf.RIC)]
    binE: the bin-edges for which we are binning into. 
    '''
    snr = cat.planet_transit_snr
    recH = np.zeros((len(binE)-1,))
    
    for i, binB in enumerate(np.transpose([binE[:-1], binE[1:]])):
        m = np.logical_and(snr >= binB[0], snr < binB[1])
        roinum = np.sum(m)
        recH[i] = len(np.intersect1d(cat.planet_id[m], recroi))/roinum
        if roinum < 1: print(f'No ROIs between {np.round(binB[0], 5)}-{np.round(binB[1], 5)}.')
        
    return recH

def gammafit(x, a, b, c):
    '''
    The parameters to fit in the cdf!
    '''
    return gamma.cdf(x, a, loc=b, scale=c)

def getgamma(binE, recfrac, ax=None, **kwargs):
    '''
    binE: the bin-edges for which the midpoint we have recfracs for.
    recfrac: (# ROIs recovered)/(total ROIs in the SNR bin)
    ax: if we want to plot, then we send in an ax.
    '''
    midpoints = 1/2*(binE[1:]+binE[:-1])
    pars, errs = sopt.curve_fit(gammafit, midpoints, recfrac)

    x = np.linspace(np.min(binE), np.max(binE), 10**5)
    y = gammafit(x, *pars)
    idx = np.argmin(np.abs(y-0.5))
    print(f'The closest point to 0.5 with {len(x)} points is {x[idx]}!')

    if ax is not None:
        ax.plot(x, y, **kwargs)
        return ax, pars, (x, y)

    return pars, (x, y)

def noisefloorhist(recroi, cat, ax, binnum, binfunc=logBins, **kwargs):
    '''
    recroi: the ROIs we consider recovered.
    cat: note that this catalogue should only be for the RICs considered.
    '''
    if isinstance(binnum, np.ndarray):
        binE = binnum # we can set our bin edges manually
    else: # otherwise, we can pass binfunc for log/linear
        binE = binfunc(*getLims(cat.planet_transit_snr), binnum)

    recfrac = getrecfrac(recroi, cat, binE)    
    ax.hist(binE[:-1], binE, weights=recfrac, **kwargs)
    return ax, (binE, recfrac)

def dataVis(roi, t5puts, figsize=(8, 4)):

    _, t, f, err = prepInputs(int(roi), t5puts)
    roirow = getROIRow(roi, t5puts.df)
    t0 = roirow.transit_t0_BRJD + (0.5 - 105)
    tdur = roirow.transit_duration/24
    print(t0, tdur)
    
    # batman transit model?
    # if roirow.planet_n_transits == 1:
    fig, ax = setupPlot(figsize, logscale=0)
    ax.axvline(x=t0, lw=3, ls='dashed', color='black', label='T0')
    ax.axvline(x=(t0-tdur/2), lw=2, color='black', label='T0')
    ax.axvline(x=(t0+tdur/2), lw=2, color='black', label='T0')
    ax.scatter(t-t5puts.zerotime, f, marker='.', s=10)
    ax.set_xlabel('Time')

    return fig, ax

# rics = np.load('/home/sliu/digitalliance/D2/scratch/lastrunrics.npy').astype(int)
# piput = pipelineIns()
# piput.zerotime = ROMANOFF
# piput.rics = rics
# piput.boxbin, piput.dsigclip = 3.0, 0
# piput.itmax = 10
# piput.plots = 0
# piput.filename = 'last'
# piput.blsfunc = gbls.bls
# piput.tlsfunc = gbls.tls
# piput.saveIt = 100
# piput.lcdir = '/home/sliu/digitalliance/D1/data/rlc'
# main(piput)
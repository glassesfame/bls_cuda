import sys
from pathlib import Path
# Can have compatability issues with imports so redefine it.
funcdir = str((Path(__file__).resolve()).parent)
homedir = str(Path(funcdir).parent.absolute())
# funcdir -> folder (parent removes the file); homedir -> bls_cuda
datadir = lambda file: f'{homedir}/data/{file}'
rlcdir = lambda file: f'/home/sliu/digitalliance/data/{file}'

import h5py
import copy
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
SAVEDIR = f'{gbls.homedir}/data/output'
LCDIR = rlcdir('rlc')
COLS = np.array(['RIC', 'Period', 'T0', 'TDur', 'TDepth', \
                 'RawPer', 'SR', 'Power', 'SNR', 'Time'])
RECCOLS = ['overlap_fraction', 'precision', 'true_positive', \
          'false_positive', 'false_negative', 'period_factor', \
          'period_factor_type', 'is_recovered']
starID = np.loadtxt(rlcdir('stID.txt')).astype(int)
cat = pd.read_csv(datadir('trunccat.csv')) # trimmed planet catalogue!

# inherit from this class, so we can not deal with compatability issues and just put this into our functionalities?
class pipelineIns(tpy5.tpy5_inputs_class):

    def __init__(self):
        super().__init__() 
        self.rics = starID
        self.df = cat
        self.blsfunc = gbls.bls
        self.tlsfunc = gbls.tls
        self.single = False
        self.tbuff = 0.25
        self.bbuff = 0.05
        self.saveIt = 150
        self.savedir, self.lcdir = self.t5Compat()
        
    def t5Compat(self):
        '''
        Changing the directionaries so that we save it to the correct directionaries
        and also configuring the lcdir so that it points to an absolute path.
        '''
        # Configure the save mechanism and the light-curve directory.
        if not self.lcdir:
            self.lcdir = LCDIR
        if not self.savedir:
            self.savedir = lambda method, num: f'{SAVEDIR}/{self.filename}{method}{num}.csv'
            
        if isinstance(self.savedir, str): # Ensuring nothing will go wrong with saving.
            savedir = self.savedir.strip('/')
            self.savedir = lambda file: f'/{savedir}/{file}'
        return self.savedir, self.lcdir

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
    if t5puts.single: # Do we loop or do we not?
        return np.array(ricRun)

    it = 1
    # Looping until we do not see a signal or the baseline of 10.
    while np.abs(stats[-1]) > 6 and it < 10:
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
    
    for i, ric in enumerate(t5puts.rics):
        print(f'Starting RIC: {ric}, which is number: {i}!')
        t5puts, t, f, err = prepInputs(ric, t5puts)

        if t5puts.tlsfunc is not None:
            tarr = loop(t5puts, np.copy(t), np.copy(f), np.copy(err), t5puts.tlsfunc, t5puts.tbuff)
            tlsFrames.append(writedf(ric, tarr))
    
        if t5puts.blsfunc is not None:
            barr = loop(t5puts, np.copy(t), np.copy(f), np.copy(err), t5puts.blsfunc, t5puts.bbuff)
            blsFrames.append(writedf(ric, barr))

        ## Saving if necessary!
        if max(len(tlsFrames), len(blsFrames)) >= t5puts.saveIt:
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

def roirec(roirow, rundf, rlcdir=None):
    '''
    For each ROI, we must iterate through all the runs for that specific RIC.
    '''
    roirow = roirow.iloc[0]
    roi = float(roirow.planet_id)
    
    ricrundf = rundf[np.isin(rundf.RIC, int(roi))]
    recdf = pd.DataFrame(columns=RECCOLS)
    # The solution and time array are constant for all runs.
    roisol = makesol(roirow)
    if rlcdir == None:
        rlcdir = f'{LCDIR}/{int(roi)}raw.h5'
    phot = read2Phot(rlcdir)
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
    m = np.array(recdf.period_factor_type != 'mismatch')
    if bool(np.sum(m)):
        adddf = ricrundf.loc[m, ['Period', 'T0', 'TDur', 'TDepth', 'SNR']]
        dfs = pd.concat([adddf, recdf[m]], axis=1)
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
    # Count the number of pulse runs
    pulsem = np.isin(rics, rics0)
    # If a RIC only has pulse runs then a periodic signal was never searched for.
    return rics0[counts[pulsem] == counts0]

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

# import pickle
# with open(datadir('lowsnr.pickle'), 'rb') as f:
#     lowrics = pickle.load(f)

# piput = pipelineIns()
# piput.zerotime = ROMANOFF
# piput.boxbin, piput.dsigclip = 3.0, 0
# piput.filename = 'pulse_noise_floor'
# piput.rics = lowrics['all']
# piput.tlsfunc = None
# piput.plots = 0
# piput.blsfunc = gbls.bls_pulse
# piput.saveIt = 500
# main(piput)

# piput.filename = 'bls_noise_floor'
# piput.blsfunc = gbls.bls
# main(piput)
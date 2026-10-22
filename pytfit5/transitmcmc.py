import numpy as np
import pytfit5.transitmodel as transitm
import pytfit5.keplerian as kep
import pytfit5.mcmcroutines as mcmc
import pytfit5.transitPy5 as tpy5
import time


DSCALE_PARAM = "dscale"
DSCALE_LOG_BOUNDS = (np.log(1.0e-6), np.log(10.0))   # dscale in (0, 10]; initial default = 1.0
DSCALE_DEFAULT = 1.0
DSCALE_DEFAULT_ERROR = 0.01


class priors_class:
    """
    Optional Gaussian (split-normal) soft priors for any transit model parameter.

    Stellar parameters (scalars): set as [mean, sigma_up, sigma_lo]
        priors.rho  = [1.5, 0.3, 0.3]
        priors.nl3  = [0.367, 0.007, 0.007]

    Planet parameters: set as a list of [mean, sigma_up, sigma_lo] per planet, or
    a single [mean, sigma_up, sigma_lo] applied to all planets.
        priors.bb   = [[0.3, 0.2, 0.2], None]   # Prior on planet 1; no prior on planet 2
        priors.rdr  = [[0.1, 0.02, 0.02]]        # Single planet
        priors.per  = [3.0, 0.01, 0.01]          # Same prior applied to all planets

    All attributes default to None (no prior).
    sigma_up and sigma_lo must be > 0; a zero or negative sigma silently skips that term.
    """
    def __init__(self):
        # Stellar parameters (scalar)
        self.rho = None
        self.nl1 = None
        self.nl2 = None
        self.nl3 = None
        self.nl4 = None
        self.dil = None
        self.vof = None
        self.zpt = None
        # Planet parameters (list of priors per planet, or a single spec for all planets)
        self.t0  = None
        self.per = None
        self.bb  = None
        self.rdr = None
        self.ecw = None
        self.esw = None
        self.krv = None
        self.ted = None
        self.ell = None
        self.alb = None


_ST_PARAM_NAMES = ["rho", "nl1", "nl2", "nl3", "nl4", "dil", "vof", "zpt"]
_PL_PARAM_NAMES = ["t0", "per", "bb", "rdr", "ecw", "esw", "krv", "ted", "ell", "alb"]


def _gaussian_prior_contribution(value, spec):
    """
    Split-normal log prior contribution.  spec = [mean, sigma_up, sigma_lo].
    Returns 0.0 for a None spec or non-positive sigma.
    """
    if spec is None:
        return 0.0
    mean, sig_up, sig_lo = spec
    residual = value - mean
    sigma = sig_up if residual >= 0 else sig_lo
    if sigma <= 0:
        return 0.0
    return -0.5 * (residual / sigma) ** 2


def _apply_gaussian_priors(sol, priors):
    """
    Sum Gaussian prior contributions across all parameters.

    sol: flat parameter array in natural (linear) units.
    priors: priors_class instance, or None.  Returns 0.0 immediately if None.

    For planet parameters, spec_all can be:
      - None                      -> no prior
      - [mean, sig_up, sig_lo]    -> same prior applied to every planet
      - [[mean, sig_up, sig_lo],
         None, ...]               -> per-planet specs (None = no prior for that planet)
    """
    if priors is None:
        return 0.0

    lp = 0.0

    # Stellar scalar parameters (indices 0-7)
    for k, name in enumerate(_ST_PARAM_NAMES):
        spec = getattr(priors, name, None)
        lp += _gaussian_prior_contribution(sol[k], spec)

    # Planet parameters
    npl = (len(sol) - transitm.nb_st_param) // transitm.nb_pl_param
    for i in range(npl):
        for j, name in enumerate(_PL_PARAM_NAMES):
            spec_all = getattr(priors, name, None)
            if spec_all is None:
                continue
            # Detect per-planet list-of-lists vs. a single [mean, s+, s-] for all planets.
            # spec_all[0] is a sequence only for the nested (per-planet) case.
            try:
                len(spec_all[0])
                spec = spec_all[i] if i < len(spec_all) else None
            except TypeError:
                spec = spec_all
            idx = transitm.nb_st_param + i * transitm.nb_pl_param + j
            lp += _gaussian_prior_contribution(sol[idx], spec)

    return lp


def _get_fit_parameter_layout(sol, params_to_fit):
    id_to_fit = np.array([transitm.var_to_ind[param] for param in params_to_fit if param != DSCALE_PARAM], dtype=int)
    log_space_params = np.array([transitm.var_to_ind["rho"], transitm.var_to_ind["rdr"]], dtype=int)
    fit_dscale = DSCALE_PARAM in params_to_fit

    # Expand indices arrays to fit multiple planets
    for i in range(sol.npl - 1):
        mask = (id_to_fit >= transitm.nb_st_param) & (id_to_fit < (transitm.nb_pl_param + transitm.nb_st_param))
        id_to_fit = np.append(id_to_fit, id_to_fit[mask] + (i + 1) * transitm.nb_pl_param)

        mask_log = (log_space_params >= transitm.nb_st_param) & (log_space_params < (transitm.nb_pl_param + transitm.nb_st_param))
        log_space_params = np.append(log_space_params, log_space_params[mask_log] + (i + 1) * transitm.nb_pl_param)

    return id_to_fit, log_space_params, fit_dscale


def _get_dscale_values(sol):
    dscale = getattr(sol, DSCALE_PARAM, DSCALE_DEFAULT)
    ddscale = getattr(sol, "d" + DSCALE_PARAM, DSCALE_DEFAULT_ERROR)
    if dscale <= 0:
        dscale = DSCALE_DEFAULT
    if ddscale <= 0:
        ddscale = DSCALE_DEFAULT_ERROR
    return dscale, ddscale


def setup_priors(sol):
    """
    Builds the lower and upper bound arrays for the priors.
    This function should be called ONLY ONCE before the MCMC loop.
    """
    min_t = -1.0e9
    max_t =  1.0e9

    base_ubounds = np.array([1e3, 2, 1, 1, 1, 1, 1, 5, max_t, max_t, 2, 1, 1, 1, 5, 1e3, 1e3, 1e4])
    base_lbounds = np.array([1e-4, 0, -1, 0, 0, 0, 0, -5, min_t, 0, 0, 0, -1, -1, -5, 0, -1e3, 0])
    
    npl = (len(sol) - transitm.nb_st_param) // transitm.nb_pl_param

    if npl > 1:
        planet_only_lbounds = base_lbounds[transitm.nb_st_param : (transitm.nb_st_param + transitm.nb_pl_param)]
        planet_only_ubounds = base_ubounds[transitm.nb_st_param : (transitm.nb_st_param + transitm.nb_pl_param)]
        
        additional_lbounds = np.tile(planet_only_lbounds, npl - 1)
        additional_ubounds = np.tile(planet_only_ubounds, npl - 1)

        lbounds = np.hstack([base_lbounds, additional_lbounds])
        ubounds = np.hstack([base_ubounds, additional_ubounds])
    else:
        lbounds = base_lbounds
        ubounds = base_ubounds
        
    return lbounds, ubounds


def genmcmcInput(sol, params_to_fit, priors=None):
    """
    Returns the new log function, the sol array and the beta array to use for mcmc

    sol: Transit Model object containing the initial parameters
    params_to_fit: List containing strings of the names of the parameters to fit according to the tm class
    priors: Optional priors_class instance with Gaussian soft priors for any parameter. Defaults to None.

    return: New log prob function, Array of initial parameters to pass to mcmc, Initial guess for beta using errors
    """
    id_to_fit, log_space_params, fit_dscale = _get_fit_parameter_layout(sol, params_to_fit)

    sol_a = sol.to_array()

    def newLogprob(fit_sol, time, flux, ferror, itime, nintg, ntt, tobs, omc, lbounds, ubounds):
        for i, ind in enumerate(id_to_fit):
            if ind in log_space_params:
                sol_a[ind] = np.exp(fit_sol[i])
            else:
                sol_a[ind] = fit_sol[i]

        if fit_dscale:
            if not DSCALE_LOG_BOUNDS[0] <= fit_sol[len(id_to_fit)] <= DSCALE_LOG_BOUNDS[1]:
                return -np.inf
            dscale = np.exp(fit_sol[len(id_to_fit)])
        else:
            dscale = DSCALE_DEFAULT

        return logprob(sol_a, time, flux, ferror, itime, nintg, ntt, tobs, omc, lbounds, ubounds, dscale=dscale, priors=priors)
    
    err_a = sol.err_to_array()

    # Change parameters in log space
    for i in log_space_params:
        if i in id_to_fit:
            err_a[i] = err_a[i]/sol_a[i] # Error on f=ln(x) is error(x)/x
            sol_a[i] = np.log(sol_a[i])

    x = np.array(sol_a[id_to_fit], dtype=np.float64)
    beta = np.array(err_a[id_to_fit], dtype=np.float64)

    if fit_dscale:
        dscale, ddscale = _get_dscale_values(sol)
        x = np.append(x, np.log(dscale))
        beta = np.append(beta, ddscale / dscale)

    return newLogprob, x, beta

def getParams(chain, burnin, sol, params_to_fit):
    """
    Generates a transit model object from the markov chain.
    """
    # Get params from mcmc
    cut_chain = chain[burnin:,:]
    mm = np.median(cut_chain, axis=0)
    std = np.std(cut_chain, axis=0)

    # Return to the full array
    id_to_fit, log_space_params, fit_dscale = _get_fit_parameter_layout(sol, params_to_fit)

    sol_full = sol.to_array()
    err_full = np.zeros(len(sol_full))
    for i, ind in enumerate(id_to_fit):
        if ind in log_space_params:
            sol_full[ind] = np.exp(mm[i])
            err_full[ind] = np.exp(mm[i]) * std[i] # Error on f=exp(x) is exp(x)*error(x)
        else:
            sol_full[ind] = mm[i]
            err_full[ind] = std[i]

    # Generate the object
    sol_output = transitm.transit_model_class()
    sol_output.from_array(sol_full)
    sol_output.load_errors(err_full)

    if fit_dscale:
        dscale_index = len(id_to_fit)
        sol_output.dscale = np.exp(mm[dscale_index])
        sol_output.ddscale = np.exp(mm[dscale_index]) * std[dscale_index]
    else:
        sol_output.dscale = getattr(sol, DSCALE_PARAM, DSCALE_DEFAULT)
        sol_output.ddscale = getattr(sol, "d" + DSCALE_PARAM, 0.0)

    return sol_output

def cutOutOfTransit(sol, phot, tdurcut=2):
    """
    Cuts data that is out of transit.
    
    Returns a new phot object.
    """

    condition = False

    for i in range(sol.npl):
        phase = (phot.time - sol.t0[i])/sol.per[i] - np.floor((phot.time - sol.t0[i])/sol.per[i])
        phase[phase<-0.5] += 1.0
        phase[phase>0.5] -= 1.0

        tdur = kep.transitDuration(sol, i)/sol.per[i]

        condition |= (phase > -tdurcut*tdur) & (phase < tdurcut*tdur)

    phot_out = tpy5.phot_class()
    phot_out = phot[condition]

    return phot_out

# def logprob(sol, time, flux, ferror, itime, nintg, ntt, tobs, omc):
#     return loglikehood(transitm._transitModel, sol, time, flux, ferror, itime, nintg, ntt, tobs, omc) + logprior(sol, time)
def logprob(sol, time, flux, ferror, itime, nintg, ntt, tobs, omc, lbounds, ubounds, dscale=DSCALE_DEFAULT, priors=None):
    # The loglikehood call stays exactly the same for now
    # print("sol_a:", sol)
    ll = loglikehood(transitm.transitModel, sol, time, flux, ferror, itime, nintg, ntt, tobs, omc, dscale=dscale)
    if np.isnan(ll):
        ll = -1.0e30
    
    # The logprior call now passes the pre-built bounds instead of the 'time' array
    lp = logprior(sol, lbounds, ubounds, priors=priors)
    # if np.isinf(lp):
    #     lp = 0

    # print('LogLikehood:', ll, ' LogPrior:', lp, ' LogProb:', ll + lp)
    
    return ll + lp

def loglikehood(modelFunc, sol, time, flux, ferror, itime, nintg, ntt, tobs, omc, dscale=DSCALE_DEFAULT):

    if not np.isfinite(dscale) or dscale <= 0:
        return -1.0e30

    model = modelFunc(sol, time, itime, nintg, ntt, tobs, omc)

    n = len(flux)

    if n < 1:
        ll = -1e30
    else:
        variance = ferror * ferror * dscale * dscale
        ll = -0.5 * (n * np.log(2 * np.pi) + np.sum(np.log(variance) + ((flux - model) * (flux - model) / variance)))

    return ll

def logprior(sol, lbounds, ubounds, priors=None):
    badprior = -np.inf
    lprior = 0

    # # min_t = min(time)
    # # max_t = max(time)
    # min_t = -1.0e9
    # max_t =  1.0e9

    # ubounds = np.array([1e3, 2, 1, 1, 1, 1, 1, 5, max_t, max_t, 2, 1, 1, 1, 5, 1e3, 1e3, 1e4])
    # lbounds = np.array([1e-4, 0, -1, 0, 0, 0, 0, -5, min_t, 0, 0, 0, -1, -1, -5, 0, -1e3, 0])

    npl = (len(sol) - transitm.nb_st_param) // transitm.nb_pl_param

    # Expand bounds for multiple planets
    for i in range(npl - 1):
        lbounds = np.append(lbounds, lbounds[transitm.nb_st_param : (transitm.nb_pl_param + transitm.nb_st_param)])
        ubounds = np.append(ubounds, ubounds[transitm.nb_st_param : (transitm.nb_pl_param + transitm.nb_st_param)])

    for i in range(len(sol)):
        if lbounds[i] <= sol[i] <= ubounds[i]:
            continue
        else:
            return badprior

    # Joint (b, k) prior:
    #   - Hard bound: b >= 1 + k means no transit chord exists (geometrically impossible) -> -inf
    #   - Soft penalty: once in the grazing zone (b > 1-k), apply a Gaussian penalty
    #     whose width is scaled by k so that large planets tolerate more grazing than small ones.
    for i in range(npl):
        b   = sol[transitm.nb_st_param + 2 + i * transitm.nb_pl_param]  # bb
        rdr = sol[transitm.nb_st_param + 3 + i * transitm.nb_pl_param]  # r/R*

        # Hard bound: transit chord length = sqrt((1+k)^2 - b^2) becomes imaginary
        if b >= 1.0 + rdr:
            return badprior

        # Soft penalty in the grazing zone, width proportional to k
        grazing_excess = b - (1.0 - rdr)
        if grazing_excess > 0.0 and rdr > 0.0:
            lprior += -(grazing_excess / rdr) ** 2

    lprior += _apply_gaussian_priors(sol, priors)

    return lprior

def demcmcRoutine(x, beta, phot, sol_a, serr, params, lnprob, nintg=41, ntt=-1, tobs=-1, omc=-1):
    start_time = time.time()

    n_planet = (len(sol_a) - transitm.nb_st_param) // transitm.nb_pl_param
    nb_pts = len(phot.time[(phot.icut == 0) & (phot.tflag == 1)])
    # Handle TTV inputs
    if type(ntt) is int:
        ntt = np.zeros(n_planet, dtype="int32") # Number of TTVs measured 
        tobs = np.zeros((n_planet, nb_pts)) # Time stamps of TTV measurements (days)
        omc = np.zeros((n_planet, nb_pts)) # TTV measurements (O-C) (days)

    # ==================== NEW SETUP BLOCK ====================
    selection_mask = (phot.icut == 0) & (phot.tflag == 1)
    times  = phot.time[selection_mask]
    flux_f = phot.flux_f[selection_mask]
    ferr   = phot.ferr[selection_mask]
    itime  = phot.itime[selection_mask]

    # Call the setup function ONCE to get the prior bounds
    lbounds, ubounds = setup_priors(sol_a) 

    # =======================================================

    # Read params
    nsteps1 = params[0]
    nsteps2 = params[1]
    nsteps_inc = params[2]
    burninf = params[3]
    niter_cor = params[4]
    burnin_cor = params[5]
    nthin = params[6]
    nloopmax = params[7]
    converge_crit = params[8] 
    buf_converge_crit = params[9]

    #Check for run-away model
    TPnsteps=5000
    TPnthin=1
    
    chain,accept=mcmc.genchain(x,beta,TPnsteps,lnprob,mcmc.mhgmcmc,
                               times,flux_f,ferr,itime,nintg, ntt, tobs, omc, lbounds, ubounds)
    
    runtest=np.array(tpy5.checkperT0(chain,burninf,TPnthin,sol_a,serr))
    print('runtest:',runtest)
    runtest2 = runtest + 1.0e-5 #add small eps to avoid division by zero.
    if int(np.sum(runtest2[runtest2<1.0]/runtest2[runtest2<1.0]))==4.0:
        TPflag=1 #flag for looping until convergence is met.
    else:
        TPflag=0 #run-away
        
    #if models are so far so good.. continue
    if TPflag==1:

        #get better beta 
        corscale=mcmc.betarescale(x,beta,niter_cor,burnin_cor,lnprob,mcmc.mhgmcmc,
                                  times,flux_f,ferr,itime,nintg, ntt, tobs, omc, lbounds, ubounds, imax=10)
        # corscale = np.ones(len(beta)) # TEMPORARY FIX: Disable beta rescaling due to issues with small step sizes
        
        #first run with M-H to create buffer.

        nloop=0
        nsteps=np.copy(nsteps1)
        mcmcloop=True
        while mcmcloop==True:
        
            nloop+=1 #count number of loops
            
            hchain1,haccept1=mcmc.genchain(x,beta*corscale,TPnsteps,lnprob,mcmc.mhgmcmc,
                                           times,flux_f,ferr,itime,nintg, ntt, tobs, omc, lbounds, ubounds)
            hchain2,haccept2=mcmc.genchain(x,beta*corscale,TPnsteps,lnprob,mcmc.mhgmcmc,
                                           times,flux_f,ferr,itime,nintg, ntt, tobs, omc, lbounds, ubounds)
            hchain3,haccept3=mcmc.genchain(x,beta*corscale,TPnsteps,lnprob,mcmc.mhgmcmc,
                                           times,flux_f,ferr,itime,nintg, ntt, tobs, omc, lbounds, ubounds)

            if nloop==1:
                chain1=np.copy(hchain1)
                chain2=np.copy(hchain2)
                chain3=np.copy(hchain3)
                accept1=np.copy(haccept1)
                accept2=np.copy(haccept2)
                accept3=np.copy(haccept3)
            else:
                chain1=np.concatenate((chain1,hchain1))
                chain2=np.concatenate((chain2,hchain2))
                chain3=np.concatenate((chain3,hchain3))
                accept1=np.concatenate((accept1,haccept1))
                accept2=np.concatenate((accept2,haccept2))
                accept3=np.concatenate((accept3,haccept3))
            
            burnin=int(chain1.shape[0]*burninf)
            mcmc.calcacrate(accept1,burnin)

            grtest=mcmc.gelmanrubin(chain1,chain2,chain3,burnin=burnin,npt=len(times))
            print('Gelman-Rubin Convergence:')
            print('parameter  Rc')
            for i in range(0,len(chain1[1,:])):
                print('%8s  %.4f' %(str(i),grtest[i]))
            if int(np.sum(grtest[grtest<buf_converge_crit]/grtest[grtest<buf_converge_crit]))==len(grtest):
                mcmcloop=False
            else:
                mcmcloop=True
                nsteps+=nsteps1
                
            runtest=np.array(tpy5.checkperT0(chain1,burninf,TPnthin,sol_a,serr))
            print('runtest:',runtest)
            if int(np.sum(runtest[runtest<1.0]/runtest[runtest<1.0]))!=4.0:
                mcmcloop=False #run-away
                
            if nloop>=nloopmax: #break if too many loops
                mcmcloop=False
                
            print("---- %s seconds ----" % (time.time() - start_time))
                
        ## Re-use runtest for buffer creation loop.
        #runtest=np.array(tf.checkperT0(chain1,burninf,TPnthin,sol,serr))
        #print('runtest:',runtest)
        if int(np.sum(runtest[runtest<1.0]/runtest[runtest<1.0]))==4.0:
            mcmcloop=True #flag for looping until convergence is met.
        else:
            mcmcloop=False #run-away

        #mcmcloop=True
        nloop=0
        nsteps=np.copy(nsteps2)
        while mcmcloop==True:

            nloop+=1 #count number of loops

            burnin=int(chain1.shape[0]*burninf)
            buffer=np.concatenate((chain1[burnin:],chain2[burnin:],chain3[burnin:])) #create buffer for deMCMC
            x1=np.copy(chain1[chain1.shape[0]-1,:])
            x2=np.copy(chain1[chain1.shape[0]-1,:])
            x3=np.copy(chain1[chain1.shape[0]-1,:])
            corbeta=0.3
            burnin=int(chain1.shape[0]*burninf)
            chain1,accept1=mcmc.genchain(x1,beta*corscale,nsteps,lnprob,mcmc.demhmcmc,\
                                         times, flux_f, ferr, itime,\
                                         nintg, ntt, tobs, omc,lbounds,ubounds,buffer=buffer,corbeta=corbeta)
            chain2,accept2=mcmc.genchain(x2,beta*corscale,nsteps,lnprob,mcmc.demhmcmc,\
                                         times, flux_f, ferr, itime,\
                                         nintg, ntt, tobs, omc,lbounds,ubounds,buffer=buffer,corbeta=corbeta)
            chain3,accept3=mcmc.genchain(x3,beta*corscale,nsteps,lnprob,mcmc.demhmcmc,\
                                         times, flux_f, ferr, itime,\
                                         nintg, ntt, tobs, omc,lbounds,ubounds,buffer=buffer,corbeta=corbeta)

            burnin=int(chain1.shape[0]*burninf)

            grtest=mcmc.gelmanrubin(chain1,chain2,chain3,burnin=burnin,npt=len(times))
            print('Gelman-Rubin Convergence:')
            print('parameter  Rc')
            for i in range(0,len(chain1[1,:])):
                print('%8s  %.4f' %(str(i),grtest[i]))

            if int(np.sum(grtest[grtest<converge_crit]/grtest[grtest<converge_crit]))==len(grtest):
                mcmcloop=False
            else:
                mcmcloop=True

            burnin=int(chain1.shape[0]*burninf)
            chain=np.concatenate((chain1[burnin:,],chain2[burnin:,],chain3[burnin:,]))
            accept=np.concatenate((accept1[burnin:,],accept2[burnin:,],accept3[burnin:,]))
            burnin=int(chain.shape[0]*burninf)
            mcmc.calcacrate(accept,burnin)

            nsteps+=nsteps_inc #make longer chain to help with convergence
            
            #check for run-away Chain.
            runtest=np.array(tpy5.checkperT0(chain1,burninf,nthin,sol_a,serr))
            print('runtest:',runtest)
            if int(np.sum(runtest[runtest<1.0]/runtest[runtest<1.0]))!=4.0:
                mcmcloop=False #run-away

            if nloop>=nloopmax: #break if too many loops
                mcmcloop=False
                
            print("---- %s seconds ----" % (time.time() - start_time))
            
        print("done %s seconds ---" % (time.time() - start_time))

        return chain, accept, burnin

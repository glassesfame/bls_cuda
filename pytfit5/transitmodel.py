import numpy as np
import pytfit5.keplerian as kep
import pytfit5.occult as occ
from pytfit5.effects import albedoMod, ttv_lininterp
from numba import njit, prange

# Constants
G = 6.674e-11
Cs = 2.99792458e8

# Dictionary relating the params to indices of an array
var_to_ind = {
    "rho": 0, "nl1": 1, "nl2": 2, "nl3": 3, "nl4": 4, "dil": 5,
    "vof": 6, "zpt": 7, "t0": 8, "per": 9, "bb": 10, "rdr": 11,
    "ecw": 12, "esw": 13, "krv": 14, "ted": 15, "ell": 16, "alb": 17
}

pl_params = ["t0", "per", "bb", "rdr", "ecw", "esw", "krv", "ted", "ell", "alb"]

# Number of parameters of each type
nb_st_param = 8
nb_pl_param = 10

class transit_model_class:
    """
    Class containing all the transit model parameters
    """
    def __init__(self):

        self.rho = 1.0      # Mean stellar density (g/cm^3)
        self.nl1 = 0.0      # LDC - Only used for non-linear limb-darkening
        self.nl2 = 0.0      # LDC - Only used for non-linear limb-darkening
        self.nl3 = 0.3      # LDC - q1
        self.nl4 = 0.4      # LDC - q2
        self.dil = 0.0      # Dilution
        self.vof = 0.0      # Velocity offset
        self.zpt = 0.0      # Photometric zero point
        self.dscale = 1.0   # Multiplicative error-bar scale for MCMC likelihood
        self.ddscale = 0.0  # Error on multiplicative error-bar scale
        self.npl = 1        # Number of planets
        self.t0  = [0.0]    # Center of transit time (days)
        self.per = [1.0]    # Orbital period (days)
        self.bb  = [0.5]    # Impact parameter (NOT the square of b, I should probably change its name)
        self.rdr = [0.1]    # Rp/R*
        self.ecw = [0.0]    # sqrt(e)cos(w)
        self.esw = [0.0]    # sqrt(e)sin(w)
        self.krv = [0.0]    # RV amplitude (m/s)
        self.ted = [0.0]    # Thermal eclipse depth (ppm)
        self.ell = [0.0]    # Ellipsodial variations (ppm)
        self.alb = [0.0]    # Albedo amplitude (ppm)
    
    def load_errors(self, err_arr):
        """
        Load errors from 1D array. Array has to have a length of (8+10*n) where n is the nb of planets
        """
        if (len(err_arr) - nb_st_param) % nb_pl_param != 0:
            print("Error array doesn't have the right length")
            return

        self.npl = (len(err_arr) - nb_st_param) // nb_pl_param

        self.drho, self.dnl1, self.dnl2, self.dnl3, self.dnl4, self.ddil, self.dvof, self.dzpt = err_arr[:nb_st_param]

        self.dt0, self.dper, self.dbb, self.drdr, self.decw, self.desw, \
                    self.dkrv, self.dted, self.dell, self.dalb = ([None]*self.npl for i in range(nb_pl_param))

        for i in range(self.npl):
            self.dt0[i]  = err_arr[nb_pl_param*i + nb_st_param + 0]
            self.dper[i] = err_arr[nb_pl_param*i + nb_st_param + 1]
            self.dbb[i]  = err_arr[nb_pl_param*i + nb_st_param + 2]
            self.drdr[i] = err_arr[nb_pl_param*i + nb_st_param + 3]
            self.decw[i] = err_arr[nb_pl_param*i + nb_st_param + 4]
            self.desw[i] = err_arr[nb_pl_param*i + nb_st_param + 5]
            self.dkrv[i] = err_arr[nb_pl_param*i + nb_st_param + 6]
            self.dted[i] = err_arr[nb_pl_param*i + nb_st_param + 7]
            self.dell[i] = err_arr[nb_pl_param*i + nb_st_param + 8]
            self.dalb[i] = err_arr[nb_pl_param*i + nb_st_param + 9]

    def from_array(self, sol):
        """
        Load parameters from 1D array. Array has to have a length of (8+10*n) where n is the nb of planets
        """
        if (len(sol) - nb_st_param) % nb_pl_param != 0:
            print("Parameters array doesn't have the right length")
            return

        self.npl = (len(sol) - nb_st_param) // nb_pl_param

        self.rho, self.nl1, self.nl2, self.nl3, self.nl4, self.dil, self.vof, self.zpt = sol[:nb_st_param]

        self.t0, self.per, self.bb, self.rdr, self.ecw, self.esw, \
                    self.krv, self.ted, self.ell, self.alb = ([None]*self.npl for i in range(nb_pl_param))

        for i in range(self.npl):
            self.t0[i]  = sol[nb_pl_param*i + nb_st_param + 0]
            self.per[i] = sol[nb_pl_param*i + nb_st_param + 1]
            self.bb[i]  = sol[nb_pl_param*i + nb_st_param + 2]
            self.rdr[i] = sol[nb_pl_param*i + nb_st_param + 3]
            self.ecw[i] = sol[nb_pl_param*i + nb_st_param + 4]
            self.esw[i] = sol[nb_pl_param*i + nb_st_param + 5]
            self.krv[i] = sol[nb_pl_param*i + nb_st_param + 6]
            self.ted[i] = sol[nb_pl_param*i + nb_st_param + 7]
            self.ell[i] = sol[nb_pl_param*i + nb_st_param + 8]
            self.alb[i] = sol[nb_pl_param*i + nb_st_param + 9]

    def to_array(self):
        """
        Return a 1D array that _transitModel() can read, since it is a numba function
        """
        len_array = nb_st_param + self.npl*nb_pl_param
        sol = np.zeros(len_array)

        sol[:nb_st_param] = self.rho, self.nl1, self.nl2, self.nl3, self.nl4, self.dil, self.vof, self.zpt

        for i in range(self.npl):
            sol[nb_pl_param*i+nb_st_param : nb_pl_param*i+nb_pl_param+nb_st_param] = self.t0[i], self.per[i], \
                    self.bb[i], self.rdr[i], self.ecw[i], self.esw[i], self.krv[i], self.ted[i], self.ell[i], self.alb[i]
            
        return sol
    
    def err_to_array(self):
        """
        Return a 1D array containing errors on parameters
        """
        len_array = nb_st_param + self.npl*nb_pl_param
        serr = np.zeros(len_array)

        serr[:nb_st_param] = self.drho, self.dnl1, self.dnl2, self.dnl3, self.dnl4, self.ddil, self.dvof, self.dzpt

        for i in range(self.npl):
            serr[nb_pl_param*i+nb_st_param : nb_pl_param*i+nb_pl_param+nb_st_param] = self.dt0[i], self.dper[i], \
                    self.dbb[i], self.drdr[i], self.decw[i], self.desw[i], self.dkrv[i], self.dted[i], self.dell[i], self.dalb[i]
            
        return serr
    
    def __setattr__(self, name, value):
        """
        Allows to pass float (or int) values as planet parameters. Useful with single-planet models
        """
        if name in pl_params and isinstance(value, (float, int)):
            value = [value]
        
        return super().__setattr__(name, value)
        

def transitModel(sol, time, itime=-1, nintg=41, ntt=-1, tobs=-1, omc=-1):
    """
    Computes a transit light curve.

    sol: Array or transit model object containing all the parameters. To view the list of params, see transit_model_class
    time: Time array
    itime: Integration time array. Optional, defaults to 30 minutes.
    nintg: Number of points inside the integration time. Optional, defaults to 41.
    ntt: 1D array containing nb of ttv. shape=(nb_planet,)
    tobs: 2D array containing times of ttv. shape=(nb_planet, nb_ttv)
    omc: 2D array of o-c. shape=(nb_planet, nb_ttv)

    return: Array containing the flux values. Same length as the time array
    """

    def _coerce_array(values, dtype):
        array = np.asarray(values, dtype=dtype)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return array

    # Handle parameters
    if isinstance(sol, (np.ndarray, list)):
        sol_a = _coerce_array(sol, np.float64)
        n_planet = (len(sol) - nb_st_param) // nb_pl_param
    else:
        sol_a = _coerce_array(sol.to_array(), np.float64)
        n_planet = sol.npl

    time = _coerce_array(time, np.float64)

    nb_pts = len(time)

    # Handle integration time
    if type(itime) in (int, float):
        if itime < 0:
            itime = np.full(nb_pts, 0.020434) # 30 minutes integration time
        else:
            itime = np.full(nb_pts, float(itime))
    else:
        itime = _coerce_array(itime, np.float64)

    # Handle TTV inputs
    if type(ntt) is int:
        ntt = np.zeros(n_planet, dtype="int32") # Number of TTVs measured 
        tobs = np.zeros((n_planet, nb_pts)) # Time stamps of TTV measurements (days)
        omc = np.zeros((n_planet, nb_pts)) # TTV measurements (O-C) (days)
    else:
        ntt = _coerce_array(ntt, np.int32)
        tobs = _coerce_array(tobs, np.float64)
        omc = _coerce_array(omc, np.float64)
    
    return _transitModel(sol_a, time, itime, nintg, ntt, tobs, omc)

@njit(parallel=True, cache=True)
def _transitModel(sol, time, itime, nintg, ntt, tobs, omc):
    """
    Computes a transit light curve without all the input checking.

    sol: Array containing all the parameters. To view the list of params, see transit_model_class
    time: Time array
    itime: Integration time array. Has to be the same length as time
    nintg: Number of points inside the integration time
    ntt: 1D array containing nb of ttv. shape=(nb_planet,)
    tobs: 2D array containing times of ttv. shape=(nb_planet, nb_ttv)
    omc: 2D array of o-c. shape=(nb_planet, nb_ttv)

    return: Array containing the flux values. Same length as the time array
    """

    # Reading parameters
    density = sol[0]
    c1 = sol[1]
    c2 = sol[2]
    c3 = sol[3]
    c4 = sol[4]
    dil = sol[5]
    voff = sol[6]
    zpt = sol[7]

    # Kipping Coefficients
    a1 = 2 * np.sqrt(c3) * c4
    a2 = np.sqrt(c3) * (1 - 2*c4)

    nb_pts = len(time)
    tmodel = np.zeros(nb_pts)
    dtype = np.zeros(nb_pts) # Photometry only

    n_planet = (len(sol) - nb_st_param) // nb_pl_param

    # Loop over every planet
    for ii in range(n_planet):

        # Read parameters for the planet
        epoch = sol[nb_pl_param*ii + nb_st_param + 0]
        Per = sol[nb_pl_param*ii + nb_st_param + 1]
        b = sol[nb_pl_param*ii + nb_st_param + 2]
        Rp_Rs = sol[nb_pl_param*ii + nb_st_param + 3]

        ecw = sol[nb_pl_param*ii + nb_st_param + 4]
        esw = sol[nb_pl_param*ii + nb_st_param + 5]
        eccn = ecw*ecw + esw*esw

        # Calculation for omega (w) here
        if eccn >= 1:
            eccn = 0.99
        elif eccn == 0:
            w = 0
        else:
            # arctan2 gives a result in [-pi, pi], so we add 2pi to the negative values
            w = np.arctan2(esw, ecw)
            if w < 0:
                w += 2*np.pi

        # Calculate a/R*
        a_Rs = 10 * np.cbrt(density * G * (Per*86400)**2 / (3*np.pi))

        K = sol[nb_pl_param*ii + nb_st_param + 6] # RV amplitude
        ted = sol[nb_pl_param*ii + nb_st_param + 7]/1e6 # Occultation Depth
        ell = sol[nb_pl_param*ii + nb_st_param + 8]/1e6 # Ellipsoidal variations
        ag = sol[nb_pl_param*ii + nb_st_param + 9]/1e6 # Albedo amplitude

        # Calculate phi0
        Eanom = 2 * np.arctan(np.tan(w/2) * np.sqrt((1 - eccn)/(1 + eccn)))
        phi0 = Eanom - eccn*np.sin(Eanom)

        # To avoid calculating transit twice if b=0 (y2=0 then)
        if b == 0:
            b = 1e-10

        # Calculate inclinaison
        Tanom = kep.trueAnomaly(eccn, Eanom)
        d_Rs = kep.distance(a_Rs, eccn, Tanom) # Distance over R*
        cincl = b/d_Rs # cos(incl)

        # Precompute
        eccsw = eccn*np.sin(w)
        y2 = 0 # We define y2 to avoid an error in the prange

        # Loop over all of the points
        for i in prange(nb_pts):
            time_i = time[i]
            itime_i = itime[i]
            ttcor = ttv_lininterp(tobs, omc, ntt, time_i, ii)

            tflux = np.empty(nintg)
            vt = np.empty(nintg)
            tide = np.empty(nintg)
            alb = np.empty(nintg)
            bt = np.empty(nintg)

            for j in range(nintg):
                
                # Time-Convolution
                t = time_i - itime_i * (0.5 - 1/(2*nintg) - j/nintg) - epoch - ttcor

                phi = t/Per - np.floor(t/Per)
                Manom = phi * 2*np.pi + phi0

                # Make sure Manom is in [0, 2pi]
                Manom = Manom % (2*np.pi)
                
                Eanom = kep.solve_kepler_eq(eccn, Manom, Manom) # Use Manom as the guess for Eanom, otherwise we have a race condition problem
                Tanom = kep.trueAnomaly(eccn, Eanom)
                d_Rs = kep.distance(a_Rs, eccn, Tanom)

                # Precompute some variables
                Tanom_w = Tanom - w
                sTanom_w = np.sin(Tanom_w)
                cTanom_w = np.cos(Tanom_w)

                x2 = d_Rs * sTanom_w
                y2 = d_Rs * cTanom_w*cincl

                bt[j] = np.sqrt(x2*x2 + y2*y2)

                # Calculation of RV, ellip and albedo here

                vt[j] = K * (eccsw - sTanom_w)
                tide[j] = ell * np.cbrt(d_Rs/a_Rs) * (sTanom_w*sTanom_w - cTanom_w*cTanom_w)
                alb[j] = albedoMod(Tanom_w, ag) * a_Rs/d_Rs
            
            if dtype[i] == 0:
                if y2 >= 0:
                    # Check for transit
                    is_transit = 0
                    for b in bt:
                        if b <= 1 + Rp_Rs:
                            is_transit = 1
                            break
                    
                    if is_transit:
                        # Quadratic coefficients
                        if (c3 == 0 and c4 == 0):
                            tflux = occ.occultQuad(bt, c1, c2, Rp_Rs)
                        
                        # Kipping coefficients
                        elif (c1 == 0 and c2 == 0):
                            tflux = occ.occultQuad(bt, a1, a2, Rp_Rs)
                        
                        # Non linear
                        else:
                            tflux = occ.occultSmall(bt, c1, c2, c3, c4, Rp_Rs)

                    # If no transit, tflux = 1
                    else:
                        tflux[:] = 1

                    if Rp_Rs <= 0:
                        tflux[:] = 1

                    # Add all the contributions
                    tm = 0
                    for j in range(nintg):
                        tm += tflux[j] - vt[j]/Cs + tide[j] + alb[j]

                    tm = tm/nintg
                
                # Eclipse
                else:
                    bp = bt/Rp_Rs
                    # Treat the star as the object blocking the light
                    occult = occ.occultUniform(bp, 1/Rp_Rs)
                    
                    if Rp_Rs < 0:
                        ratio = np.zeros(nintg)
                    else:
                        ratio = 1 - occult

                    tm = 0
                    for j in range(nintg):
                        tm += 1 - ted*ratio[j] - vt[j]/Cs + tide[j] + alb[j]

                    tm = tm/nintg

                tm += (1 - tm)*dil # Add dilution
            
            # Radial velocity
            else:
                tm = 1
                pass # To do

            tmodel[i] += tm
    
    # Add zero point
    for i in range(nb_pts):
        if dtype[i] == 0:
            tmodel[i] += zpt - (n_planet - 1) # If n_planet > 1, we need to bring down the model
        else:
            tmodel[i] += voff - 1

    return tmodel


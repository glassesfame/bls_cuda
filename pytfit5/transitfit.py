import numpy as np
from scipy.optimize import least_squares
import pytfit5.transitmodel as transitm
import pytfit5.transitPy5 as tpy5
import pytfit5.bls_cpu as gbls


def _coerce_array(values, dtype):
    array = np.asarray(values, dtype=dtype)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array

def analyseLightCurve(gbls_inputs):
    """
    Function to call to analyse a light curve and get the best-fit parameters

    fileLoc: Location of string
    gbls_inputs: Inputs of the bls

    Returns: phot object, best-fit parameters returned by fit, answers from BLS
    """

    # Read data
    phot = tpy5.readphot(gbls_inputs.lcdir + gbls_inputs.filename)
    
    # Apply BLS
    gbls_inputs.zerotime = min(phot.time)
    gbls_ans = gbls.bls(gbls_inputs, phot.time[phot.icut == 0], phot.flux_f[phot.icut == 0])
    
    # Fit using BLS answers
    sol_fit = fitFromBLS(gbls_ans, phot.time - gbls_inputs.zerotime, phot.flux_f + 1, phot.ferr, phot.itime)

    return phot, sol_fit, gbls_ans

def fitFromBLS(gbls_ans, time, flux, ferror, itime):
    """
    Fits a transit model using the answers from the bls.

    gbls_ans: Answers from the bls
    time, flux, ferror: Data arrays
    itime: Integration time array

    return: Array containing the best-fit parameters for the transit model
    """
    id_to_fit = np.array([0, 7, 8, 9, 10, 11]) # We fit only: rho, zpt, t0, Per, b, Rp/Rs

    sol = np.zeros(18) # Single planet model has up-to 18-model parameters

    # Set the initial guess using the bls answers.
    # ld coeff and rho are temporary values

    sol[0]  = 0.6  # Mean stellar density (g/cm^3)
    sol[1]  = 0.0  # Only used for non-linear limb-darkening
    sol[2]  = 0.0  # Only used for non-linear limb-darkening
    sol[3]  = 0.6  # q1 (limb-darkening)
    sol[4]  = 0.4  # q2 (limb-darkening)
    sol[5]  = 0.0  # dilution
    sol[6]  = 0.0  # Velocity offset
    sol[7]  = 0.0  # Photometric zero point
    
    sol[8]  = gbls_ans.epo             # Center of transit time (days)
    sol[9]  = gbls_ans.bper            # Orbital Period (days)
    sol[10] = 0.5                      # Impact parameter
    if gbls_ans.depth < 0:             # Rp/R*
        sol[11] = 1e-5  
    else:
        sol[11] = np.sqrt(gbls_ans.depth)
    sol[12] = 0.0  # sqrt(e)cos(w)
    sol[13] = 0.0  # sqrt(e)sin(w)
    sol[14] = 0.0  # RV amplitude (m/s)
    sol[15] = 0.0  # thermal eclipse depth (ppm)
    sol[16] = 0.0  # Ellipsodial variations (ppm)
    sol[17] = 0.0  # Albedo amplitude (ppm)

    # Sometimes t0 is negative and crashes the fit
    if gbls_ans.epo < 0:
        sol[8] += gbls_ans.bper

    return fitTransitModel(sol, id_to_fit, time, flux, ferror, itime)

def createBounds(time, id_to_fit, sol_obj):
    """
    Creates the bounds for the parameters

    time: time array
    id_to_fit: Array containing the indices of the parameters to fit
    sol_obj: Transit model object with initial parameters
    """
    min_t = min(time)
    max_t = max(time)

    # Rho and Rp/Rs are in log space
    lower_bound = np.array([np.log(1e-4), 0, -1, 0, 0, 0, -np.inf, -np.inf,      0, -np.inf, 0, -np.inf, -1, -1, -np.inf, -np.inf, -np.inf, -np.inf])
    upper_bound = np.array([np.log(1e3),  2,  1, 1, 1, 1,  np.inf,  np.inf, np.inf,  np.inf, 2,       1,  1,  1,  np.inf,  np.inf,  np.inf,  np.inf])

    # Expand bounds for multiple planets
    for i in range(sol_obj.npl - 1):
        lower_bound = np.append(lower_bound, lower_bound[transitm.nb_st_param : (transitm.nb_pl_param + transitm.nb_st_param)])
        upper_bound = np.append(upper_bound, upper_bound[transitm.nb_st_param : (transitm.nb_pl_param + transitm.nb_st_param)])

    return (lower_bound[id_to_fit], upper_bound[id_to_fit])

def fitTransitModel(sol_obj, params_to_fit, phot, nintg=41, ntt=-1, tobs=-1, omc=-1):
    """
    Function to call for fitting

    sol_obj: Transit model object with initial parameters
    params_to_fit: List containing strings of the names of the parameters to fit according to the tm class
    phot: Phot object from reading data file

    return: New transit model object containing the parameters and errors after fitting
    """

    n_planet = sol_obj.npl
    nb_pts = len(phot.time[(phot.icut == 0) & (phot.tflag == 1)])
    # Handle TTV inputs
    if type(ntt) is int:
        ntt = np.zeros(n_planet, dtype="int32") # Number of TTVs measured 
        tobs = np.zeros((n_planet, nb_pts)) # Time stamps of TTV measurements (days)
        omc = np.zeros((n_planet, nb_pts)) # TTV measurements (O-C) (days)
    else:
        ntt = _coerce_array(ntt, np.int32)
        tobs = _coerce_array(tobs, np.float64)
        omc = _coerce_array(omc, np.float64)

    # Read phot class
    selection = (phot.icut == 0) & (phot.tflag == 1)
    time = _coerce_array(phot.time[selection], np.float64)
    flux = _coerce_array(phot.flux_f[selection], np.float64) # - np.median(phot.flux) + 1
    ferror = _coerce_array(phot.ferr[selection], np.float64)
    itime = _coerce_array(phot.itime[selection], np.float64)
    
    # Transform solution object to array
    sol = _coerce_array(sol_obj.to_array(), np.float64)

    log_space_params = np.array([transitm.var_to_ind["rho"], transitm.var_to_ind["rdr"]]) # Rho and Rp/Rs are in log space
    id_to_fit = np.array([transitm.var_to_ind[param] for param in params_to_fit])

    # Expand indices arrays to fit multiple planets
    for i in range(sol_obj.npl - 1):
        mask = (id_to_fit >= transitm.nb_st_param) & (id_to_fit < (transitm.nb_pl_param + transitm.nb_st_param))
        id_to_fit = np.append(id_to_fit, id_to_fit[mask] + (i+1)*transitm.nb_pl_param)

        mask_log = (log_space_params >= transitm.nb_st_param) & (log_space_params < (transitm.nb_pl_param + transitm.nb_st_param))
        log_space_params = np.append(log_space_params, log_space_params[mask_log] + (i+1)*transitm.nb_pl_param)

    # Fit only the parameters in id_to_fit
    sol_full = np.copy(sol) # Copy the initial guess
    def wrapperTransit(sol_free, time, flux, ferror, itime, npl, nintg, ntt, tobs, omc):
        """ Wrapper function that takes only the free parameters as arguments """
        for i, ind in enumerate(id_to_fit):
            if ind in log_space_params:
                sol_full[ind] = np.exp(sol_free[i])
            else:
                sol_full[ind] = sol_free[i]
        ans = transitToOptimize(sol_full, time, flux, ferror, itime, npl, nintg, ntt, tobs, omc)
        return ans
    
    bounds = createBounds(time, id_to_fit, sol_obj)

    # Take the log of log space parameters
    for i in log_space_params:
        # Only take the log if we fit the parameter
        if i in id_to_fit:
            sol[i] = np.log(sol[i])


    res = least_squares(wrapperTransit, _coerce_array(sol[id_to_fit], np.float64), bounds=bounds, \
                        args=(time, flux, ferror, itime, sol_obj.npl, nintg, ntt, tobs, omc))

    # res = least_squares(wrapperTransit, sol[id_to_fit], bounds=bounds, \
    #                     args=(time, flux, ferror, itime, sol_obj.npl, nintg, ntt, tobs, omc),
    #                    ftol=1e-12, # Change in cost
    #                    xtol=1e-12, # Change in parameters
    #                    gtol=1e-12)

    # res = least_squares(wrapperTransit, sol[id_to_fit], \
    #                     args=(time, flux, ferror, itime, sol_obj.npl, nintg, ntt, tobs, omc),
    #                    ftol=1e-12, # Change in cost
    #                    xtol=1e-12, # Change in parameters
    #                    gtol=1e-12,
    #                    method='lm')

    # Calculate error
    is_weighted = True
    fit_params, fit_error, fit_covar = calculate_parameter_errors(res, is_weighted)

    # Recreate the full solution with the result
    err_full = np.zeros(len(sol_full))
    for i, ind in enumerate(id_to_fit):
        if ind in log_space_params:
            sol_full[ind] = np.exp(fit_params[i])
            err_full[ind] = np.exp(fit_params[i]) * fit_error[i] # Error on f=exp(x) is exp(x)*error(x)
        else:
            sol_full[ind] = fit_params[i]
            err_full[ind] = fit_error[i]

    # Return to the transitmodel object
    fit_params = transitm.transit_model_class()
    fit_params.from_array(sol_full)
    fit_params.load_errors(err_full)

    return fit_params

def transitToOptimize(sol, time, flux, ferror, itime, npl, nintg, ntt, tobs, omc):
    """
    Handles constraints and returns the vector of differences. You shouldn't have to call this function

    sol: Array of transit model parameters for fitting
    time, flux, ferror: Data arrays
    itime: Integration time array
    npl: Number of planets
    """
    n = len(time)

    # Parameter constraints (Return a big number if constraint is not respected)
    for i in range(npl):
        b = sol[10 + i*transitm.nb_pl_param]
        Rp_Rs = sol[11 + i*transitm.nb_pl_param]
        if b > 1 + Rp_Rs:
            return np.full(n, 1e20)
    
    c1, c2, c3, c4 = sol[1], sol[2], sol[3], sol[4]
    # Quadratic coefficients
    if (c3 == 0 and c4 == 0):
        if not (0 < c1 < 2) or not (-1 < c2 < 1):
            return np.full(n, 1e20)

    # Kipping coefficients
    elif (c1 == 0 and c2 == 0):
        if not (0 < c3 < 1) or not (0 < c4 < 1):
            return np.full(n, 1e20)
    
    # Non linear law
    else:
        if not (0 < c1 < 1) or not (0 < c2 < 1) or not (0 < c3 < 1) or not (0 < c4 < 1):
            return np.full(n, 1e20)

    y_model = transitm._transitModel(sol, time, itime, nintg, ntt, tobs, omc)

    # print('hello')
    # print(np.sum((y_model - flux)**2/ferror**2))

    return (y_model - flux)/ferror

def calculate_parameter_errors(opt_result, residual_func_returns_weighted=True):
    """
    Calculates parameter errors (standard deviations) from scipy.optimize.least_squares results.

    Args:
        opt_result: The OptimizeResult object returned by least_squares.
        residual_func_returns_weighted (bool): Set to True if your residual
            function returns (data - model) / error. Set to False if it
            returns (data - model). Defaults to True.

    Returns:
        tuple: (best_fit_params, param_errors, covariance_matrix)
               param_errors and covariance_matrix might be None or contain NaNs
               if calculation fails.
    """
    best_fit_params = opt_result.x
    param_errors = None
    covariance_matrix = None

    if not opt_result.success:
        print(f"Warning: Optimization failed or did not converge: {opt_result.message}")
        return best_fit_params, np.full_like(best_fit_params, np.nan), None

    jacobian = opt_result.jac
    residuals = opt_result.fun # Residuals at the solution

    if jacobian is None or not np.all(np.isfinite(jacobian)):
         print("Warning: Jacobian is None or contains non-finite values. Cannot compute errors.")
         return best_fit_params, np.full_like(best_fit_params, np.nan), None

    N = len(residuals)
    M = len(best_fit_params)

    if N <= M:
        print(f"Warning: Number of data points ({N}) <= number of parameters ({M}).")
        print("Cannot reliably estimate errors or covariance.")
        return best_fit_params, np.full_like(best_fit_params, np.nan), None

    try:
        jtj = jacobian.T @ jacobian

        try:
            covariance_matrix = np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            # print("Warning: Jacobian transpose * Jacobian is singular or near-singular.")
            # print("Using pseudo-inverse. Parameter errors might be unreliable, check correlations.")
            covariance_matrix = np.linalg.pinv(jtj)

        if not residual_func_returns_weighted:
            dof = N - M
            mse = np.sum(residuals**2) / dof
            covariance_matrix = covariance_matrix * mse

        # --- Extract Errors ---
        # Variances are the diagonal elements
        # Ensure variances is a writeable copy using .copy()!
        variances = np.diag(covariance_matrix).copy() # <--- FIX IS HERE

        # Handle potential small negative values (now safe to modify)
        variances[variances < 0] = 0

        param_errors = np.sqrt(variances)

    except (np.linalg.LinAlgError, ValueError, TypeError, ZeroDivisionError) as e: # Added ZeroDivisionError
        print(f"Error calculating covariance/errors: {e}")
        param_errors = np.full_like(best_fit_params, np.nan)
        # Ensure covariance_matrix is None if errors couldn't be calculated,
        # unless it was successfully calculated before the error occurred during variance extraction
        if 'variances' not in locals(): # Check if error happened before variance calculation
             covariance_matrix = None

    return best_fit_params, param_errors, covariance_matrix


def calculate_reduced_chisq_and_error_scale(sol_obj, phot, nintg=41, ntt=-1, tobs=-1, omc=-1):
    """
    Calculate reduced chi-square for a fitted transit model and return the error scaling factor
    needed to achieve reduced chi-square = 1.
    
    This function uses the same data cuts as fitTransitModel: (phot.icut == 0) & (phot.tflag == 1)
    
    Parameters
    ----------
    sol_obj : transit_model_class
        Fitted transit model object containing the best-fit parameters
    phot : phot_class
        Photometry object containing time, flux_f, ferr, itime, icut, and tflag
    nintg : int, optional
        Number of integration points for transit model (default: 41)
    ntt : int or array, optional
        Number of TTVs (default: -1, meaning no TTVs)
    tobs : array, optional
        Time stamps of TTV measurements (default: -1)
    omc : array, optional
        TTV measurements (O-C) (default: -1)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'chi_square': float, the chi-square value
        - 'dof': int, degrees of freedom (n_data - n_params)
        - 'reduced_chi_square': float, chi-square / dof
        - 'error_scale_factor': float, factor to multiply phot.ferr by to get reduced chi-square = 1
        - 'n_data': int, number of data points used
        - 'n_params': int, number of fitted parameters (from covariance matrix)
    
    Notes
    -----
    The error_scale_factor is calculated as sqrt(reduced_chi_square).
    To rescale errors: phot.ferr *= result['error_scale_factor']
    """
    
    n_planet = sol_obj.npl
    nb_pts = len(phot.time[(phot.icut == 0) & (phot.tflag == 1)])
    
    # Handle TTV inputs
    if type(ntt) is int:
        ntt = np.zeros(n_planet, dtype="int32")
        tobs = np.zeros((n_planet, nb_pts))
        omc = np.zeros((n_planet, nb_pts))
    else:
        ntt = _coerce_array(ntt, np.int32)
        tobs = _coerce_array(tobs, np.float64)
        omc = _coerce_array(omc, np.float64)
    
    # Apply the SAME data cuts as fitTransitModel
    mask = (phot.icut == 0) & (phot.tflag == 1)
    time = _coerce_array(phot.time[mask], np.float64)
    flux = _coerce_array(phot.flux_f[mask], np.float64)
    ferror = _coerce_array(phot.ferr[mask], np.float64)
    itime = _coerce_array(phot.itime[mask], np.float64)
    
    n_data = len(time)
    
    # Convert solution object to array
    sol = _coerce_array(sol_obj.to_array(), np.float64)
    
    # Calculate the model
    y_model = transitm._transitModel(sol, time, itime, nintg, ntt, tobs, omc)
    
    # Calculate residuals
    residuals = flux - y_model
    
    # Calculate chi-square
    chi_square = np.sum((residuals / ferror)**2)
    
    # Estimate number of fitted parameters from the error array
    # Parameters with non-zero errors are the ones that were fitted
    err_array = sol_obj.err_to_array()
    n_params = np.sum(err_array > 0)
    
    # Calculate degrees of freedom
    dof = n_data - n_params
    
    if dof <= 0:
        print(f"Warning: Non-positive degrees of freedom (dof={dof}). Cannot calculate reduced chi-square.")
        return {
            'chi_square': chi_square,
            'dof': dof,
            'reduced_chi_square': np.nan,
            'error_scale_factor': np.nan,
            'n_data': n_data,
            'n_params': n_params
        }
    
    # Calculate reduced chi-square
    reduced_chi_square = chi_square / dof
    
    # Calculate error scaling factor
    # To achieve reduced_chi_square = 1, we need to scale errors by sqrt(reduced_chi_square)
    error_scale_factor = np.sqrt(reduced_chi_square)
    
    return {
        'chi_square': chi_square,
        'dof': dof,
        'reduced_chi_square': reduced_chi_square,
        'error_scale_factor': error_scale_factor,
        'n_data': n_data,
        'n_params': n_params
    }

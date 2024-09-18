import numpy as np
import numba
from numba import cuda


# -----------------------------------------------------------------------------------

_name_:           str = 'Linear Regression - Deviation from expected value'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# --- CALCULATION DESCRIPTION: -------------------------------------------------------


# Calculate Price deviation from Expected by Linear Regression.

# Parameters:
# -----------------------------

# LR_Period:       int - period of Linear Regression.
# expected_period: int - period of expected value, aftre the Linear Regression, to which we project price.
#                         e.g. if expected_period = 1, we project price to the next period.
#                              if expected_period = 2, we project price to the next 2 periods.

# expected_price = slope * (LR_Period + expected_period) + intercept
# deviation = (price - expected_price) / expected_price = price / expected_price - 1.0 = Ln(price / expected_price)

# ------------------------------------------------------------------------------------

# sum_x     = (n * (n - 1)) / 2
# sum_x^2   = (n * (n - 1) * (2 * n - 1)) / 6
# divisor   = (sum_x * sum_x) - n * sum_x^2
# slope     = ((n * sum_xy) - (sum_x * sum_y)) / divisor
# intercept = (sum_y - slope * sum_x) / n
# y         = slope * x + intercept
# sum_xy    = sum_x * sum_y = sum(x * y)

# divisor = (3 * n**2 * (n - 1)**2 - 2 * n**2 * (n - 1) * (2 * n - 1)) / 12
# divisor = (3 * n**4 - 10 * n**3 + 11 * n**2 - 4 * n) / 12
# divisor = (n**2 * (n - 1) * (3 * (n -1) - 2 * (2 * n - 1))) / 12
# divisor = (n**2 * (n - 1) * (3 * n - 3 - 4 * n + 2)) / 12
# divisor = (n**2 * (n - 1) * (- n - 1)) / 12
# divisor = - (n**2 * (n - 1)**2) / 12
# divisor = - (sum_x)**2 / 3


# n * sum_xy = n * sum_x * sum_y = n * sum(x * y) = n * sum(x) * sum(y)

# slope     = ((n * sum_xy) - (sum_x * sum_y)) / divisor = ((n * sum(x) * sum(y)) - (sum(x) * sum(y))) / divisor = 
#           = (sum(x) * sum(y) * (n - 1)) / divisor  = (sum(x) * sum(y) * (n - 1)) / (- (n**2 * (n - 1)**2) / 12) = 
#           = - (12 * sum(x) * sum(y) * (n - 1)) / (n**2 * (n - 1)**2) = - (12 * sum(x) * sum(y)) / (n * (n - 1))

# sum(x)    = (n * (n - 1)) / 2

# slope = (sum(x) * sum(y) * (n - 1)) / divisor = n * (n - 1) / 2 * sum(y) * (n - 1) / (- (n**2 * (n - 1)**2) / 12) =
#       = -12 * sum(y) * n * (n - 1)**2 / (2 * n**2 * (n - 1)**2) = -6 * sum(y) / n
# (!!! WRONG !!!) slope = -6 * sum(y) / n

# VERSION 2:
# sum_x     = (n * (n + 1)) / 2
# sum_x^2   = (n * (n + 1) * (2 * n + 1)) / 6
# divisor   = (sum_x * sum_x) - n * sum_x^2
# slope     = ((n * sum_xy) - (sum_x * sum_y)) / divisor
# intercept = (sum_y - slope * sum_x) / n
# y         = slope * x + intercept

# divisor = n**2 * (n + 1) * ( 3 * (n + 1) - 2 * (2 * n + 1)) / 12 =
#         = n**2 * (n + 1) * (1 - n) / 12 =
#         = - n**2 * (n + 1) * (n - 1) / 12 = - n**2 * (n**2 - 1) / 12

# divisor = - (n**2 * (n**2 - 1)) / 12

# -----------------------------------------------------------------------------------
#
#   LR_EXP_DEV - Linear Regression - Deviation from expected value
#
# -----------------------------------------------------------------------------------
_spec_func_numba = numba.types.Array(numba.float32, 1, 'C')(
                                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True), 
                                    numba.int32, 
                                    numba.int32, )

_locals_func_numba = {
                        'period_calc':   numba.int32,                        
                        'divisor':       numba.float32,
                        'sum_x':         numba.float32,
                        'sum_y':         numba.float32,
                        'sum_xy':        numba.float32,
                        'i_shifted':     numba.int32,
                        'slope':         numba.float32,
                        'intercept':     numba.float32,
                        'res_deviation': numba.float32,
                            }


@numba.njit(_spec_func_numba,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            locals      = _locals_func_numba,
                )
def get_lr_exp_dev(
                    data_arr:        np.ndarray[np.float32],
                    period:          np.int32,
                    expected_period: np.int32,
                        ) -> np.ndarray[np.float32]:
    '''
    Calculate Linear Regression - Deviation from expected value.
    
    ```python
    slope     = ((n * sum_xy) - (sum_x * sum_y)) / divisor
    intercept = (sum_y - slope * sum_x) / n
    y         = slope * x + intercept
    divisor   = (sum_x * sum_x) - n * sum_x^2
    sum_x     = (n * (n + 1)) / 2
    sum_x^2   = (n * (n + 1) * (2 * n + 1)) / 6
    divisor   = - (n**2 * (n - 1)**2) / 12
    
    expected_price  = slope * (period + expected_period) + intercept
    deviation       = (price - expected_price) / expected_price = 
                    = price / expected_price - 1.0 = Ln(price / expected_price)
    ```
    
    Parameters:
    -----------
    data_arr:        (`np.ndarray[np.float32]`) : Input data array.
    period:          (`np.int32`)               : Period of Linear Regression.
        **( period >= 2 )**
    expected_period: (`np.int32`)               : Period of expected value, 
        aftre the Linear Regression, 
        to which we project price.
        NOT ADVISED to use 'expected_period' > 'period'
        e.g.    if expected_period = 1, we project price to the next period.
                if expected_period = 2, we project price to the next 2 periods.
    
    Returns:
    --------
    (`np.ndarray[np.float32]`) : Deviation from expected value.
    
    ---
    
    Example:
    --------
    >>> data_arr:        np.ndarray[np.float32] = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype = np.float32)
    >>> period:          np.int32               = 2
    >>> expected_period: np.int32               = 1
    >>> res_arr:         np.ndarray[np.float32] = get_lr_exp_dev(data_arr, period, expected_period)
    >>> res_val:         np.float32             = res_arr[-1]
    >>> # res_val   - in range [-1.0 .. 1.0],
    >>> #       Usually around [-0.3 .. 0.3] for resanoable periods.    
    '''    
    
    result_arr:  np.ndarray[np.float32] = np.empty_like(data_arr, dtype = np.float32)    
    
    period_calc: np.int32   = period + expected_period    
    divisor:     np.float32 = (period**2 * (period - 1.0)**2) / 12.0  # NOTE: Change the sign to positive, for slope follow the trend.
    sum_x:       np.float32 = (period * (period + 1.0)) / 2.0
    
    result_arr[:period_calc] = 0.0 # Default value for the first period.
    
    for i in range(period_calc, len(data_arr)):
        
        sum_xy = 0.0
        sum_y  = 0.0        
        i_shifted: np.int32 = i - period_calc + 1
        
        for j in range(period):            
            sum_y  += data_arr[i_shifted + j]
            sum_xy += (j + 1) * data_arr[i_shifted + j]
            pass        

        slope         = (period * sum_xy - sum_x * sum_y) / divisor
        intercept     = (sum_y - slope * sum_x) / float(period)
        res_deviation = slope * period_calc + intercept  # Expected price.
        
        # NOTE: deviation = (expected_price - price) / price = expected_price / price - 1.0 = Ln(expected_price / price)
        res_deviation = 1.0 - res_deviation / data_arr[i]        
        result_arr[i] = res_deviation
        
        # Boundary Clipping:
        if res_deviation > 1.0:
            result_arr[i] = 1.0
        elif res_deviation < -1.0:
            result_arr[i] = -1.0
    
    return result_arr

# -----------------------------------------------------------------------------------
#
#  LR_EXP_DEV - MINI ( With less inputs parameters, like in other TI functions )
#
# -----------------------------------------------------------------------------------

_spec_func_numba = numba.types.Array(numba.float32, 1, 'C')(
                                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True), 
                                    numba.int32, )

_locals_func_numba = {
                        'period':           numba.int32,
                        'expected_period':  numba.int32,
                        'period_calc':      numba.int32,                        
                        'divisor':          numba.float32,
                        'sum_x':            numba.float32,
                        'sum_y':            numba.float32,
                        'sum_xy':           numba.float32,
                        'i_shifted':        numba.int32,
                        'slope':            numba.float32,
                        'intercept':        numba.float32,
                        'res_deviation':    numba.float32,                        
                            }

@numba.njit(_spec_func_numba,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            inline      = 'always',
            locals      = _locals_func_numba,
                )
def get_lr_exp_dev_mini(
                        data_arr:   np.ndarray[np.float32],
                        period_mix: np.int32,
                            ) -> np.ndarray[np.float32]:
    '''
    Calculate Linear Regression - Deviation from expected value.
    
    ```python
    slope     = ((n * sum_xy) - (sum_x * sum_y)) / divisor
    intercept = (sum_y - slope * sum_x) / n
    y         = slope * x + intercept
    divisor   = (sum_x * sum_x) - n * sum_x^2
    sum_x     = (n * (n + 1)) / 2
    sum_x^2   = (n * (n + 1) * (2 * n + 1)) / 6
    divisor   = - (n**2 * (n - 1)**2) / 12
    
    expected_price  = slope * (period + expected_period) + intercept
    deviation       = (price - expected_price) / expected_price = 
                    = price / expected_price - 1.0 = Ln(price / expected_price)
    ```
    
    ---
    
    Parameters:
    -----------
    data_arr:   (`np.ndarray[np.float32]`) : Input data array.
    period_mix: (`np.int32`)               : combined code of period and expected_period.
        first 3 digits - period, 
        thousands      - expected_period.
        e.g. period_mix =  15    -> period = 15, expected_period = 0
        e.g. period_mix =  1_002 -> period = 2,  expected_period = 1
        e.g. period_mix = 11_010 -> period = 10, expected_period = 11
    
    Returns:
    --------
    (`np.ndarray[np.float32]`) : Deviation from expected value.
    
    result_val - in range `([-1.0 .. 1.0])`,
    Usually around `([-0.3 .. 0.3])` for resanoable periods.
    '''
    
    period:          np.int32 = period_mix % 1000
    expected_period: np.int32 = period_mix // 1000
    
    return get_lr_exp_dev(data_arr, period, expected_period)

# -----------------------------------------------------------------------------------
#
#  LR_EXP_DEV - MINI, (tsf) - Thread Safe Function
#
# -----------------------------------------------------------------------------------

_signature_tsf = numba.void(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True,  aligned = True),
                    numba.int32,
                    numba.int32,
                    numba.types.Array(numba.float32, 1, 'C', readonly = False, aligned = True), )

_locals_tsf = {
                'period':           numba.int32,
                'expected_period':  numba.int32,
                'period_calc':      numba.int32,
                'divisor':          numba.float32,
                'sum_x':            numba.float32,
                'sum_y':            numba.float32,
                'sum_xy':           numba.float32,
                'i_shifted':        numba.int32,
                'slope':            numba.float32,
                'intercept':        numba.float32,
                'res_deviation':    numba.float32,
                'data_temp':        numba.float32,
                'data_temp_shft':   numba.float32, }
    
@numba.njit(_signature_tsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            locals      = _locals_tsf,
                )
def get_lr_exp_dev_mini_tsf(
                            data_arr:   np.ndarray[np.float32], 
                            period_mix: np.int32,
                            data_size:  np.int32,
                            result_arr: np.ndarray[np.float32],                  
                                    ) -> None:
    '''
    Calculate Linear Regression - Deviation from expected value.
    Thread Safe Function.
    
    Parameters:
    -----------
    data_arr:   (`np.ndarray[np.float32]`) : Input data array.
    period_mix: (`np.int32`)               : combined code of period and expected_period.
        first 3 digits - period, 
        thousands      - expected_period.
        e.g. period_mix =  15    -> period = 15, expected_period = 0
        e.g. period_mix =  1_002 -> period = 2,  expected_period = 1
        e.g. period_mix = 11_010 -> period = 10, expected_period = 11
    data_size:  (`np.int32`)               : Data size.
    result_arr: (`np.ndarray[np.float32]`) : Result array.
        Result array will be filled with the computed array.    
    '''
    
    if data_size < 0:
        data_size = data_arr.shape[0]
    
    period:          np.int32 = period_mix % 1000
    expected_period: np.int32 = period_mix // 1000    
    period_calc:     np.int32 = period + expected_period
    
    divisor: np.float32 = (period**2 * (period - 1.0)**2) / 12.0  # NOTE: Change the sign to positive, for slope follow the trend.
    sum_x:   np.float32 = (period * (period + 1.0)) / 2.0
    
    for i in range(period_calc):
        result_arr[i] = 0.0 # Default value for the first period.
    
    
    for i in range(period_calc, data_size):
        
        sum_xy = 0.0
        sum_y  = 0.0        
        i_shifted: np.int32 = i - period_calc + 1
        
        for j in range(period):          
            data_temp_shft = data_arr[i_shifted + j]  
            sum_y         += data_temp_shft
            sum_xy        += (j + 1) * data_temp_shft
        

        slope         = (period * sum_xy - sum_x * sum_y) / divisor
        intercept     = (sum_y - slope * sum_x) / float(period)
        res_deviation = slope * period_calc + intercept  # Expected price.
        
        # NOTE: deviation = (expected_price - price) / price = expected_price / price - 1.0 = Ln(expected_price / price)
        data_temp     = data_arr[i]
        res_deviation = 1.0 - res_deviation / data_temp
        res_deviation = max(-1.0, min(1.0, res_deviation))  # Boundary Clipping.
        result_arr[i] = res_deviation
        pass
    
    return

# -----------------------------------------------------------------------------------
#
#  LR_EXP_DEV - MINI, (vtsf) - Single Value calculation, Thread Safe Function
#
# -----------------------------------------------------------------------------------

_spec_func_vtsf = numba.types.float32(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True),
                    numba.int32,
                    numba.int32, )

_locals_vtsf = {
                'period':           numba.int32,
                'expected_period':  numba.int32,
                'period_calc':      numba.int32,
                'divisor':          numba.float32,
                'sum_x':            numba.float32,
                'sum_y':            numba.float32,
                'sum_xy':           numba.float32,
                'i_shifted':        numba.int32,
                'slope':            numba.float32,
                'intercept':        numba.float32,
                'res_deviation':    numba.float32,
                'data_temp':        numba.float32,
                'data_temp_shft':   numba.float32, }

@numba.njit(_spec_func_vtsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,            
            locals      = _locals_vtsf,  )
def get_lr_exp_dev_mini_vtsf(
                                data_arr:   np.ndarray[np.float32],
                                period_mix: np.int32,
                                data_indx:  np.int32,
                                        ) -> np.float32:
    '''
    Linear Regression - Deviation from expected value.
    Single Value calculation, Thread Safe Function.
    
    Parameters:
    -----------
    data_arr:   (`np.ndarray[np.float32]`) : Input data array.
    period_mix: (`np.int32`)               : combined code of period and expected_period.
        first 3 digits - period, 
        thousands      - expected_period.
        e.g. period_mix =  15    -> period = 15, expected_period = 0
        e.g. period_mix =  1_002 -> period = 2,  expected_period = 1
        e.g. period_mix = 11_010 -> period = 10, expected_period = 11
    data_indx:  (`np.int32`)               : Data index.
    
    Returns:
    --------
    (`np.float32`) : Deviation from expected value.
    
    result_val - in range `([-1.0 .. 1.0])`,
    Usually around `([-0.3 .. 0.3])` for resanoable periods.    
    '''
    
    period:          np.int32   = period_mix % 1000
    expected_period: np.int32   = period_mix // 1000    
    period_calc:     np.int32   = period + expected_period    
    divisor:         np.float32 = (period**2 * (period - 1.0)**2) / 12.0  # NOTE: Change the sign to positive, for slope follow the trend.
    sum_x:           np.float32 = (period * (period + 1.0)) / 2.0
    sum_xy:          float      = 0.0
    sum_y:           float      = 0.0    
    data_temp:       np.float32 = 0.0
    res_deviation:   np.float32 = 0.0    
    start_indx_shft: int        = data_indx - period_calc + 1

    for j in range(period):                
        data_temp: float = data_arr[start_indx_shft + j]  
        sum_y           += data_temp
        sum_xy          += (j + 1) * data_temp
    
    slope:     float = (period * sum_xy - sum_x * sum_y) / divisor
    intercept: float = (sum_y - slope * sum_x) / float(period)            
    data_temp: float = data_arr[data_indx]        
    # NOTE: deviation = (expected_price - price) / price = expected_price / price - 1.0 = Ln(expected_price / price)
    res_deviation: float = slope * period_calc + intercept  # Expected price.            
    res_deviation        = 1.0 - res_deviation / data_temp
    func_val_temp        = res_deviation
    
    if res_deviation > 1.0:
        func_val_temp = 1.0
    elif res_deviation < -1.0:
        func_val_temp = -1.0
    
    return func_val_temp

# -----------------------------------------------------------------------------------
#
#  LR_EXP_DEV - MINI, GPU version, single value calculation. 
#           NOTE: For use inside CUDA Functions only.
#
# -----------------------------------------------------------------------------------

@cuda.jit()
def get_lr_exp_dev_mini_vtsf_cuda(
                    data_arr:   np.ndarray[np.float32],
                    period_mix: np.int32,
                    data_indx:  np.int32,
                    res_indx:   np.int32,
                    res_arr:    np.ndarray[np.float32],
                        ) -> None:
    '''
    Linear Regression - Deviation from expected value.
    
    Single Value calculation.
    
    GPU version.
    
    Parameters:
    -----------
    data_arr:   (`np.ndarray[np.float32]`) : Input data array.
    period_mix: (`np.int32`)               : combined code of period and expected_period.
        first 3 digits - period,
        thousands      - expected_period.
        e.g. period_mix =  15    -> period = 15, expected_period = 0
        e.g. period_mix =  1_002 -> period = 2,  expected_period = 1
        e.g. period_mix = 11_010 -> period = 10, expected_period = 11
    data_indx:  (`np.int32`)               : Data index.
    res_indx:   (`np.int32`)               : Result index.
    res_arr:    (`np.ndarray[np.float32]`) : Result array.
        Single result value will be stored in the 'res_arr' at the 'res_indx'.
        
    '''
    
    period:          int   = period_mix % 1000
    expected_period: int   = period_mix // 1000    
    period_calc:     int   = period + expected_period    
    divisor:         float = (period**2 * (period - 1.0)**2) / 12.0  # NOTE: Change the sign to positive, for slope follow the trend.
    sum_x:           float = (period * (period + 1.0)) / 2.0
    sum_xy:          float = 0.0
    sum_y:           float = 0.0    
    data_temp:       float = 0.0
    res_deviation:   float = 0.0
    
    start_indx_shft: int = data_indx - period_calc + 1

    for j in range(period):                
        data_temp: float = data_arr[start_indx_shft + j]  
        sum_y           += data_temp
        sum_xy          += (j + 1) * data_temp
    
    slope:     float = (period * sum_xy - sum_x * sum_y) / divisor
    intercept: float = (sum_y - slope * sum_x) / float(period)            
    data_temp: float = data_arr[data_indx]        
    # NOTE: deviation = (expected_price - price) / price = expected_price / price - 1.0 = Ln(expected_price / price)
    res_deviation: float = slope * period_calc + intercept  # Expected price.            
    res_deviation        = 1.0 - res_deviation / data_temp    
    func_val_temp        = res_deviation
    
    if res_deviation > 1.0:
        func_val_temp = 1.0
    elif res_deviation < -1.0:
        func_val_temp = -1.0
    
    res_arr[res_indx] = func_val_temp
    
    return

# -----------------------------------------------------------------------------------
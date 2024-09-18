import numpy as np
import numba
from numba import cuda


# -----------------------------------------------------------------------------------

_name_:           str = 'Linear Regression - Slope'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# --- CALCULATION DESCRIPTION: -------------------------------------------------------

# Calculate Linear Regression Slope indicator.

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
#       Linear Regression Slope Calculation:
#
# -----------------------------------------------------------------------------------

_spec_func_numba = numba.types.Array(numba.float32, 1, 'C')(
                                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True),
                                    numba.int32)

_locals_func_numba = {
                        'divisor':   numba.float32,
                        'sum_x':     numba.float32,
                        'sum_y':     numba.float32,
                        'sum_xy':    numba.float32,
                        'i_shifted': numba.int32,         
                        'x_temp':    numba.float32,
                        'atan_val':  numba.float32,             
                            }

@numba.njit(_spec_func_numba,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            locals      = _locals_func_numba, )
def get_lr_slope(
                    data_arr: np.ndarray[np.float32],
                    period: np.int32,
                        ) -> np.ndarray[np.float32]:
    '''
    Get Linear Regression Slope indicator array.
    
    Parameters:
    -----------
    data_arr: (`np.ndarray[np.float32]`) : Input data array.
    period:   (`np.int32`)               : Period of linear regression.
    
    Returns:
    --------
    (`np.ndarray[np.float32]`) : Linear Regression Slope indicator array.
    
    Result is '( ArcTan(Slope) * (0.5 / Pi) )'
    
    result_val - in range '([-1.0 .. 1.0])'    
    '''
    
    result_arr: np.ndarray[np.float32] = np.empty_like(data_arr, dtype = np.float32)    
    
    divisor: np.float32 = (period**2 * (period - 1.0)**2) / 12.0  # Change the sign to positive, for slope follow the trend.
    sum_x:   np.float32 = (period * (period + 1.0)) / 2.0
    
    result_arr[:period] = 0.0 # Default value for the first period.
    
    for i in range(period, len(data_arr)):
        
        sum_xy = 0.0
        sum_y  = 0.0
        
        i_shifted: np.int32 = i - period + 1
        
        for j in range(period):
            sum_y  += data_arr[i_shifted + j]
            sum_xy += (j + 1) * data_arr[i_shifted + j]
            pass
    
        lr_slope_temp: np.float32 = (period * sum_xy - sum_x * sum_y) / divisor
        
        # ArcTan(Slope) / (0.5 * PI)
        # atan normalized to -1 to 1 calculation:
        x_temp: np.float32 = abs(lr_slope_temp)
        if x_temp > 1:
            x_temp = 1.0 / x_temp
        
        atan_val: np.float32 = x_temp - (x_temp**3 / 3) + (x_temp**5 / 5) - (x_temp**7 / 7) # NOTE: Taylor series for ArcTan(x)
        atan_val            *= 0.6366       # NOTE: 0.6366 = 1.0 / (np.pi / 2.0)
        
        if abs(lr_slope_temp) > 1:
            atan_val = 1.0 - atan_val    
        if lr_slope_temp < 0:
            atan_val = -atan_val    
        
        result_arr[i] = atan_val
        pass
    
    return result_arr

# -----------------------------------------------------------------------------------
#
#       Linear Regression Slope Indicator (tsf) - Thread Safe Function:
#
# -----------------------------------------------------------------------------------

_signature_tsf = numba.void(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True,  aligned = True),
                    numba.int32,
                    numba.int32,
                    numba.types.Array(numba.float32, 1, 'C', readonly = False, aligned = True), )

_locals_tsf = {
                'divisor':       numba.float32,
                'sum_x':         numba.float32,
                'sum_y':         numba.float32,
                'sum_xy':        numba.float32,
                'i_shifted':     numba.int32,                
                'data_temp':     numba.float32,
                'res_val':       numba.float32,
                'x_temp':        numba.float32,
                'atan_val':      numba.float32,
                'lr_slope_temp': numba.float32,
                }

@numba.njit(_signature_tsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            locals      = _locals_tsf, )
def get_lr_slope_tsf(
                        data_arr:   np.ndarray[np.float32], 
                        period:     np.int32,
                        data_size:  np.int32,
                        result_arr: np.ndarray[np.float32],
                            ) -> None:
    '''
    Get Linear Regression Slope indicator array.
    
    Parameters:
    -----------
    data_arr:   (`np.ndarray[np.float32]`) : Input data array.
    period:     (`np.int32`)               : Period of linear regression.
    data_size:  (`np.int32`)               : Data size.
    result_arr: (`np.ndarray[np.float32]`) : Result array.
        Result is '( ArcTan(Slope) * (0.5 / Pi) )'
        result_val - in range '([-1.0 .. 1.0])'
        Result array must be pre-allocated.
        Result array will be filled with the computed array.
    
    '''
    
    if data_size < 0:
        data_size = data_arr.shape[0]
    
    divisor: np.float32 = (period**2 * (period - 1.0)**2) / 12.0  # Change the sign to positive, for slope follow the trend.
    sum_x:   np.float32 = (period * (period + 1.0)) / 2.0
    
    for i in range(period):
        result_arr[i] = 0.0  # Default value for the first period.
    
    for i in range(period, data_size):
        
        sum_xy:    np.float32 = 0.0
        sum_y:     np.float32 = 0.0        
        i_shifted: np.int32   = i - period + 1
        
        for j in range(period):
            data_temp = data_arr[i_shifted + j]
            sum_y    += data_temp
            sum_xy   += (j + 1) * data_temp
            pass
        
        lr_slope_temp: np.float32 = (period * sum_xy - sum_x * sum_y) / divisor
        
        # ArcTan(Slope) / (0.5 * PI)
        # atan normalized to -1 to 1 calculation:
        x_temp: np.float32 = abs(lr_slope_temp)
        if x_temp > 1:
            x_temp = 1.0 / x_temp
        
        atan_val: np.float32 = x_temp - (x_temp**3 / 3) + (x_temp**5 / 5) - (x_temp**7 / 7) # NOTE: Taylor series for ArcTan(x)
        atan_val            *= 0.6366      # NOTE: 0.6366 = 1.0 / (np.pi / 2.0)
        
        if abs(lr_slope_temp) > 1:
            atan_val = 1.0 - atan_val    
        if lr_slope_temp < 0:
            atan_val = -atan_val    
                
        result_arr[i] = atan_val
        pass
    
    return

# -----------------------------------------------------------------------------------
#
#  Linear Regression Slope (vtsf) - Single Value Calculation, Thread Safe Function:
#
# -----------------------------------------------------------------------------------

_spec_func_vtsf = numba.types.float32(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True),
                    numba.int32,
                    numba.int32, )

_locals_vtsf = {
                'divisor':   numba.float32,
                'sum_x':     numba.float32,
                'sum_y':     numba.float32,
                'sum_xy':    numba.float32,
                'i_shifted': numba.int32,                
                'data_temp': numba.float32,
                'res_val':   numba.float32,
                    }

@numba.njit(_spec_func_vtsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,            
            locals      = _locals_vtsf,  )
def get_lr_slope_vtsf(
                        data_arr:  np.ndarray[np.float32],
                        period:    np.int32,
                        data_indx: np.int32,
                            ) -> np.float32:
    '''
    Get Linear Regression Slope indicator value.
    
    Single Value Calculation.
    
    Parameters:
    -----------
    data_arr:  (`np.ndarray[np.float32]`) : Input data array.
    period:    (`np.int32`)               : Period of linear regression.
    data_indx: (`np.int32`)               : Index of the current data.
    
    Returns:
    --------
    (`np.float32`) : Linear Regression Slope indicator value.
    '''

    divisor: np.float32 = (period**2 * (period - 1.0)**2) / 12.0  # Change the sign to positive, for slope follow the trend.
    sum_x:   np.float32 = (period * (period + 1.0)) / 2.0
    sum_xy:  np.float32 = 0.0
    sum_y:   np.float32 = 0.0   
    
    start_indx_shft: np.int32 = data_indx - period + 1
    
    for j in range(period):                
        data_temp: np.float32 = data_arr[start_indx_shft + j]
        sum_y  += data_temp
        sum_xy += (j + 1) * data_temp

    lr_slope_temp = (period * sum_xy - sum_x * sum_y) / divisor
    
    # atan normalized to -1 to 1 calculation:
    x_temp: np.float32 = abs(lr_slope_temp)
    if x_temp > 1:
        x_temp = 1.0 / x_temp
    
    atan_val: np.float32 = x_temp - (x_temp**3 / 3) + (x_temp**5 / 5) - (x_temp**7 / 7) # NOTE: Taylor series for ArcTan(x)
    atan_val            *= 0.6366       # NOTE: 0.6366 = 1.0 / (np.pi / 2.0)
    
    if abs(lr_slope_temp) > 1:
        atan_val = 1.0 - atan_val    
    if lr_slope_temp < 0:
        atan_val = -atan_val    
        
    return atan_val

# -----------------------------------------------------------------------------------
#
#   Linear Regression Slope Indicator (CUDA) - Single Value Calculation.
#        NOTE: For use inside CUDA Functions only.
#
# -----------------------------------------------------------------------------------

@cuda.jit()
def get_lr_slope_vtsf_cuda(
                    data_arr:   np.ndarray[np.float32],
                    period:     np.int32,
                    data_indx:  np.int32,
                    res_indx:   np.int32,
                    res_arr:    np.ndarray[np.float32],
                        ) -> None:
    '''
    Get Linear Regression Slope indicator value.
    
    Single Value Calculation.
    
    CUDA Version.
    
    Parameters:
    -----------
    data_arr:  (`np.ndarray[np.float32]`) : Input data array.
    period:    (`np.int32`)               : Period of linear regression.
    data_indx: (`np.int32`)               : Index of the current data.
    res_indx:  (`np.int32`)               : Index of the result array.
    res_arr:   (`np.ndarray[np.float32]`) : Result array.
        Will ve uodated with the calculated single value.
    
    '''

    divisor: float = (period**2 * (period - 1.0)**2) / 12.0  # Change the sign to positive, for slope follow the trend.
    sum_x:   float = (period * (period + 1.0)) / 2.0    
    sum_xy:  float = 0.0
    sum_y:   float = 0.0   
    
    start_indx_shft: int = data_indx - period + 1
    
    for j in range(period):                
        data_temp: float = data_arr[start_indx_shft + j]
        sum_y           += data_temp
        sum_xy          += (j + 1) * data_temp

    lr_slope_temp = (period * sum_xy - sum_x * sum_y) / divisor
    
    # atan normalized to -1 to 1 calculation:
    x_temp: float = abs(lr_slope_temp)
    if x_temp > 1:
        x_temp = 1.0 / x_temp
    
    atan_val: float = x_temp - (x_temp**3 / 3) + (x_temp**5 / 5) - (x_temp**7 / 7) # NOTE: Taylor series for ArcTan(x)
    atan_val       *= 0.6366          # NOTE: 0.6366 = 1.0 / (np.pi / 2.0)
    
    if abs(lr_slope_temp) > 1:
        atan_val = 1.0 - atan_val    
    if lr_slope_temp < 0:
        atan_val = -atan_val    
        
    res_arr[res_indx] = atan_val
    
    return atan_val

# -----------------------------------------------------------------------------------

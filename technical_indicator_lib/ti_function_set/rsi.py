import numpy as np
import numba
from numba import cuda


# -----------------------------------------------------------------------------------

_name_:           str = 'RSI - TI Lib'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# --- CALCULATION DESCRIPTION: -------------------------------------------------------

# Calculate RSI
# rsi = 100.0 - (100.0 / (1.0 + accum_gain / accum_loss))

# rsi = 100.0 * (1 - 1 / (1 + accum_gain / accum_loss))
# rsi = 100.0 * (((1 + accum_gain / accum_loss) - 1) / (1 + accum_gain / accum_loss))
# rsi = 100.0 * ((accum_gain / accum_loss) / (1 + accum_gain / accum_loss))
# rsi = 100.0 * (accum_gain / accum_loss) / ((accum_loss + accum_gain) / accum_loss)
# rsi = 100.0 * (accum_gain) / ((accum_loss + accum_gain))
# rsi = 100.0 * (accum_gain) / (accum_loss + accum_gain)

# -----------------------------------------------------------------------------------
#
#               RSI: Relative Strength Index
#
# -----------------------------------------------------------------------------------

_spec_func_numba = numba.types.Array(numba.float32, 1, 'C')(
                numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True), 
                numba.int32)

_locals_func_numba = {                       
        'accum_gain':   numba.float32,
        'accum_loss':   numba.float32,
        'multiplier':   numba.float32,
        'multiplier_2': numba.float32,
        'diff':         numba.float32,  }

@numba.njit(_spec_func_numba,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            inline      = 'always',
            locals      = _locals_func_numba, )
def get_rsi(
            data:   np.ndarray[np.float32],             
            period: np.int32, 
                ) -> np.ndarray[np.float32]:
    '''
    Get RSI of the given array.
    
    ---
    
    Parameters:
    -----------
    data_arr: (`np.ndarray[np.float32]`) : array of values.
    period:   (`np.int32`)               : period of RSI.
    
    Returns:
    --------
    result: (`np.ndarray[np.float32]`) : array of RSI.
    '(result[i] - in range [0.0 .. 100.0])'    
    '''
    
    multiplier:   np.float32 = 1.0 / period
    multiplier_2: np.float32 = 1.0 - multiplier
    
    result_arr: np.ndarray[np.float32] = np.empty_like(data, dtype = np.float32)
    
    # Calculate initial average gain and loss.
    accum_gain: np.float32 = 0.0
    accum_loss: np.float32 = 0.0
        
    result_arr[0] = 50.0 # Default value for the first period.
    
    for i in range(1, period):
        
        result_arr[i] = 50.0 # Default value for the first period.
        
        diff = data[i] - data[i - 1]
        if diff > 0.0:
            accum_gain += diff
        else:
            accum_loss -= diff
            pass
        
        pass

    accum_gain *= multiplier
    accum_loss *= multiplier
    
    for i in range(period, len(data)):
        diff        = data[i] - data[i - 1]        
        accum_gain *= multiplier_2
        accum_loss *= multiplier_2                
        diff       *= multiplier  # NOTE: Diff scaled in advanced here.
        
        if diff > 0.0:
            accum_gain += diff
        else:
            accum_loss -= diff
        
        res_val: np.float32 = 50.0
        
        accum_range: np.float32 = accum_gain + accum_loss
        if accum_range > 0.0:
            res_val = 100.0 * accum_gain / accum_range
        
        result_arr[i] = res_val
    
    return result_arr

# -----------------------------------------------------------------------------------
#
#             RSI (tsf) - Thread Safe Function
#
# -----------------------------------------------------------------------------------

_signature_tsf = numba.void(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True),
                    numba.int32,
                    numba.int32,
                    numba.types.Array(numba.float32, 1, 'C', readonly = False, aligned = True), )

_locals_tsf = {
        'multiplier':   numba.float32,
        'multiplier_2': numba.float32,
        'diff':         numba.float32,
        'accum_gain':   numba.float32,
        'accum_loss':   numba.float32,
        'res_val':      numba.float32, }

@numba.njit(_signature_tsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            locals      = _locals_tsf, )
def get_rsi_tsf(
                data:       np.ndarray[np.float32],             
                period:     np.int32, 
                data_size:  np.int32,
                result_arr: np.ndarray[np.float32],                  
                    ) -> None:
    '''
    Get RSI of the given array.
    
    Thread Safe Function.
    
    ---
    
    Parameters:
    -----------
    data:       (`np.ndarray[np.float32]`) : array of values.
    period:     (`np.int32`)               : period of RSI.
    data_size:  (`np.int32`)               : size of data array.
    result_arr: (`np.ndarray[np.float32]`) : array of RSI.
        Results will be rewritten in this array.
    
    '''

    if data_size < 0:
        data_size = data.shape[0]
    
    multiplier:   np.float32 = 1.0 / float(period)
    multiplier_2: np.float32 = 1.0 - multiplier
    
    # Calculate initial average gain and loss.
    accum_gain: np.float32 = 0.0
    accum_loss: np.float32 = 0.0
        
    result_arr[0] = 50.0        # Default value for the first period.
    for i in range(1, period):
        
        result_arr[i] = 50.0        # Default value for the first period.        
        diff          = data[i] - data[i - 1]
        
        if diff > 0.0:
            accum_gain += diff
        else:
            accum_loss -= diff
        
    accum_gain *= multiplier
    accum_loss *= multiplier
    
    for i in range(period, data_size):
        diff = data[i] - data[i - 1]
        
        accum_gain *= multiplier_2
        accum_loss *= multiplier_2                
        diff       *= multiplier  # NOTE: Diff scaled in advanced here.
        
        if diff > 0.0:
            accum_gain += diff
        else:
            accum_loss -= diff
        
        res_val:     np.float32 = 50.0                
        accum_range: np.float32 = accum_gain + accum_loss
        
        if accum_range > 0.0:
            res_val = 100.0 * accum_gain / accum_range
        
        result_arr[i] = res_val
    
    return

# -----------------------------------------------------------------------------------
#
#             RSI (vtsf) - Single Value, Thread Safe Function
#
# -----------------------------------------------------------------------------------

_spec_func_vtsf = numba.types.float32(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True),
                    numba.int32,
                    numba.int32, )

_locals_vtsf = {
        'multiplier':   numba.float32,
        'multiplier_2': numba.float32,
        'diff':         numba.float32,
        'accum_gain':   numba.float32,
        'accum_loss':   numba.float32,
        'accum_range':  numba.float32,
        'res_val':      numba.float32, }

@numba.njit(_spec_func_vtsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,            
            locals      = _locals_vtsf,)
def get_rsi_vtsf(
                    data:       np.ndarray[np.float32],
                    period:     np.int32,
                    data_indx:  np.int32,
                        ) -> np.float32:
    '''
    Get RSI Single Value.
    
    Thread Safe Function.
    
    ---
    
    Parameters:
    -----------
    data:       (`np.ndarray[np.float32]`) : array of values.
    period:     (`np.int32`)               : period of RSI.
    data_indx:  (`np.int32`)               : Data index.
    
    Returns:
    --------
    (`np.float32`) : RSI Single Value.
    '''    

    multiplier:   np.float32 = 1.0 / float(period)
    multiplier_2: np.float32 = 1.0 - multiplier
    
    # Calculate initial average gain and loss.
    accum_gain: np.float32 = 0.0
    accum_loss: np.float32 = 0.0
    
    for i in range(1, period):
        
        diff = data[i] - data[i - 1]
        if diff > 0.0:
            accum_gain += diff
        else:
            accum_loss -= diff
            

    accum_gain *= multiplier
    accum_loss *= multiplier
    
    for i in range(period, data_indx + 1):
        diff = data[i] - data[i - 1]
        
        accum_gain *= multiplier_2
        accum_loss *= multiplier_2                
        diff       *= multiplier  # NOTE: Diff scaled in advanced here.
        
        if diff > 0.0:
            accum_gain += diff
        else:
            accum_loss -= diff
        
    res_val     = 50.0        
    accum_range = accum_gain + accum_loss
    
    if accum_range > 0:
        res_val: np.float32 = 100.0 * accum_gain / accum_range
        
    return res_val

# -----------------------------------------------------------------------------------
#
#             RSI (GPU) - NOTE: For use inside CUDA Functions only.
#
# -----------------------------------------------------------------------------------

@cuda.jit()
def get_rsi_vtsf_cuda(
                    data:       np.ndarray[np.float32],
                    period:     np.int32,
                    data_indx:  np.int32,
                    res_indx:   np.int32,
                    res_arr:    np.ndarray[np.float32],
                        ) -> None:

    multiplier:   float = 1.0 / float(period)
    multiplier_2: float = 1.0 - multiplier
    
    # Calculate initial average gain and loss.
    accum_gain: float = 0.0
    accum_loss: float = 0.0
    
    for i in range(1, period):
        
        diff = data[i] - data[i - 1]
        if diff > 0.0:
            accum_gain += diff
        else:
            accum_loss -= diff

    accum_gain *= multiplier
    accum_loss *= multiplier
    
    for i in range(period, data_indx + 1):
        diff = data[i] - data[i - 1]
        
        accum_gain *= multiplier_2
        accum_loss *= multiplier_2                
        diff       *= multiplier    # NOTE: Diff scaled in advanced here.
        
        if diff > 0.0:
            accum_gain += diff
        else:
            accum_loss -= diff
        
    res_val     = 50.0    
    accum_range = accum_gain + accum_loss
    
    if accum_range > 0.0:
        res_val: float = 100.0 * accum_gain / accum_range
        
    res_arr[res_indx] = res_val  
    
    return 

# -----------------------------------------------------------------------------------
import numpy as np
import numba
from numba import cuda


# -----------------------------------------------------------------------------------

_name_:           str = 'BB: Bollinger Bands'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# --- CALCULATION DESCRIPTION: -------------------------------------------------------

# Calculate Bollinger Bands.
#                     BB     = SMA +- 2 * STD
#                     BB_val = (Close - SMA) / 2 * STD

# -----------------------------------------------------------------------------------
#
#               BOLLINGER BANDS
#
# -----------------------------------------------------------------------------------

_spec_func_numba = numba.types.Array(numba.float32, 1, 'C')(
                                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True), 
                                    numba.int32,   )

_locals_func_numba = {
                        'accum':        numba.float32,
                        'accum_sq':     numba.float32, 
                        'multiplier_1': numba.float32, 
                        'bb_res':       numba.float32,                       
                        'sma':          numba.float32, 
                        'accum_var':    numba.float32, 
                        'std':          numba.float32, 
                            }


@numba.njit(_spec_func_numba,
            cache        = True, 
            fastmath     = True, 
            nogil        = True,
            boundscheck  = False,
            inline       = 'always',
            locals       = _locals_func_numba, )
def get_bb(
            data:   np.ndarray[np.float32], 
            period: np.int32,
                    ) -> np.ndarray[np.float32]:
    '''
    Get Bollinger Bands of the given array.
    
    ```python
    BB_val = (Close - SMA) / (std_range * STD)
    ```
    
    Parameters:
    -----------
    data:   (`np.ndarray[float]`) : Input data array.
    period: (`int`)               : Period.
    
    Returns:
    --------
    (`np.ndarray[float]`): Bollinger Bands result array.
    
    ---
    
    Example:
    --------
    >>> data:   np.ndarray[float] = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype = np.float32)
    >>> period: int = 5    
    >>> bb_res: np.ndarray[float] = get_bb(data, period)
    >>> # bb_res[i] - in range (-3.0 .. 3.0)
    '''
    
    result_arr = np.empty_like(data, dtype = np.float32)
    
    # Calculate initial average.
    accum:    np.float32 = 0.0
    accum_sq: np.float32 = 0.0
    
    for i in range(period):
        accum        += data[i]
        accum_sq     += data[i] ** 2
        result_arr[i] = 0.0     # Default value for the first period.
        pass

    multiplier_1: np.float32 = 1.0 / np.float32(period)    

    for i in range(period, len(data)):
        data_temp:     np.float32 = data[i]
        data_temp_old: np.float32 = data[i - period]        
        
        accum      += data_temp - data_temp_old
        accum_sq   += data_temp ** 2 - data_temp_old ** 2        
        sma         = accum * multiplier_1
        accum_var   = accum_sq * multiplier_1 - sma * sma
        std         = np.sqrt(accum_var)        
        diff: float = data_temp - sma        
        bb_res      = 3.0
        
        if abs(diff) < std * 3.0:
            bb_res = diff / std
        elif diff < 0:
            bb_res = -3.0
        
        result_arr[i] = bb_res
    
    return result_arr

# -----------------------------------------------------------------------------------
#
#               BOLLINGER BANDS (tsf) - Thread Safe Function
#
# -----------------------------------------------------------------------------------

_signature_tsf = numba.void(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True,  aligned = True),     # data
                    numba.int32,                                                                    # period                    
                    numba.int32,                                                                    # data_size                                 
                    numba.types.Array(numba.float32, 1, 'C', readonly = False, aligned = True), )   # result_arr

_locals_tsf = {
        'accum':         numba.float32,
        'accum_sq':      numba.float32,
        'data_temp':     numba.float32,
        'data_temp_old': numba.float32,
        'sma':           numba.float32,
        'accum_var':     numba.float32,
        'std':           numba.float32,
        'bb_res':        numba.float32,
        'multiplier_1':  numba.float32, }

@numba.njit(_signature_tsf,
            cache        = True, 
            fastmath     = True, 
            nogil        = True,
            boundscheck  = False,
            locals       = _locals_tsf, )
def get_bb_tsf( 
                data:       np.ndarray[np.float32], 
                period:     np.int32,
                data_size:  np.int32,
                result_arr: np.ndarray[np.float32],                  
                        ) -> None:
    '''
    Update `result_arr[]` with calulated Bollinger Bands values.
    
    Thread Safe Function.
    
    Parameters:
    -----------
    data:       (`np.ndarray[np.float32]`) : Input data array.
    period:     (`np.int32`)               : Period.
    data_size:  (`np.int32`)               : Size of data array.
    result_arr: (`np.ndarray[np.float32]`) : Result Array.
        Array updated with Bollinger Bands result values.    
    '''
    
    if data_size < 0:
        data_size = data.shape[0]
    
    # Calculate initial average.
    accum:    np.float32 = 0.0
    accum_sq: np.float32 = 0.0
    
    for i in range(period):
        data_temp: np.float32 = data[i]
        
        accum        += data_temp
        accum_sq     += data_temp ** 2
        result_arr[i] = 0.0         # Default value for the first period.
        pass

    multiplier_1: np.float32 = 1.0 / np.float32(period)    

    for i in range(period, data_size):
        data_temp:     np.float32 = data[i]
        data_temp_old: np.float32 = data[i - period]
        
        accum      += data_temp - data_temp_old
        accum_sq   += data_temp ** 2 - data_temp_old ** 2        
        sma         = accum * multiplier_1
        accum_var   = accum_sq * multiplier_1 - sma * sma
        std         = accum_var ** 0.5        
        diff: float = data_temp - sma        
        bb_res      = 3.0
        
        if abs(diff) < std * 3.0:
            bb_res = diff / std
        elif diff < 0:
            bb_res = -3.0
        
        result_arr[i] = bb_res
    
    return 

# -----------------------------------------------------------------------------------
#
#           BOLLINGER BANDS (vtsf) - Single Value calculation, Thread Safe Function
#
# -----------------------------------------------------------------------------------


_spec_func_vtsf = numba.types.float32(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True), # data
                    numba.int32,                                                               # period                      
                    numba.int32, )                                                             # data_indx

_locals_vtsf = {
        'accum':         numba.float32,
        'accum_sq':      numba.float32,
        'data_temp':     numba.float32,
        'data_temp_old': numba.float32,
        'sma':           numba.float32,
        'accum_var':     numba.float32,
        'std':           numba.float32,
        'bb_res':        numba.float32,
        'multiplier_1':  numba.float32, }

@numba.njit(_spec_func_vtsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,            
            locals      = _locals_vtsf, )
def get_bb_vtsf(
                data:      np.ndarray[np.float32],
                period:    np.int32,
                data_indx: np.int32,
                        ) -> np.float32:
    '''
    Calculate Bollinger Bands for the given index.
    
    Parameters:
    -----------
    data:      (`np.ndarray[[np.float32]`) : Input data array.
    period:    (`np.int32`)                : Period.
    data_indx: (`np.int32`)                : Index.
    
    Returns:
    --------
    (`np.float32`) : Bollinger Bands result value.
    '''
    
    # Calculate initial average.
    accum:     np.float32 = 0.0
    accum_sq:  np.float32 = 0.0    
    data_temp: np.float32 = 0.0
    
    for i in range(data_indx + 1 - period, data_indx + 1):
        data_temp = data[i]
        accum    += data_temp
        accum_sq += data_temp ** 2
        pass
    
    multiplier_1: np.float32 = 1.0 / np.float32(period)    
    
    sma         = accum * multiplier_1    
    accum_var   = accum_sq * multiplier_1 - sma * sma
    std         = accum_var ** 0.5    
    diff: float = data_temp - sma        
    bb_res      = 3.0
    
    if abs(diff) < std * 3.0:
        bb_res = diff / std
    elif diff < 0:
        bb_res = -3.0
    
    return bb_res

# -----------------------------------------------------------------------------------
#
#    BOLLINGER BANDS (GPU) - Single Value calculation
#        NOTE: For use inside CUDA Functions only.
#
# -----------------------------------------------------------------------------------

@cuda.jit
def get_bb_vtsf_cuda(
                    data:       np.ndarray[np.float32],
                    period:     np.int32,
                    data_indx:  np.int32,
                    res_indx:   np.int32,
                    res_arr:    np.ndarray[np.float32],
                        ) -> None:
    '''
    Get Bollinger Bands Single Value for the given index, by updating `res_arr[]`.
    GPU Version.
    UPDATE: `res_arr[]` with Bollinger Bands value.
    
    Parameters:
    -----------
    data:       (`np.ndarray[np.float32]`) : Input data array.
    period:     (`np.int32`)               : Period.
    data_indx:  (`np.int32`)               : Index.
    res_indx:   (`np.int32`)               : Result Index.
    res_arr:    (`np.ndarray[np.float32]`) : Result Array.
        Array updated with Bollinger Bands result value.
    '''
    
    # Calculate initial average.
    accum:     float = 0.0
    accum_sq:  float = 0.0    
    data_temp: float = 0.0
    
    for i in range(data_indx + 1 - period, data_indx + 1):
        data_temp = data[i]
        accum    += data_temp
        accum_sq += data_temp ** 2
        
    multiplier_1: float = 1.0 / np.float32(period)    
    
    sma       = accum * multiplier_1    
    accum_var = accum_sq * multiplier_1 - sma * sma
    std       = accum_var ** 0.5    
    diff      = data_temp - sma
    bb_res    = 3.0

    if abs(diff) < std * 3.0:
        bb_res = diff / std
    elif diff < 0:
        bb_res = -3.0
    
    res_arr[res_indx] = bb_res

    return

# -----------------------------------------------------------------------------------
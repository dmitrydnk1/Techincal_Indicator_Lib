import numpy as np
import numba
from numba import cuda


# -----------------------------------------------------------------------------------

_name_:           str = 'Aroon - TI Lib'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# --- CALCULATION DESCRIPTION: -----------------------------------------------------

# Calculate Aroon Indicator.

# Aroon Up = ((period - Days Since Period High) / period)
# Aroon Down = ((period - Days Since Period Low) / period)
# Aroon Oscillator = Aroon Up - Aroon Down = 
#           = Aroon_Oscillator = (low_index - high_index) / period

# Aroon indicator is a technical analysis indicator used to measure 
# the presence and strength of a trend. It consists of two lines: 
# Aroon Up and Aroon Down. 
# The Aroon Up measures the number of periods since the highest high, 
# and the Aroon Down measures the number of periods since the lowest low. 
# The Aroon Oscillator is then calculated as the 
# difference between Aroon Up and Aroon Down.

# -----------------------------------------------------------------------------------
#
#                 AROON - NUMBA:
#
# -----------------------------------------------------------------------------------

_spec_func_numba = numba.types.Array(numba.float32, 1, 'C')(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True),  # data_arr
                    numba.int32)                                                                # period

_locals_func_numba = {
        'period_multiplier': numba.float32,
        'high_index':        numba.int32,
        'low_index':         numba.int32,
        'i_shft':            numba.int32,
        'high_val':          numba.float32,
        'low_val':           numba.float32, }


@numba.njit(_spec_func_numba,
            cache        = True, 
            fastmath     = True, 
            nogil        = True,
            boundscheck  = False,
            locals       = _locals_func_numba,
                )
def get_aroon(
                data_arr: np.ndarray[np.float32],
                period:   np.int32,
                    ) -> np.ndarray[np.float32]:
    '''
    Get Aroon Indicator of the given array.
    ```python
    Aroon_Up         = ((period - Days_Since_Period_High) / period)
    Aroon_Down       = ((period - Days_Since_Period_Low) / period)
    Aroon_Oscillator = Aroon_Up - Aroon_Down
    Aroon_Oscillator = (low_index - high_index) / period
    ```
    ---
    
    Parameters:
    -----------
    data_arr: (`np.ndarray[np.float32]`) : array of values.
    period:   (`np.int32`)               : period of Aroon Indicator.
        Warning: **( period >= 2 )**
    
    Returns:
    --------
    (`np.ndarray[np.float32]`) : array of Aroon Indicator.
    result[i] - in range [-1.0 .. 1.0]
    '''
    
    result_arr: np.ndarray[np.float32] = np.empty_like(data_arr, dtype = np.float32)
    
    result_arr[:period] = 0.0  # Default value for the first period.
        
    period_multiplier: np.float32 = 1.0 / np.float32(period)

    for i in range(period, len(data_arr)):
        
        high_index: np.int32 = 0
        low_index:  np.int32 = 0
        i_shft:     np.int32 = i - period + 1        
        high_val: np.float32 = data_arr[i_shft]
        low_val:  np.float32 = data_arr[i_shft]
        
        for j in range(1, period):
            data_temp: np.float32 = data_arr[i_shft + j]
            if data_temp > high_val:
                high_val   = data_temp
                high_index = j
                
            if data_temp < low_val:
                low_val   = data_temp
                low_index = j
            
        result_arr[i] = period_multiplier * (low_index - high_index)
        
        pass
    
    return result_arr

# -----------------------------------------------------------------------------------
#
#              AROON (tsf) - Thread Safe Function
#
# -----------------------------------------------------------------------------------

_spec_func_numba =  numba.types.void(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True,  aligned = True),     # data_arr
                    numba.int32,                                                                    # period
                    numba.int32,                                                                    # data_size
                    numba.types.Array(numba.float32, 1, 'C', readonly = False, aligned = True), )   # result_arr

_locals_func_numba = {
        'period_multiplier': numba.float32,
        'high_index':        numba.int32,
        'low_index':         numba.int32,
        'i_shft':            numba.int32,
        'high_val':          numba.float32,
        'low_val':           numba.float32,
        'temp_res':          numba.float32, }

@numba.njit(_spec_func_numba,
            cache        = True, 
            fastmath     = True, 
            nogil        = True,
            boundscheck  = False,
            locals       = _locals_func_numba,  )
def get_aroon_tsf(  
                    data_arr:   np.ndarray[np.float32],
                    period:     np.int32,
                    data_size:  np.int32,
                    result_arr: np.ndarray[np.float32],                  
                        ) -> None:
    '''
    Get Aroon Indicator of the given array.
    Thread Safe Function.
    
    Function updates the `result_arr[]` with Aroon Indicator values.
    
    ```python
    Aroon_Up         = ((period - Days_Since_Period_High) / period)
    Aroon_Down       = ((period - Days_Since_Period_Low) / period)
    Aroon_Oscillator = Aroon_Up - Aroon_Down
    Aroon_Oscillator = (low_index - high_index) / period
    ```
    ---
    
    Parameters:
    -----------
    data_arr:   (`np.ndarray[np.float32]`) : array of values.
    period:     (`np.int32`)               : period of Aroon Indicator.
        Warning: **( period >= 2 )**
    data_size:  (`np.int32`)               : size of data array.
    result_arr: (`np.ndarray[np.float32]`) : array of Aroon Indicator.
        !!! UPDATING array with Arroon Indicator values. !!!
        Warning: **( len(result_arr) >= data_size )**
    '''
    
    if data_size < 0:
        data_size = data_arr.shape[0]
    
    period_multiplier: np.float32 = 1.0 / np.float32(period)
    
    for i in range(period):
        result_arr[i] = 0.0  # Default value for the first period.

    for i in range(period, data_size):
        
        high_index: np.int32 = 0
        low_index:  np.int32 = 0
        i_shft:     np.int32 = i - period + 1        
        high_val: np.float32 = data_arr[i_shft]
        low_val:  np.float32 = data_arr[i_shft]
        
        for j in range(1, period):
            data_temp: np.float32 = data_arr[i_shft + j]
            if data_temp > high_val:
                high_val   = data_temp
                high_index = j
                
            if data_temp < low_val:
                low_val   = data_temp
                low_index = j
        
        temp_res: np.float32 = period_multiplier * (low_index - high_index)
        result_arr[i]        = temp_res
    
    return

# -----------------------------------------------------------------------------------
#
#            AROON (vtsf) - Value, Thread Safe Function
#
# -----------------------------------------------------------------------------------

_spec_func_vtsf = numba.types.float32(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True),
                    numba.int32,
                    numba.int32, )

_locals_vtsf = {
        'period_multiplier': numba.float32,
        'start_indx':        numba.int32,
        'end_indx':          numba.int32,
        'high_index':        numba.int32,
        'low_index':         numba.int32,
        'i_shft':            numba.int32,
        'high_val':          numba.float32,
        'low_val':           numba.float32,
        'temp_res':          numba.float32,  }

@numba.njit(_spec_func_vtsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,            
            locals      = _locals_vtsf,)
def get_aroon_vtsf(
                    data_arr:  np.ndarray[np.float32],
                    period:    np.int32,
                    data_indx: np.int32,
                        ) -> np.float32:
    '''
    Get Aroon Indicator Single Value result.
    Thread Safe Function.
    
    Parameters:
    -----------
    data_arr:  (`np.ndarray[np.float32]`) : array of values.
    period:    (`np.int32`)               : period of Aroon Indicator.
        Warning: **( period >= 2 )**
    data_indx: (`np.int32`)               : index of the data array.
    
    Returns:
    --------
    (`np.float32`) : Aroon Indicator Single Value result.
    '''

    high_index:      np.int32   = 0
    low_index:       np.int32   = 0
    start_indx_shft: np.int32   = data_indx - period + 1
    data_temp:       np.float32 = data_arr[start_indx_shft]
    high_val:        np.float32 = data_temp
    low_val:         np.float32 = data_temp
    
    for i in range(1, period):   # from 1 to period, because first data is already calculated.
        data_temp = data_arr[start_indx_shft + i]
        if data_temp > high_val:
            high_val   = data_temp
            high_index = i
        
        if data_temp < low_val:
            low_val    = data_temp
            low_index  = i
    
    temp_res: np.float32 = float((low_index - high_index)) * (1.0 / float(period)) 
    
    return temp_res

# -----------------------------------------------------------------------------------
#
#            AROON - GPU Function:
#                   NOTE: For use inside CUDA Functions only.
#
# -----------------------------------------------------------------------------------

@cuda.jit()
def get_aroon_vtsf_cuda(
                    data_arr:   np.ndarray[np.float32],
                    period:     np.int32,
                    data_indx:  np.int32,
                    res_indx:   np.int32,
                    res_arr:    np.ndarray[np.float32],
                        ) -> None:
    '''
    Get Aroon Indicator array.
    GPU Version.
        
    Result updates the `res_arr[]` with Aroon Indicator values.
    
    Parameters:
    -----------
    data_arr:  (`np.ndarray[np.float32]`) : array of values.
    period:    (`np.int32`)               : period of Aroon Indicator.
        Warning: **( period >= 2 )**
    data_indx: (`np.int32`)               : index of the data array.
    res_indx:  (`np.int32`)               : index of the result array.
    res_arr:   (`np.ndarray[np.float32]`) : array of Aroon Indicator.
        !!! UPDATING array with Arroon Indicator values. !!!
        Warning: **( len(res_arr) >= data_size )**
    '''

    high_index:      int   = 0
    low_index:       int   = 0                            
    start_indx_shft: int   = data_indx - period + 1
    data_temp:       float = data_arr[start_indx_shft]
    high_val:        float = data_temp
    low_val:         float = data_temp
    
    for i in range(1, period):   # from 1 to period, because first data is already calculated.
        data_temp = data_arr[start_indx_shft + i]
        if data_temp > high_val:
            high_val   = data_temp
            high_index = i

        if data_temp < low_val:
            low_val    = data_temp
            low_index  = i

    temp_res: float   = float((low_index - high_index)) * (1.0 / float(period))    
    res_arr[res_indx] = temp_res
    
    return

# -----------------------------------------------------------------------------------
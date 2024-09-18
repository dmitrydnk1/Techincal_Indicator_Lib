import numpy as np
import numba
from numba import cuda


# -----------------------------------------------------------------------------------

_name_:           str = 'William %R OC - TI Lib'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# --- CALCULATION DESCRIPTION: -------------------------------------------------------

# The Williams %R indicator is a momentum indicator that 
#       measures overbought or oversold conditions in a financial market. 
# It is calculated using the following formula:
#             %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Calculate william_oc indicator.
#                     _oc = Only Close Flag. If True, then only Close prices will be used.
#                           William %R = (Highest High - Close) / (Highest High - Lowest Low) (! * -100 not used)
#                           Highest High = Highest High for the period.
#                           Lowest Low = Lowest Low for the period.
#                           n = 14 [period]

# -----------------------------------------------------------------------------------
#
#               William %R OC: william_oc
#
# -----------------------------------------------------------------------------------

_spec_func_numba = numba.types.Array(numba.float32, 1, 'C', aligned = True)(
                                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True), 
                                    numba.int32)

_locals_func_numba = {
        'lowest_low':       numba.float32,
        'highest_high':     numba.float32,
        'high_low_range':   numba.float32,  
        'i_shifted':        numba.int32,    }

@numba.njit(_spec_func_numba,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            inline      = 'always',
            locals      = _locals_func_numba, )
def get_william_oc(
                    data_arr: np.ndarray[np.float32],
                    period:   np.int32,
                        ) -> np.ndarray[np.float32]:
    '''
    Get William OC of the given array.
    
    OC:  Only Close Flag. Only Close prices will be used.
    
    ---
    
    Parameters:
    -----------
    data_arr: (`np.ndarray[np.float32]`) : array of values.
    period:   (`np.int32`)               : period of William OC.
        '(period >= 2)'
    
    Returns:
    --------
    (`np.ndarray[np.float32]`) : array of Willian OC.
    '(result[i] - in range [0.0 .. 1.0])'
    
    '''
    
    result_arr: np.ndarray[np.float32] = np.empty_like(data_arr, dtype = np.float32)
    
    # Calculate initial values.
    lowest_low:   np.float32 = data_arr[0]
    highest_high: np.float32 = data_arr[0]
    
    result_arr[0] = 0.5  # Default value for the first period. # NOTE: It's assigned later as well for the rest of the period.
    
    for i in range(1, period):
        
        result_arr[i] = 0.5    # Default value for the first period.
        lowest_low    = min(lowest_low, data_arr[i])
        highest_high  = max(highest_high, data_arr[i])

    # Calculate William %R.
    for i in range(period, len(data_arr)):
        
        i_shifted: np.int32 = i - period
        
        # --- Check for update Lowest Low:
        if data_arr[i] < lowest_low:
            # If new value is lower than lowest_low, then update lowest_low.
            lowest_low = data_arr[i]
        elif data_arr[i_shifted] == lowest_low:
            # If removed value was lowest_low, then recalculate lowest_low.            
            # Recalculate Lowest Low.
            lowest_low = data_arr[i_shifted + 1]
            for j in range(i_shifted + 2, i + 1):                
                lowest_low = min(lowest_low, data_arr[j])
                
        # --- Check for update Highest High:
        if data_arr[i] > highest_high:
            # If new value is higher than highest_high, then update highest_high.
            highest_high = data_arr[i]
        elif data_arr[i_shifted] == highest_high:
            # If removed value was highest_high, then recalculate highest_high.            
            # Recalculate Highest High.
            highest_high = data_arr[i_shifted + 1]
            for j in range(i_shifted + 2, i + 1):
                highest_high = max(highest_high, data_arr[j])
        
        high_low_range: np.float32 = highest_high - lowest_low
        
        result_arr[i] = 0.0
        if high_low_range > 0:
            result_arr[i] = (highest_high - data_arr[i]) / high_low_range
    
    return result_arr

# -----------------------------------------------------------------------------------
#
#               William %R OC (tsf) - Thread Safe Function
#
# -----------------------------------------------------------------------------------

_signature_tsf = numba.void(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True,  aligned = True),
                    numba.int32,
                    numba.int32,
                    numba.types.Array(numba.float32, 1, 'C', readonly = False, aligned = True), )

_locals_tsf = {
        'lowest_low':       numba.float32,
        'highest_high':     numba.float32,
        'high_low_range':   numba.float32,
        'i_shifted':        numba.int32,
        'data_temp':        numba.float32,
        'data_temp_shft':   numba.float32,
        'data_temp_j':      numba.float32,
        'res_val':          numba.float32,  }

@numba.njit(_signature_tsf,
            cache       =True, 
            fastmath    =True, 
            nogil       =True,
            boundscheck =False,
            locals      =_locals_tsf,   )
def get_william_oc_tsf( 
                        data:       np.ndarray[np.float32], 
                        period:     np.int32,
                        data_size:  np.int32,
                        result_arr: np.ndarray[np.float32],
                            ) -> None:
    '''
    Get William OC of the given array.
    
    Thread Safe Function.
    
    OC:  Only Close Flag. Only Close prices will be used.
    
    ---
    
    Parameters:
    -----------
    data:       (`np.ndarray[np.float32]`) : array of values.
    period:     (`np.int32`)               : period of William OC.
        '(period >= 2)'
    data_size:  (`np.int32`)               : size of data array.
    result_arr: (`np.ndarray[np.float32]`) : array of William OC.
        Results will be rewritten in this array.
    
    '''
        
    if data_size < 0:
        data_size = data.shape[0]
    
    # Calculate initial values.
    lowest_low:   np.float32 = data[0]
    highest_high: np.float32 = data[0]
    
    result_arr[0] = 0.5  # Default value for the first period. # NOTE: It's assigned later as well for the rest of the period.
    
    for i in range(1, period):
        result_arr[i]         = 0.5    # Default value for the first period.        
        data_temp: np.float32 = data[i]        
        lowest_low            = min(lowest_low, data_temp)
        highest_high          = max(highest_high, data_temp)            
        pass

    # Calculate William %R.
    for i in range(period, data_size):
        
        i_shifted:      np.int32    = i - period        
        data_temp:      np.float32  = data[i]
        data_temp_shft: np.float32  = data[i_shifted]
        
        # --- Check for update Lowest Low:
        lowest_low = min(lowest_low, data_temp)
        
        if data_temp_shft == lowest_low:
            # If removed value was lowest_low, then recalculate lowest_low.            
            # Recalculate Lowest Low.
            lowest_low = data[i_shifted + 1]
            for j in range(i_shifted + 2, i + 1):
                data_temp_j: np.float32 = data[j]
                lowest_low = min(lowest_low, data_temp_j)
            pass
        
        # --- Check for update Highest High:
        highest_high = max(highest_high, data_temp)
        
        if data_temp_shft == highest_high:
            # If removed value was highest_high, then recalculate highest_high.            
            # Recalculate Highest High.
            highest_high = data[i_shifted + 1]
            for j in range(i_shifted + 2, i + 1):
                data_temp_j: np.float32 = data[j]
                highest_high = max(highest_high, data_temp_j)
            pass
        
        high_low_range: np.float32 = highest_high - lowest_low
        
        result_arr[i] = 0.0
        if high_low_range > 0:
            result_arr[i] = (highest_high - data_temp) / high_low_range
    
    return

# -----------------------------------------------------------------------------------
#
#         William %R OC (vtsf) - Single Value Calculation,  Thread Safe Function
#
# -----------------------------------------------------------------------------------

_spec_func_vtsf = numba.types.float32(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True),
                    numba.int32,
                    numba.int32, )

_locals_vtsf = {
        'lowest_low':       numba.float32,
        'highest_high':     numba.float32,
        'high_low_range':   numba.float32,                
        'data_temp':        numba.float32,                
        'res_val':          numba.float32,  }

@numba.njit(_spec_func_vtsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,            
            locals      = _locals_vtsf, )
def get_william_oc_vtsf(
                    data:       np.ndarray[np.float32],
                    period:     np.int32,
                    data_indx:  np.int32,
                        ) -> np.float32:
    '''
    Returns single value of William OC.
    
    Parameters:
    -----------
    data:      (`np.ndarray[np.float32]`) : array of values.
    period:    (`np.int32`)               : period of William OC.
    data_indx: (`np.int32`)               : index of value.
    
    Returns:
    --------
    (`np.float32`) : William OC value.    
    '''
    
    star_indx: np.int32   = data_indx - period + 1
    data_temp: np.float32 = data[star_indx]

    # Calculate initial values.
    lowest_low:   np.float32 = data_temp
    highest_high: np.float32 = data_temp
    
    for i in range(star_indx + 1, data_indx + 1):
        
        data_temp: np.float32 = data[i]        
        lowest_low            = min(lowest_low, data_temp)
        highest_high          = max(highest_high, data_temp)
    
    # Calculate William %R.
    high_low_range: np.float32 = highest_high - lowest_low
    
    if high_low_range < 0.0000000001:
        return 0.0
    
    res_val: np.float32 = (highest_high - data_temp) / high_low_range
    
    return res_val

# -----------------------------------------------------------------------------------
#
#               William %R OC (GPU) - NOTE: For use inside CUDA Functions only.
#
# -----------------------------------------------------------------------------------

@cuda.jit()
def get_william_oc_vtsf_cuda(
                data:       np.ndarray[np.float32],
                period:     np.int32,
                data_indx:  np.int32,
                res_indx:   np.int32,
                res_arr:    np.ndarray[np.float32],
                    ) -> None:

    star_indx:    int   = data_indx - period + 1
    data_temp:    float = data[star_indx]
    lowest_low:   float = data_temp
    highest_high: float = data_temp
    
    for i in range(star_indx + 1, data_indx + 1):
        
        data_temp: float = data[i]        
        lowest_low       = min(lowest_low, data_temp)
        highest_high     = max(highest_high, data_temp)
    
    # Calculate William %R.
    high_low_range: float = highest_high - lowest_low

    res_val = 0.0
    
    if high_low_range > 0.0000000001:
        res_val: float = (highest_high - data_temp) / high_low_range
    
    res_arr[res_indx] = res_val
    
    return

# -----------------------------------------------------------------------------------
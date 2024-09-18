import numpy as np
import numba
from numba import cuda


# -----------------------------------------------------------------------------------

_name_:           str = 'MA BOP OC - TI Lib'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# --- CALCULATION DESCRIPTION: -------------------------------------------------------

# MA BOP OC: Moving Average of Balance of Power Oscillator.
#             NOTE: OC, means, that indicator was updated to work only with Close prices, 
#                           instead of vanilla MA BOP indicator.

# MA-BOP (Moving Average Balance of Power) indicator. 
# The Balance of Power (BOP) is a technical indicator 
#   that measures the strength of buyers against sellers in the market. 
# The OC version is Only Close version.
#
# if high_prices[i - 1] != low_prices[i - 1]:
#             bop_values[i] = ((close_prices[i] - open_prices[i]) / (high_prices[i] - low_prices[i])) * volume[i]
#         else:
#            bop_values[i] = 0.0

# _oc version basically calculates part of rising in price candles in the period.

# -----------------------------------------------------------------------------------
#
#               MA BOP OC: Moving Average of Balance of Power (Only Close Prices)
#
# -----------------------------------------------------------------------------------

_spec_func_numba = numba.types.Array(numba.float32, 1, 'C')(
                                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True), 
                                    numba.int32)

_locals_func_numba = {
        'period_up':  numba.int32,
        'period_rev': numba.float32,
        'res_val':    numba.float32, }

@numba.njit(_spec_func_numba,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            inline      = 'always',
            locals      = _locals_func_numba, )
def get_mabop_oc(
                    data_arr: np.ndarray[np.float32],
                    period:   np.int32,
                        ) -> np.ndarray[np.float32]:
    '''
    Get MA BOP OC indicator of the given array.
    
    OC version is Only Close-Price version.
    
    Calculate part of rising in price candles in the period.
    
    ---
    
    Parameters:
    -----------
    data_arr: (`np.ndarray[np.float32]`) : array of values.
    period:   (`np.int32`)               : period of MA BOP OC.
    
    Returns:
    --------
    (`np.ndarray[np.float32]`) : array of MA BOP OC.
    
    '(result[i] - in range [0.0 .. 1.0])'
    '''
    
    result_arr: np.ndarray[np.float32] = np.empty_like(data_arr, dtype = np.float32)
    
    period_up:  np.int32 = 0
    period_rev: np.float32 = 1.0 / np.float32(period)

    result_arr[0] = 0.5   # Default value for the first period.
    
    for i in range(1, period + 1):
        result_arr[i] = 0.5   # Default value for the first period.        
        period_up    += (data_arr[i] > data_arr[i - 1])        
        pass
    
    result_arr[period] = period_rev * period_up 

    for i in range(period + 1, len(data_arr)):
        
        period_up          += (data_arr[i] > data_arr[i - 1])
        period_up          -= (data_arr[i - period] > data_arr[i - period - 1])        
        res_val: np.float32 = period_rev * period_up
        result_arr[i]       = res_val        
        pass

    return result_arr

# -----------------------------------------------------------------------------------
#
#               MA BOP OC: Moving Average of Balance of Power (Only Close Prices)
#                  (tsf) - Thread Safe Function
#
# -----------------------------------------------------------------------------------

_signature_tsf = numba.void(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True,  aligned = True),
                    numba.int32,
                    numba.int32,
                    numba.types.Array(numba.float32, 1, 'C', readonly = False, aligned = True), )

_locals_tsf = {
    'period_up':  numba.int32,
    'period_rev': numba.float32,
    'res_val':    numba.float32,  }

@numba.njit(_signature_tsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            locals      = _locals_tsf,  )
def get_mabop_oc_tsf(
                        data_arr:   np.ndarray[np.float32],
                        period:     np.int32,
                        data_size:  np.int32,
                        result_arr: np.ndarray[np.float32],                  
                                ) -> None:
    '''
    Get MA BOP OC indicator of the given array.
    
    OC version is Only Close-Price version.
    
    Calculate part of rising in price candles in the period.
    
    Thread Safe Function.
    
    ---
    
    Parameters:
    -----------
    data_arr:   (`np.ndarray[np.float32]`) : array of values.
    period:     (`np.int32`)               : period of MA BOP OC.
    data_size:  (`np.int32`)               : size of data array.
    result_arr: (`np.ndarray[np.float32]`) : array of MA BOP OC.
        Results will be rewritten in this array.
    
    '''    
    
    if data_size < 0:
        data_size = data_arr.shape[0]
    
    period_up:  np.int32 = 0
    period_rev: np.float32 = 1.0 / np.float32(period)

    result_arr[0] = 0.5   # Default value for the first period.
    
    for i in range(1, period + 1):
        result_arr[i] = 0.5   # Default value for the first period.        
        period_up    += (data_arr[i] > data_arr[i - 1])
        pass
    
    result_arr[period] = period_rev * period_up 

    for i in range(period + 1, data_size):
        
        period_up          += (data_arr[i] > data_arr[i - 1])
        period_up          -= (data_arr[i - period] > data_arr[i - period - 1])        
        res_val: np.float32 = period_rev * period_up
        result_arr[i]       = res_val
        pass    
    
    return

# -----------------------------------------------------------------------------------
#
#               MA BOP OC: Moving Average of Balance of Power (Only Close Prices)
#                  (vtsf) - Single Value Calculation,  Thread Safe Function
#
# -----------------------------------------------------------------------------------

_spec_func_vtsf = numba.types.float32(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True),
                    numba.int32,
                    numba.int32, )

_locals_vtsf = {
        'period_up': numba.int32,
        'res_val':   numba.float32, }

@numba.njit(_spec_func_vtsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,            
            locals      = _locals_vtsf,)
def get_mabop_oc_vtsf(
                    data_arr:  np.ndarray[np.float32],
                    period:    np.int32,
                    data_indx: np.int32,
                        ) -> np.float32:
    '''
    Calculate MA BOP OC indicator for the single value.
    
    OC version is Only Close-Price version.
    
    Calculate part of rising in price candles in the period.
    
    Thread Safe Function.
    
    ---
    
    Parameters:
    -----------
    data_arr:  (`np.ndarray[np.float32]`) : array of values.
    period:    (`np.int32`)               : period of MA BOP OC.
    data_indx: (`np.int32`)               : index of value.
    
    Returns:
    --------
    (`np.float32`) : MA BOP OC value.    
    '''

    period_step: np.float32 = 1.0 / float(period)
    res_val:     np.float32 = 0.0

    for i in range(data_indx - period + 1, data_indx + 1):
        if data_arr[i] > data_arr[i - 1]:
            res_val += period_step
    
    return res_val

# -----------------------------------------------------------------------------------
#
#               MA BOP OC: Moving Average of Balance of Power (Only Close Prices)
#                 ( GPU Version ) NOTE: For use inside CUDA Functions only.
#
# -----------------------------------------------------------------------------------

@cuda.jit()
def get_mabop_oc_vtsf_cuda(
                    data_arr:   np.ndarray[np.float32],
                    period:     np.int32,
                    data_indx:  np.int32,
                    res_indx:   np.int32,
                    res_arr:    np.ndarray[np.float32],
                        ) -> None:
    '''
    Calculate MA BOP OC indicator for the single value.
    
    OC version is Only Close-Price version.
    
    Calculate part of rising in price candles in the period.
    
    GPU Version, Calculate only single value.
    
    ---
    
    Parameters:
    -----------
    data_arr:  (`np.ndarray[np.float32]`) : array of values.
    period:    (`np.int32`)               : period of MA BOP OC.
    data_indx: (`np.int32`)               : index of value.
    res_indx:  (`np.int32`)               : index of result value.
    res_arr:   (`np.ndarray[np.float32]`) : array of MA BOP OC.
        Result single value will be rewritten in this array.    
    
    '''
    
    period_step: float = 1.0 / float(period)
    res_val:     float = 0.0

    for i in range(data_indx - period + 1, data_indx + 1):
        if data_arr[i] > data_arr[i - 1]:
            res_val += period_step
        
    res_arr[res_indx] = res_val
    
    return

# -----------------------------------------------------------------------------------
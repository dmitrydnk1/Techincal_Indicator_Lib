import numpy as np
import numba
from numba import cuda


# -----------------------------------------------------------------------------------

_name_:           str = 'Percent Change - TI Lib'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# --- CALCULATION DESCRIPTION: -------------------------------------------------------

# Period_Change or Percent_Change = [(Close - PrevClose) / PrevClose]
#                                 = [(Close / PrevClose) - 1.0]

# -----------------------------------------------------------------------------------
#
#               Percent Change: pcnt_ch
#
# -----------------------------------------------------------------------------------

_spec_func_numba = numba.types.Array(numba.float32, 1, 'C', aligned = True)(
                                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True), 
                                    numba.int32,  )

@numba.njit(_spec_func_numba,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            inline      = 'always', )
def get_pcnt_ch(
                data_arr: np.ndarray[np.float32],
                period:   np.int32,
                    ) -> np.ndarray[np.float32]:
    '''
    Get Percent Change. of given data array.
    
    ---
    
    Parameters:
    -----------
    data_arr: (`np.ndarray[np.float32]`) : Input data array.
    period:   (`np.int32`)               : Period.
    
    Returns:
    --------
    (`np.ndarray[np.float32]`) : Percent Change array.    
    '''

    result_arr          = np.empty_like(data_arr, dtype = np.float32)    
    result_arr[:period] = 0.0       # Default value for the first period.

    for i in range(period, len(data_arr)):
        result_arr[i] = (data_arr[i] / data_arr[i - period]) - 1.0

    return result_arr

# -----------------------------------------------------------------------------------
#
#             Percent Change (tsf) - Thread Safe Function
#
# -----------------------------------------------------------------------------------

_signature_tsf = numba.void(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True,  aligned = True),
                    numba.int32,
                    numba.int32,
                    numba.types.Array(numba.float32, 1, 'C', readonly = False, aligned = True), )

_locals_tsf = {
    'res_val': numba.float32, }

@numba.njit(_signature_tsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            locals      = _locals_tsf, )
def get_pcnt_ch_tsf(    
                    data:       np.ndarray[np.float32], 
                    period:     np.int32,
                    data_size:  np.int32,
                    result_arr: np.ndarray[np.float32],
                        ) -> None:
    '''
    Get Percent Change. of given data array.
    
    ---
    
    Parameters:
    -----------
    data:       (`np.ndarray[np.float32]`) : Input data array.
    period:     (`np.int32`)               : Period.
    data_size:  (`np.int32`)               : Data size.
    result_arr: (`np.ndarray[np.float32]`) : Result array.
        Result data will be rewritten in the same array.
    
    '''
    if data_size < 0:
        data_size = data.shape[0]
    
    for i in range(period):
        result_arr[i] = 0.0     # Default value for the first period.

    for i in range(period, data_size):
        res_val:     np.float32 = (data[i] / data[i - period]) - 1.0        
        result_arr[i]           = res_val

    return

# -----------------------------------------------------------------------------------
#
#             Percent Change (vtsf) - Single Value, Thread Safe Function
#
# -----------------------------------------------------------------------------------

_spec_func_vtsf = numba.types.float32(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True),
                    numba.int32,
                    numba.int32, )

_locals_vtsf = {'res_val': numba.float32,}

@numba.njit(_spec_func_vtsf,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,            
            locals      = _locals_vtsf,  )
def get_pcnt_ch_vtsf(
                    data_arr:  np.ndarray[np.float32],
                    period:    np.int32,
                    data_indx: np.int32,
                        ) -> np.float32:
    '''
    Get Percent Change Single Value.
    
    ---
    
    Parameters:
    -----------
    data_arr:  (`np.ndarray[np.float32]`) : Input data array.
    period:    (`np.int32`)               : Period.
    data_indx: (`np.int32`)               : Data index.
    
    Returns:
    --------
    (`np.float32`) : Percent Change Single Value.
    '''

    res_val: np.float32 = (data_arr[data_indx] / data_arr[data_indx - period]) - 1.0
    
    return res_val

# -----------------------------------------------------------------------------------
#
#             Percent Change (GPU) - NOTE: For use inside CUDA Functions only.
#
# -----------------------------------------------------------------------------------


@cuda.jit()
def get_pcnt_ch_vtsf_cuda(
                    data:       np.ndarray[np.float32],
                    period:     np.int32,
                    data_indx:  np.int32,
                    res_indx:   np.int32,
                    res_arr:    np.ndarray[np.float32],
                        ) -> None:
    
    res_val: np.float32 = (data[data_indx] / data[data_indx - period]) - 1.0
    res_arr[res_indx]   = res_val
    
    return

# -----------------------------------------------------------------------------------
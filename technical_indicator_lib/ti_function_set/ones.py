import numpy as np
import numba
from numba import cuda


# -----------------------------------------------------------------------------------

_name_:           str = 'ONES'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# --- CALCULATION DESCRIPTION: -------------------------------------------------------

# Just returns an array of ones, i.e. [1.0, 1.0, 1.0, ...]

# -----------------------------------------------------------------------------------
#
#               Just ONES
#
# -----------------------------------------------------------------------------------

_spec_func_numba = numba.types.Array(numba.float32, 1, 'C', readonly = False, aligned = True)(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True,  aligned = True), 
                    numba.int32,  )

@numba.njit(_spec_func_numba,
            cache       = True, 
            nogil       = True,
            boundscheck = False, )
def get_ones(
                data:   np.ndarray[np.float32],             
                period: np.int32, 
                    ) -> np.ndarray[np.float32]:
    '''
    Get array of ONES.
    
    Parameters:
    -----------
    data:   (`np.ndarray[np.float32]`) : Input data array.
        Not Used, just to return array of ones, 
            with same length as input data.
    period: (`np.int32`)               : Period, Not used.
    
    ---
    
    Returns:
    --------
    (`np.ndarray[np.float32]`) : Array of ones.
    '''    
    return np.ones(len(data), dtype = np.float32)

# -----------------------------------------------------------------------------------
#
#              ONES (tsf) - Thread Safe Function
#
# -----------------------------------------------------------------------------------

_signature_tsf = numba.void(
                    numba.types.Array(numba.float32, 1, 'C', readonly = True,  aligned = True),
                    numba.int32,
                    numba.int32,
                    numba.types.Array(numba.float32, 1, 'C', readonly = False, aligned = True), )

@numba.njit(_signature_tsf,
            cache       = True, 
            nogil       = True,
            boundscheck = False,  )
def get_ones_tsf(
                    data:       np.ndarray[np.float32],             
                    period:     np.int32, 
                    data_size:  np.int32,
                    result_arr: np.ndarray[np.float32],                  
                        ) -> None:
    
    if data_size < 0:
        data_size = data.shape[0]
    
    for i in range(data_size):
        result_arr[i] = 1.0
    
    return

# -----------------------------------------------------------------------------------
#
#               ONES (vtsf) - Single Value, Thread Safe Function
#
# -----------------------------------------------------------------------------------

_spec_func_vtsf = numba.types.float32(                    
                    numba.types.Array(numba.float32, 1, 'C', aligned = True),
                    numba.int32,
                    numba.int32, )

@numba.njit(_spec_func_vtsf,
            cache       = True, 
            nogil       = True,
            boundscheck = False, )
def get_ones_vtsf(
                    data_arr:   np.ndarray[np.float32],
                    period:     np.int32,
                    data_indx:  np.int32,
                        ) -> np.float32:
    return 1.0

# -----------------------------------------------------------------------------------
import numpy as np
import numba

# -----------------------------------------------------------------------------------

_name_:           str = 'Shift - TI Lib'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# -----------------------------------------------------------------------------------
#
#               SHIFT
#
# -----------------------------------------------------------------------------------

_spec_func_numba = numba.types.Array(numba.float32, 1, 'C')(
                numba.types.Array(numba.float32, 1, 'C', readonly = True, aligned = True), 
                numba.int32)

@numba.njit(_spec_func_numba,
            cache       = True, 
            fastmath    = True, 
            nogil       = True,
            boundscheck = False,
            inline      = 'always', )
def get_shift(
                data:  np.ndarray[np.float32],             
                shift: np.int32, 
                    ) -> np.ndarray[np.float32]:
    '''
    Get shifted array.
        
    ---
    
    Parameters:
    -----------
    data:  (`np.ndarray[np.float32]`) : array of values.
    shift: (`np.int32`)               : shift period.
    
    Returns:
    --------
    (`np.ndarray[np.float32]`) : shifted array.
    
    ---
    
    Example:
    --------
    >>> arr: np.ndarray[np.float32] = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype = np.float32)
    >>> get_shift(arr, 2)  
        # Shifts array to right ( >> ).
        # Return: np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype = np.float32)
    >>> get_shift(arr, -2) 
        # Shifts array to left ( << ).
        # Return: np.array([3.0, 4.0, 5.0, 0.0, 0.0], dtype = np.float32)
    '''
    
    if shift > 0:
        res_arr: np.ndarray[np.float32] = np.roll(data, -shift)
        res_arr[-shift:] = 0.0  # fill unknown values with zeros.
        return res_arr
    
    elif shift < 0:
        res_arr: np.ndarray[np.float32] = np.roll(data, -shift)
        res_arr[:shift] = 0.0   # fill unknown values with zeros.
        return res_arr
    
    return res_arr

# -----------------------------------------------------------------------------------
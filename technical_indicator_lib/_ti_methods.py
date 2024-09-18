import numpy as np


# -----------------------------------------------------------------------------------

_name_:           str = 'Private Functions - TI Lib'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# -----------------------------------------------------------------------------------
#
#               PRIVATE FUNCTIONS FOR TI_LIB
#
# -----------------------------------------------------------------------------------

def extend_data_array(  data_arr:         np.ndarray[float], 
                        desired_data_len: int, 
                            ) -> np.ndarray[float]:
    '''
    Extend data array to desired length.
    
    Parameters:
    -----------
    data_arr:         (`np.ndarray[float]`) : Input data array.
    desired_data_len: (`int`)               : Desired data length.
    
    Returns:
    --------
    (`np.ndarray[float]`) : Extended data array.    
    '''
    if len(data_arr) < desired_data_len:        
        data_arr = np.append(data_arr, np.zeros(desired_data_len - len(data_arr)))
    
    return data_arr

# -----------------------------------------------------------------------------------
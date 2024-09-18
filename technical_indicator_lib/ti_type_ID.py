from enum import IntEnum
import numpy as np
import numba

# -----------------------------------------------------------------------------------

_name_:           str = 'Types - TI Lib'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# -----------------------------------------------------------------------------------
#
#               ENUM: TI_ID
#
# -----------------------------------------------------------------------------------

class TI_ID(IntEnum):
    
    NONE        = -1
    PCNT_CH     = 0
    ONES        = 1
    RSI         = 2
    BB          = 3
    AROON       = 4
    WILLIAM_OC  = 5
    MABOP_OC    = 6
    LR_SLOPE    = 7
    LR_EXP_DEV  = 8
    
    pass

# -----------------------------------------------------------------------------------
#
#               Get TI Type Str by ID
#
# -----------------------------------------------------------------------------------

@numba.njit(cache = True)
def get_ti_type_str(ti_ID: np.int64) -> str:
    '''
    Returns TI Type Name ('str') by ID ('int').
    '''
    
    if ti_ID == -1:
        return 'NONE'
    if ti_ID == 0:
        return 'PCNT_CH'
    if ti_ID == 1:
        return 'ONES'
    if ti_ID == 2:
        return 'RSI'
    if ti_ID == 3:
        return 'BB'
    if ti_ID == 4:
        return 'AROON'
    if ti_ID == 5:
        return 'WILLIAM_OC'
    if ti_ID == 6:
        return 'MABOP_OC'
    if ti_ID == 7:
        return 'LR_SLOPE'
    if ti_ID == 8:
        return 'LR_EXP_DEV'
    
    return 'UNKNOWN'

# -----------------------------------------------------------------------------------
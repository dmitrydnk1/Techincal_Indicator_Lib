'''
'''


# -----------------------------------------------------------------------------------

_name_:           str = 'Set of TI Functions - TI Lib'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# -----------------------------------------------------------------------------------
#
#                 Technical Indicator Function Set
#
# -----------------------------------------------------------------------------------

# --- SUPPORTING FUNCTIONS: ---------------------------------------------------------

from .shift import get_shift

# --- TECHNICAL INDICATORS FUNCTIONS: -----------------------------------------------

from .aroon import (
    get_aroon, 
    get_aroon_tsf,
    get_aroon_vtsf,
    get_aroon_vtsf_cuda, )

from .bb import (
    get_bb,
    get_bb_tsf,
    get_bb_vtsf,
    get_bb_vtsf_cuda, )

from .lr_exp_dev import (
    get_lr_exp_dev,
    get_lr_exp_dev_mini,
    get_lr_exp_dev_mini_tsf,
    get_lr_exp_dev_mini_vtsf,
    get_lr_exp_dev_mini_vtsf_cuda, )
    
from .lr_slope import (
    get_lr_slope,
    get_lr_slope_tsf,
    get_lr_slope_vtsf,
    get_lr_slope_vtsf_cuda, )

from .mabop_oc import (
    get_mabop_oc,
    get_mabop_oc_tsf,
    get_mabop_oc_vtsf,
    get_mabop_oc_vtsf_cuda, )

from .ones import (
    get_ones,
    get_ones_tsf,
    get_ones_vtsf, )

from .pcnt_ch import (
    get_pcnt_ch,
    get_pcnt_ch_tsf,
    get_pcnt_ch_vtsf,
    get_pcnt_ch_vtsf_cuda, )

from .rsi import (
    get_rsi,
    get_rsi_tsf,
    get_rsi_vtsf,
    get_rsi_vtsf_cuda, )

from .william_oc import (
    get_william_oc,
    get_william_oc_tsf,
    get_william_oc_vtsf,
    get_william_oc_vtsf_cuda, )

# -----------------------------------------------------------------------------------
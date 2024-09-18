'''
TI - Technical Indicators Library

---

Contains a set of functions to calculate various technical indicators.

---

CONTAINS 4 MAIN TYPES OF FUNCTIONS:
-----------------------------------
- Functionf for calculating TI ( Technical Indicators ). With optimized performance.
- Thread Safe Functions ( TSF ) for calculating TI. Functions starts with 'tsf_' prefix. Updates input data array, and safe to use in multi-threaded environments. 
- Single Value Calulation ( VTSF ). Functions starts with 'vtsf_' prefix. Returns just a single selected value of the indicator. Suitable for streaming values, or to get just the last value.
- Functions for use inside CUDA codespace only. Functions starts with 'cuda_v_' Returns just a single selected value of the indicator.

'''

# -----------------------------------------------------------------------------------

_name_:           str = 'TI Lib: Technical Indicators Library'
__version__:      str = '0.0.1'
__version_date__: str = '2024-09-16'
VERSION:          str = f'{_name_:<20} VERSION: {__version__} @ {__version_date__}' 

# --- VERSION HISTORY: --------------------------------------------------------------

# v0.0.1 @ 2024-09-16 : Initial Release.
#

# --- TECHNICAL INDICATORS LIST: ----------------------------------------------------

# NOTE: Performance optimized functions.

from .ti_function_set import (
    get_shift,
    get_aroon,
    get_bb,
    get_lr_exp_dev,
    get_lr_exp_dev_mini,
    get_lr_slope,
    get_mabop_oc,
    get_ones,
    get_pcnt_ch,
    get_rsi,
    get_william_oc, )


# --- TECHNICAL INDICATORS - TSF ( Thread Safe Functions ): -------------------------

# NOTE: Safe to use in multi-threaded environments.

from .ti_function_set import (
    get_aroon_tsf           as tsf_aroon,
    get_bb_tsf              as tsf_bb,
    get_lr_exp_dev_mini_tsf as tsf_lr_exp_dev_mini,
    get_lr_slope_tsf        as tsf_lr_slope,
    get_mabop_oc_tsf        as tsf_mabop_oc,
    get_ones_tsf            as tsf_ones,
    get_pcnt_ch_tsf         as tsf_pcnt_ch,
    get_rsi_tsf             as tsf_rsi,
    get_william_oc_tsf      as tsf_william_oc, )

# --- TECHNICAL INDICATORS - V TSF ( Single Value , Thread Safe Functions ): --------

# NOTE: Safe to use in multi-threaded environments.
#       Returns just a single selected value of the indicator.
#           e.g. suitable for streaming values, or to get just the last value.

from .ti_function_set import (
    get_aroon_vtsf           as vtsf_aroon,
    get_bb_vtsf              as vtsf_bb,
    get_lr_exp_dev_mini_vtsf as vtsf_lr_exp_dev_mini,
    get_lr_slope_vtsf        as vtsf_lr_slope,
    get_mabop_oc_vtsf        as vtsf_mabop_oc,
    get_ones_vtsf            as vtsf_ones,
    get_pcnt_ch_vtsf         as vtsf_pcnt_ch,
    get_rsi_vtsf             as vtsf_rsi,
    get_william_oc_vtsf      as vtsf_william_oc, )


# --- Techinical Indicators - V GPU -------------------------------------------------

# NOTE: For use inside CUDA codespace only.
#       Returns just a single selected value of the indicator.

from .ti_function_set import (
    get_aroon_vtsf_cuda           as cuda_v_aroon,
    get_bb_vtsf_cuda              as cuda_v_bb,
    get_lr_exp_dev_mini_vtsf_cuda as cuda_v_lr_exp_dev_mini,
    get_lr_slope_vtsf_cuda        as cuda_v_lr_slope,
    get_mabop_oc_vtsf_cuda        as cuda_v_mabop_oc,    
    get_pcnt_ch_vtsf_cuda         as cuda_v_pcnt_ch,
    get_rsi_vtsf_cuda             as cuda_v_rsi,
    get_william_oc_vtsf_cuda      as cuda_v_william_oc,    
)

# --- TI - TYPES and TI-ID ENUM: ----------------------------------------------------

from .ti_type_ID import (
    TI_ID,
    get_ti_type_str, )

# -----------------------------------------------------------------------------------
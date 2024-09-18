# Technical Indicators Library

A set of functions to calculate various technical indicators.

## Contains 4 types of functions

1. Functionf for calculating TI ( Technical Indicators ). 
    With optimized performance.
2. Thread Safe Functions ( TSF ) for calculating TI. 
    Functions starts with 'tsf_' prefix. 
    Updates input data array, and safe to use in multi-threaded environments. 
3. Single Value Calulation ( VTSF ). 
    Functions starts with 'vtsf_' prefix. 
    Returns just a single selected value of the indicator. 
    Suitable for streaming calculation, or to get just the last value.
4. Functions for use inside CUDA codespace only. 
    Functions starts with 'cuda_v_' Returns just a single selected value of the indicator.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/dmitrydnk1/Techincal_Indicator_Lib.git    
    ```
2. Navigate to the project directory:
    ```bash
    cd repository
    ```
3. Install with required dependencies:
    ```bash
    pip install .
    ```

## Usage Example

```python
import numpy as np
import ti_lib

# Sample Inputs:
period:   int = 2
data_arr: np.ndarray[np.float32] = np.array([1, 2, 3, 4, 5, 2, 3, 5], dtype = np.float32)

# Example for RSI:
rsi_arr: np.ndarray[np.float32] = ti_lib.get_rsi(data_arr, period)
```
from typing import Tuple, Annotated
import numpy as np
from numpy.typing import NDArray


VectorInt8 = Annotated[NDArray[np.int8], Tuple[int]]
"""1D array of 8-bit signed integers."""

VectorFloat64 = Annotated[NDArray[np.float64], Tuple[int]]
"""1D array of 64-bit floating-point numbers (double precision)."""

VectorStr = Annotated[NDArray[np.str_], Tuple[int]]
"""1D array of NumPy Unicode strings."""

VectorBool = Annotated[NDArray[np.bool_], Tuple[int]]
"""1D array of NumPy booleans."""


MatrixInt8 = Annotated[NDArray[np.int8], Tuple[int, int]]
"""2D array (matrix) of 8-bit signed integers."""

MatrixFloat64 = Annotated[NDArray[np.float64], Tuple[int, int]]
"""2D array (matrix) of 64-bit floating-point numbers (double precision)."""


TensorFloat64 = Annotated[NDArray[np.float64], Tuple[int, int, int]]
"""3D array (tensor) of 64-bit floating-point numbers (double precision)."""

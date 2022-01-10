'''
'''
import ctypes
from typing import Final

# --- GLOBALS

SCREEN_WIDTH: Final[int] = 500
SCREEN_HEIGHT: Final[int] = 500


# --- DATA SIZES


FLOAT_SIZE: Final[int] = ctypes.sizeof(ctypes.c_float)
UINT_SIZE: Final[int] = ctypes.sizeof(ctypes.c_uint)


# --- C HELPERS FUNCTION


def c_array(arr: list[float]):
    """Create a ctypes.c_float_Array for OpenGL from a python list of floats

    Parameters
    ---
    arr : list[float]
        A python list of floats

    Returns
    ---
    ctypes.c_float_Array
        A ctypes.c_float array
    """
    return (ctypes.c_float * len(arr))(*arr)


def c_cast(offset: int):
    """Create a c type void* cast using ctypes.c_void_p for OpenGL

    Parameters
    ----------
    offset : int
        void* offset

    Returns
    -------
    ctypes.c_void_p
        void pointer cast
    """
    return ctypes.c_void_p(offset)
"""Helper Functions
"""
import ctypes
from typing import Final

# --- GLOBALS

SCREEN_WIDTH: Final[int] = 640
SCREEN_HEIGHT: Final[int] = 480


# --- DATA SIZES


FLOAT_SIZE: Final[int] = ctypes.sizeof(ctypes.c_float)
UINT_SIZE: Final[int] = ctypes.sizeof(ctypes.c_uint)


# --- C HELPERS FUNCTION


def c_arrayF(arr: list[float]):
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

def c_arrayU(arr: list[int]):
    """Create a ctypes.c_uint_Array for OpenGL from a python list of ints

    Parameters
    ---
    arr : list[int]
        A python list of floats

    Returns
    ---
    ctypes.c__Array
        A ctypes.c_uint array
    """
    return(ctypes.c_uint * len(arr))(*arr)


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
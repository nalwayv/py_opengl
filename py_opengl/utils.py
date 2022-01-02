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
    '''Convert list to ctype array'''
    return (ctypes.c_float * len(arr))(*arr)


def c_cast(offset: int):
    '''Cast to ctype void pointer (void*)(offset)'''
    return ctypes.c_void_p(offset)



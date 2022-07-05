"""Utils
"""
import ctypes
from typing import Final

# --- GLOBALS


SCREEN_WIDTH: Final[int]= 640
SCREEN_HEIGHT: Final[int]= 480


# --- DATA SIZES


SIZEOF_FLOAT: Final[int]= ctypes.sizeof(ctypes.c_float)
SIZEOF_UINT: Final[int]= ctypes.sizeof(ctypes.c_uint)


# --- C TYPES


NULL: Final[ctypes.c_void_p]= ctypes.c_void_p(0)


# --- HELPER FUNCTIONS


def float_to_bits(val: float) -> int:
    """Convert float value to bits
    """
    val_a= ctypes.c_float(val)
    address: int= ctypes.addressof(val_a)
    val_b= ctypes.c_int.from_address(address)

    return val_b.value


def bits_to_float(hex_val: int) -> float:
    """Convert hex bits to float
    
    Example
    ---
    float_to_bits(42) # 1109917696\n
    bits_to_float(1109917696) # 42.0
    """
    val_a= ctypes.c_int(hex_val)
    address: int= ctypes.addressof(val_a)
    val_b= ctypes.c_float.from_address(address)

    return val_b.value


def hash_code(arr: list[float]) -> int:
    """
    """
    result: int= 1
    tmp: int= 0
    for num in arr:
        tmp= float_to_bits(num)
        result= 31 * result + (tmp ^ (tmp >> 32))
    return result


# --- HELPER CLASSES


class BitMask:

    __slots__=('_mask',)

    def __init__(self) -> None:
        self._mask: int= 0
    
    def clear(self) -> None:
        if self._mask == 0:
            return
        self._mask ^= self._mask

    def has(self, v: int) -> bool:
        return (self._mask & v) != 0

    def add(self, v: int) -> None:
        if self.has(v):
            return
        self._mask |= v

    def remove(self, v: int) -> None:
        if not self.has(v):
            return
        self._mask &= ~(v)
    
    def toggle(self, v: int) -> None:
        self._mask ^= v


# --- C HELPERS FUNCTIONS


def c_arrayF(arr: list[float]) -> ctypes.Array:
    """Create a 'c' array for OpenGL
    """
    return (ctypes.c_float * len(arr))(*arr)


def c_arrayU(arr: list[int]) -> ctypes.Array:
    """Create a 'c' array for OpenGL
    """
    return(ctypes.c_uint * len(arr))(*arr)


def c_cast(offset: int) -> ctypes.c_void_p:
    """Create a 'c' void* for OpenGL

    c_cast(0) can also be used as a NULL
    """
    return ctypes.c_void_p(offset)
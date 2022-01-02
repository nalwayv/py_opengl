'''Mouse input
'''
from dataclasses import dataclass, field
from enum import Enum

class MouseState(Enum):
    PRESSED = 0
    RELEASED = 1
    HELD = 2
    DEFAULT = 3


@dataclass(eq=False, repr=False, slots=True)
class Mouse:
    states: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.states = [0xFF]*3


def _mouse_set_current(mouse: Mouse, key: int, value: int) -> None:
    ''' '''
    mouse.states[key] = (mouse.states[key] & 0xFFFFFF00) | value


def _mouse_set_previous(mouse: Mouse, key: int, value: int) -> None:
    ''' '''
    mouse.states[key] = (mouse.states[key] & 0xFFFF00FF) | (value << 8)


def _mouse_get_current(mouse: Mouse, key: int) -> int:
    ''' '''
    return 0xFF & mouse.states[key]


def _mouse_get_previous(mouse: Mouse, key: int) -> int:
    ''' '''
    return 0xFF & (mouse.states[key] >> 8)


def mouse_state(mouse: Mouse, glfw_mouse_state: tuple[int, int]) -> MouseState:
    '''Mouse button pressed

    Parameters
    ---
    mouse: Mouse
    glfw_mouse_state: tuple[int, int]
        glfw mouse button number and its state

    Returns
    ---
    MouseState: Enum
    '''
    key, state = glfw_mouse_state
    if key > 3:
        return MouseState.DEFAULT

    tmp = _mouse_get_current(mouse, key)
    _mouse_set_previous(mouse, key, tmp)
    _mouse_set_current(mouse, key, state)

    if _mouse_get_previous(mouse, key) == 0:
        if _mouse_get_current(mouse, key) == 0:
            return MouseState.DEFAULT
        else:
            # pressed
            return MouseState.PRESSED
    else:
        if _mouse_get_current(mouse, key) == 0:
            # released
            return MouseState.RELEASED
        else:
            # held
            return MouseState.HELD
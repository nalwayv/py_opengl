'''Keyboard input
'''
from dataclasses import dataclass, field
from enum import Enum


class KeyState(Enum):
    PRESSED = 0
    RELEASED = 1
    HELD = 2
    DEFAULT = 3


@dataclass(eq=False, repr=False, slots=True)
class Keyboard:
    states: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.states = [0xFF]*301


def _keyboard_set_current_at(kb: Keyboard, key: int, value: int) -> None:
    '''
    Private helper function for keyboard class.
    Helps set current keystate value
    '''
    kb.states[key] = (kb.states[key] & 0xFFFFFF00) | value


def _keyboard_set_previous_at(kb: Keyboard, key: int, value: int) -> None:
    '''
    Private helper function for keyboard class.
    Helps set previus keystate value
    '''
    kb.states[key] = (kb.states[key] & 0xFFFF00FF) | (value << 8)


def _keyboard_get_current_at(kb: Keyboard, key: int) -> int:
    '''
    Private helper function for keyboard class.
    Gets current keystate value for given key
    '''
    return 0xFF & kb.states[key]


def _keyboard_get_previous_at(kb: Keyboard, key: int) -> int:
    '''
    Private helper function for keyboard class.
    Gets previous keystate value for given key
    '''
    return 0xFF & (kb.states[key] >> 8)


def key_state(kb: Keyboard, glfw_key_state: tuple[int, int]) -> KeyState:
    '''Keyboard button pressed

    Parameters
    ---
    kb: Keyboard
    glfw_key_state: tuple[int, int]
        glfw keyboard key number and its state

    Returns
    ---
    KeyState: Enum
    '''
    key, state = glfw_key_state
    if key > 301:
        return KeyState.DEFAULT

    tmp = _keyboard_get_current_at(kb, key)
    _keyboard_set_previous_at(kb, key, tmp)
    _keyboard_set_current_at(kb, key, state)

    if _keyboard_get_previous_at(kb, key) == 0:
        if _keyboard_get_current_at(kb, key) == 0:
            return KeyState.DEFAULT
        else:
            # pressed
            return KeyState.PRESSED
    else:
        if _keyboard_get_current_at(kb, key) == 0:
            # released
            return KeyState.RELEASED
        else:
            # held
            return KeyState.HELD
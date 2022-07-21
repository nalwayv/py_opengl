"""Keyboard
"""
from enum import Enum, auto
from typing import Final

from py_opengl import window


# ---


KEYBOARD_SIZE: Final[int] = 1024


# ---


class KeyState(Enum):
    PRESSED= auto()
    RELEASED= auto()
    HELD= auto()
    DEFAULT= auto()


# ---


class Keyboard:

    __slots__= ('states', 'win')

    def __init__(self, win: window.GlWindow) -> None:
        self.states: list[int]= [0xFF] * KEYBOARD_SIZE
        self.win: window.GlWindow|None= win

    def _set_current_state_at(self, key: int, value: int) -> None:
        self.states[key]= (self.states[key] & 0xFFFFFF00) | value

    def _set_previous_state_at(self, key: int, value: int) -> None:
        self.states[key]= (self.states[key] & 0xFFFF00FF) | (value << 8)

    def _get_current_state_at(self, key: int) -> int:
        return 0xFF & self.states[key]

    def _get_previous_state_at(self, key: int) -> int:
        return 0xFF & (self.states[key] >> 8)

    def _get_state(self, glfw_key_state: int) -> KeyState:
        """Get keyboard keystate
        """
        key= glfw_key_state
        current= self.win.get_key_state(glfw_key_state)
        result= KeyState.DEFAULT

        if key < KEYBOARD_SIZE:
            prev= self._get_current_state_at(key)
            self._set_previous_state_at(key, prev)
            self._set_current_state_at(key, current)

            if self._get_previous_state_at(key) == 0:
                if self._get_current_state_at(key) == 0:
                    result= KeyState.DEFAULT
                else:
                    # pressed
                    result= KeyState.PRESSED
            else:
                if self._get_current_state_at(key) == 0:
                    # released
                    result= KeyState.RELEASED
                else:
                    # held
                    result= KeyState.HELD

        return result

    def is_key_held(self, key: int) -> bool:
        """Return true if key is being held down
        """
        return self._get_state(key) == KeyState.HELD

    def is_key_pressed(self, key: int) -> bool:
        """Return true if key if just being pressed
        """
        return self._get_state(key) == KeyState.PRESSED

    def is_key_released(self, key: int) -> bool:
        """Return true if key was released
        """
        return self._get_state(key) == KeyState.RELEASED

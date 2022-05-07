"""Keyboard
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final


# ---


KEYBOARD_SIZE: Final[int] = 1024


# ---


class KeyState(Enum):
    PRESSED= auto()
    RELEASED= auto()
    HELD= auto()
    DEFAULT= auto()


# ---


@dataclass(eq= False, repr= False, slots= True)
class Keyboard:
    _states: list[int]= field(default_factory=list)

    def __post_init__(self):
        self._states= [0xFF] * KEYBOARD_SIZE

    def _set_current_state_at(self, key: int, value: int) -> None:
        self._states[key]= (self._states[key] & 0xFFFFFF00) | value

    def _set_previous_state_at(self, key: int, value: int) -> None:
        self._states[key]= (self._states[key] & 0xFFFF00FF) | (value << 8)

    def _get_current_state_at(self, key: int) -> int:
        return 0xFF & self._states[key]

    def _get_previous_state_at(self, key: int) -> int:
        return 0xFF & (self._states[key] >> 8)

    def get_state(self, glfw_key_state: tuple[int, int]) -> KeyState:
        """Get keyboard keystate

        use GlWindow funcfion 'get_key_state' to get glfw_key_state
        """
        key, current= glfw_key_state
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

    def is_key_held(self, key_state: tuple[int, int]) -> bool:
        """Helper function for key held down state
        """
        return self.get_state(key_state) is KeyState.HELD

    def is_key_pressed(self, key_state: tuple[int, int]) -> bool:
        """Helper function for key pressed state
        """
        return self.get_state(key_state) is KeyState.PRESSED

    def is_key_released(self, key_state: tuple[int, int]) -> bool:
        """Helper function for key released state
        """
        return self.get_state(key_state) is KeyState.RELEASED

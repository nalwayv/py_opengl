"""
Keyboard
---
With help from glfw get the current state of a 
key being pressed, held or released
"""
from dataclasses import dataclass, field
from enum import Enum, auto


class KeyState(Enum):
    PRESSED = auto()
    RELEASED = auto()
    HELD = auto()
    DEFAULT = auto()


@dataclass(eq=False, repr=False, slots=True)
class Keyboard:
    states: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.states = [0xFF]*301

    def _set_current_state_at(self, key: int, value: int) -> None:
        """Set glfw key number stored current state

        Parameters
        ---
        key : int
            glfw key number
        value : int
            glfw key state
        """
        self.states[key] = (self.states[key] & 0xFFFFFF00) | value

    def _set_previous_state_at(self, key: int, value: int) -> None:
        """Set glfw key number stored previous state

        Parameters
        ---
        key : int
            glfw key number
        value : int
            glfw key state
        """
        self.states[key] = (self.states[key] & 0xFFFF00FF) | (value << 8)

    def _get_current_state_at(self, key: int) -> int:
        """Get glfw key number stored current state

        Parameters
        ---
        key : int
            glfw key number

        Returns
        ---
        int
            currently stored current state for that key
        """
        return 0xFF & self.states[key]

    def _get_previous_state_at(self, key: int) -> int:
        """Get glfw key button number stored previous state

        Parameters
        ---
        key : int
            glfw key number

        Returns
        ---
        int
            currently stored previous state for that key
        """
        return 0xFF & (self.states[key] >> 8)

    def get_state(self, glfw_key_state: tuple[int, int]) -> KeyState:
        """Get keyboard keystate

        Parameters
        ---
        glfw_key_state : tuple[int, int]
            glfw keyboard key number and its glfw state

        Example
        ---
        ```python

        win = GlWindow()
        kb = Keyboard()

        state = win.get_key_state(glfw.KEY_A)
        
        if kb.get_state(state) == KeyState.PRESSED:
            print('pressed')
        ```

        Returns
        ---
        KeyState
            is its current state held, pressed or released
        """
        key, state = glfw_key_state
        if key > 301:
            return KeyState.DEFAULT

        tmp = self._get_current_state_at(key)
        self._set_previous_state_at(key, tmp)
        self._set_current_state_at(key, state)

        if self._get_previous_state_at(key) == 0:
            if self._get_current_state_at(key) == 0:
                return KeyState.DEFAULT
            else:
                # pressed
                return KeyState.PRESSED
        else:
            if self._get_current_state_at(key) == 0:
                # released
                return KeyState.RELEASED
            else:
                # held
                return KeyState.HELD

    def is_key_held(self, key_state: tuple[int, int]) -> bool:
        """Helper function for key held down state

        Parameters
        ---
        key_state : tuple[int, int]

        Returns
        ---
        bool
        """
        return self.get_state(key_state) is KeyState.HELD

    def is_key_pressed(self, key_state: tuple[int, int]) -> bool:
        """Helper function for key pressed state

        Parameters
        ---
        key_state : tuple[int, int]

        Returns
        ---
        bool
        """
        return self.get_state(key_state) is KeyState.PRESSED

    def is_key_released(self, key_state: tuple[int, int]) -> bool:
        """Helper function for key released state

        Parameters
        ---
        key_state : tuple[int, int]

        Returns
        ---
        bool
        """
        return self.get_state(key_state) is KeyState.RELEASED
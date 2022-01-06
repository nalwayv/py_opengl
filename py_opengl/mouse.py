"""
Mouse
---
With help from glfw get the current state of a 
mouse button being pressed, held or released
"""
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


    def _set_current_state_at(self, key: int, value: int) -> None:
        """Set glfw mouse button number stored current state

        Parameters
        ---
        key : int
            glfw mouse button number
        value : int
            glfw mouse button state
        """
        self.states[key] = (self.states[key] & 0xFFFFFF00) | value


    def _set_previous_state_at(self, key: int, value: int) -> None:
        """Set glfw mouse button number stored previous state

        Parameters
        ---
        key : int
            glfw mouse button number
        value : int
            glfw mouse button state
        """
        self.states[key] = (self.states[key] & 0xFFFF00FF) | (value << 8)


    def _get_current_state_at(self, key: int) -> int:
        """Get glfw mouse button number stored current state

        Parameters
        ---
        key : int
            glfw mouse button number

        Returns
        ---
        int
            currently stored current state for that key
        """
        return 0xFF & self.states[key]


    def _get_previous_state_at(self, key: int) -> int:
        """Get glfw mouse button number stored previous state

        Parameters
        ---
        key : int
            glfw mouse button number

        Returns
        ---
        int
            currently stored previous state for that key
        """
        return 0xFF & (self.states[key] >> 8)


    def get_state(self, glfw_mouse_state: tuple[int, int]) -> MouseState:
        """Mouse button pressed

        Parameters
        ---
        glfw_mouse_state : tuple[ int, int ]
            glfw mouse button number and its glfw state

        Example
        ---
        ```python

        win = GlWindow()
        mouse = Mouse()

        state = win.get_mouse_state(glfw.MOUSE_BUTTON_LEFT)
        
        if mouse.get_state(state) == MouseState.PRESSED:
            print('pressed')
        ```

        Returns
        ---
        MouseState
            is its current state held, pressed or released
        """
        key, state = glfw_mouse_state
        if key > 3:
            return MouseState.DEFAULT

        tmp = self._get_current_state_at(key)
        self._set_previous_state_at(key, tmp)
        self._set_current_state_at(key, state)

        if self._get_previous_state_at(key) == 0:
            if self._get_current_state_at(key) == 0:
                return MouseState.DEFAULT
            else:
                # pressed
                return MouseState.PRESSED
        else:
            if self._get_current_state_at(key) == 0:
                # released
                return MouseState.RELEASED
            else:
                # held
                return MouseState.HELD
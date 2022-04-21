"""Mouse
"""
from dataclasses import dataclass, field
from enum import Enum, auto


# ---


class MouseState(Enum):
    PRESSED= auto()
    RELEASED= auto()
    HELD= auto()
    DEFAULT= auto()


# ---


@dataclass(eq= False, repr= False, slots= True)
class Mouse:
    states: list[int]= field(default_factory=list)

    def __post_init__(self):
        self.states= [0xFF]*3


    def _set_current_state_at(self, key: int, value: int) -> None:
        """Set glfw mouse button number stored current state

        Parameters
        ---
        key : int
            glfw mouse button number
        value : int
            glfw mouse button state
        """
        self.states[key]= (self.states[key] & 0xFFFFFF00) | value


    def _set_previous_state_at(self, key: int, value: int) -> None:
        """Set glfw mouse button number stored previous state

        Parameters
        ---
        key : int
            glfw mouse button number
        value : int
            glfw mouse button state
        """
        self.states[key]= (self.states[key] & 0xFFFF00FF) | (value << 8)


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
        key, current= glfw_mouse_state
        result= MouseState.DEFAULT

        if key < 3:
            prev = self._get_current_state_at(key)
            self._set_previous_state_at(key, prev)
            self._set_current_state_at(key, current)

            if self._get_previous_state_at(key) == 0:
                if self._get_current_state_at(key) == 0:
                    result= MouseState.DEFAULT
                else:
                    # pressed
                    result= MouseState.PRESSED
            else:
                if self._get_current_state_at(key) == 0:
                    # released
                    result= MouseState.RELEASED
                else:
                    # held
                    result= MouseState.HELD

        return result

    def is_button_held(self, key_state: tuple[int, int]) -> bool:
        """Helper function for mouse button held down state

        Parameters
        ---
        key_state : tuple[int, int]

        Returns
        ---
        bool
        """
        return self.get_state(key_state) is MouseState.HELD

    def is_button_pressed(self, key_state: tuple[int, int]) -> bool:
        """Helper function for mouse button pressed state

        Parameters
        ---
        key_state : tuple[int, int]

        Returns
        ---
        bool
        """
        return self.get_state(key_state) is MouseState.PRESSED

    def is_button_released(self, key_state: tuple[int, int]) -> bool:
        """Helper function for mouse button released state

        Parameters
        ---
        key_state : tuple[int, int]

        Returns
        ---
        bool
        """
        return self.get_state(key_state) is MouseState.RELEASED
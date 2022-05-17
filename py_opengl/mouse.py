"""Mouse
"""
from enum import Enum, auto


# ---


class MouseState(Enum):
    PRESSED= auto()
    RELEASED= auto()
    HELD= auto()
    DEFAULT= auto()


# ---


class Mouse:

    __slots__= ('states',)

    def __init__(self) -> None:
        self.states: list[int]= [0xFF]*3

    def _set_current_state_at(self, key: int, value: int) -> None:
        """Set glfw mouse button number stored current state
        """
        self.states[key]= (self.states[key] & 0xFFFFFF00) | value


    def _set_previous_state_at(self, key: int, value: int) -> None:
        """Set glfw mouse button number stored previous state
        """
        self.states[key]= (self.states[key] & 0xFFFF00FF) | (value << 8)


    def _get_current_state_at(self, key: int) -> int:
        """Get glfw mouse button number stored current state
        """
        return 0xFF & self.states[key]


    def _get_previous_state_at(self, key: int) -> int:
        """Get glfw mouse button number stored previous state
        """
        return 0xFF & (self.states[key] >> 8)


    def get_state(self, glfw_mouse_state: tuple[int, int]) -> MouseState:
        """Mouse button pressed

        use GlWindow function 'get_mouse_state' for glfw_mouse_state
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
        """
        return self.get_state(key_state) is MouseState.HELD

    def is_button_pressed(self, key_state: tuple[int, int]) -> bool:
        """Helper function for mouse button pressed state
        """
        return self.get_state(key_state) is MouseState.PRESSED

    def is_button_released(self, key_state: tuple[int, int]) -> bool:
        """Helper function for mouse button released state
        """
        return self.get_state(key_state) is MouseState.RELEASED
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
    states: list[int]= field(default_factory=list)

    def __post_init__(self):
        self.states= [0xFF] * KEYBOARD_SIZE

    def _set_current_state_at(self, key: int, value: int) -> None:
        self.states[key]= (self.states[key] & 0xFFFFFF00) | value

    def _set_previous_state_at(self, key: int, value: int) -> None:
        self.states[key]= (self.states[key] & 0xFFFF00FF) | (value << 8)

    def _get_current_state_at(self, key: int) -> int:
        return 0xFF & self.states[key]

    def _get_previous_state_at(self, key: int) -> int:
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

# Keyboard_CALLBACK = Callable[[Any, int], None]
# Keyboard_DICT = dict[int, Keyboard_CALLBACK]
# class KeyboardS:
#     _instance: Any|None= None
    
#     def __new__(cls: type['KeyboardS']) -> 'KeyboardS':
#         if cls._instance is None:
#             cls._instance= super(KeyboardS, cls).__new__(cls)
#         return cls._instance

#     def setup_window(self, window: Any|None = None):
#         self.data: Keyboard_DICT= {}
#         self.window: Any|None= window
#         self.keys: list[int]= [0xFF]*1024
#         self.state: int= -1

#         def cbfun(window, key: int, scancode: int, action: int, mods: int):
#             if key < 1024:
#                 current: int= action
#                 prev: int= 0xFF & self.keys[key] #get current
#                 self.keys[key]= (self.keys[key] & 0xFFFF00FF) | (prev << 8) #set prev
#                 self.keys[key]= (self.keys[key] & 0xFFFFFF00) | current #set current

#                 if 0xFF & (self.keys[key] >> 8) == 0:
#                     if 0xFF & self.keys[key] == 0:
#                         self.state= -1
#                     else:
#                         self.state= 1
#                 else:
#                     if 0xFF & self.keys[key] == 1:
#                         self.state= 0
#                     else:
#                         self.state= 2

#             if key in self.data:
#                 self.data[key](window, self.state)
        
#         glfw.set_key_callback(self.window, cbfun)

#     def add_key(self, key: int, cbfun: Keyboard_CALLBACK) -> None:
#         self.data[key]= cbfun

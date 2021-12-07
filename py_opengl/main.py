'''
'''
import ctypes
import glfw

from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from OpenGL import GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
from py_opengl import glm
from typing import Any, Final


# --- GLOBALS


SCREEN_WIDTH: Final[int] = 500
SCREEN_HEIGHT: Final[int] = 500


# --- HELPERS


def c_array(arr: list[float]):
    ''' '''
    # Example:
    # create c type array and unpack data into it
    # arr = (ctypes.c_float * 3) = [0, 0, 0]
    return (gl.GLfloat * len(arr))(*arr)


def c_cast(offset: int):
    ''' '''
    # cast(3 * sizeof(float), void*)
    return ctypes.cast(offset, ctypes.c_void_p)


# --- C TYPES


NULL_PTR = ctypes.c_void_p(0)
FLOAT_SIZE = ctypes.sizeof(ctypes.c_float)
UINT_SIZE = ctypes.sizeof(ctypes.c_uint16)


# --- CLOCK


@dataclass(eq=False, repr=False, slots=True)
class Clock:
    ticks: int = 0
    delta: float = 1.0 / 60.0
    last_time_step: float = 0.0
    accumalate: float = 0.0


def clock_update(clock: Clock) -> None:
    '''Update clock'''
    current_time_step: float = glfw.get_time()
    elapsed: float = current_time_step - clock.last_time_step
    clock.last_time_step = current_time_step
    clock.accumalate += elapsed

    while clock.accumalate >= clock.delta:
        clock.accumalate -= clock.delta
        clock.ticks += 1.0


# --- TRANSFORM


@dataclass(eq=False, repr=False, slots=True)
class Transform:
    position: glm.Vec3 = glm.Vec3()
    scale: glm.Vec3 = glm.Vec3(1.0, 1.0, 1.0)
    rotation: glm.Quaternion = glm.Quaternion(w=1.0)


def tr_get_translation(tr: Transform) -> glm.Mat4:
    '''Get mat4 transform'''

    t = glm.m4_create_identity()
    t = t * glm.m4_create_translation(tr.position)
    t = t * glm.qt_to_mat4(tr.rotation)
    t = t * glm.m4_create_scaler(tr.scale)

    return t


# --- SQUARE

# TODO()
@dataclass(eq=False, repr=False, slots=True)
class Square:
    verts: list[gl.GL_FLOAT] = field(default_factory=list)
    VAO: int = 0
    VBO: int = 0
    shader: int = 0


def sq_init(sq: Square) -> None:
    ''' '''
    sq.shader = gl.glCreateProgram()

    vertex_src = '''
    # version 430
    layout(location = 0) in vec3 a_pos;
    layout(location = 1) in vec3 a_col;

    out vec3 b_col;

    void main()
    {
        gl_Position = vec4(a_pos, 1.0);
        b_col = a_col;
    }
    '''

    fragment_src = '''
    # version 430

    in vec3 b_col;
    out vec4 c_col;

    void main()
    {
        c_col = vec4(b_col, 1.0);
    }
    '''
    sq.shader = compileProgram(
            compileShader(vertex_src, gl.GL_VERTEX_SHADER),
            compileShader(fragment_src, gl.GL_FRAGMENT_SHADER))

    sq.verts = [
            -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
            0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
            -0.5,  0.5, 0.0, 0.0, 0.0, 1.0,
            0.5,  0.5, 0.0, 1.0, 1.0, 1.0]

    vlen = len(sq.verts) * FLOAT_SIZE
    stride = 6 * FLOAT_SIZE
    step = 3 * FLOAT_SIZE

    sq.VAO = gl.glGenVertexArrays(1)
    sq.VBO = gl.glGenBuffers(1)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, sq.VBO)
    gl.glBindVertexArray(sq.VAO)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vlen, c_array(sq.verts), gl.GL_STATIC_DRAW)

    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, c_cast(0))
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, c_cast(step))


def sq_draw(sq: Square) -> None:
    ''' '''
    gl.glUseProgram(sq.shader)
    gl.glBindVertexArray(sq.VAO)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)


def sq_clean(sq: Square) -> None:
    gl.glDeleteProgram(sq.shader)
    gl.glDeleteVertexArrays(1, sq.VAO)
    gl.glDeleteBuffers(1, sq.VBO)


# --- GL WINDOW


def cb_window_resize(window, width, height):
    gl.glViewport(0, 0, width, height)


class GlWindowError(Exception):
    '''Custom error for gl window'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class GlWindow:
    window: Any = None
    width: int = 0
    height: int = 0
    title: str = "glfw_window"

    def __post_init__(self):
        self.window = glfw.create_window(
                self.width,
                self.height,
                self.title,
                None,
                None
        )

        if not self.window:
            raise GlWindowError('failed to init glfw window')


def glwin_set_window_callback(glwin: GlWindow, cb_func):
    glfw.set_window_size_callback(glwin.window, cb_func)


def glwin_should_close(glwin: GlWindow) -> bool:
    '''Close window'''
    return True if glfw.window_should_close(glwin.window) else False


def glwin_center_screen_position(glwin: GlWindow) -> None:
    '''Center glwindow to center of screen'''
    video = glfw.get_video_mode(glfw.get_primary_monitor())

    x: float = (video.size.width // 2) - (glwin.width // 2)
    y: float = (video.size.height // 2) - (glwin.height // 2)

    glfw.set_window_pos(glwin.window, x, y)


def glwin_mouse_pos(glwin: GlWindow) -> glm.Vec3:
    '''Get current cursor possition on current window'''
    cx, cy = glfw.get_cursor_pos(glwin.window)
    return glm.Vec3(x=cx, y=cy)


def glwin_mouse_state(glwin: GlWindow, button: int) -> tuple[int, int]:
    '''Get glfw mouse button state

    Parameters
    ---
    glwin: GlWindow
        glfw iwndow
    button: int
        glfw mouse button macro number
        left: 0
        right: 1
        middle: 2

    Returns
    ---
    tuple[int, int]:
        buttoncode: int
        keystate: int
            GLFW_RELEASE: 0
            GLFW_PRESS: 1
    '''
    return (button, glfw.get_mouse_button(glwin.window, button))


def glwin_key_state(glwin: GlWindow, key: int) -> tuple[int, int]:
    '''Get glfw keybutton state

    Parameters
    ---
    glwin: GlWindow
    key: int
        glfw keyboard macro number

    Returns
    ---
    tuple[int, int]:
        keycode: int
        keystate: int
            GLFW_RELEASE: 0
            GLFW_PRESS: 1
    '''
    return (key, glfw.get_key(glwin.window, key))


# --- CAMERA


@dataclass(eq=False, repr=False, slots=True)
class Camera:
    position: glm.Vec3 = glm.Vec3()
    front: glm.Vec3 = glm.Vec3(z=-1.0)
    up: glm.Vec3 = glm.Vec3(y=1.0)
    right: glm.Vec3 = glm.Vec3(x=1.0)
    aspect: float = 1.0

    fovy: float = glm.PIOVER2
    yaw: float = glm.PIOVER2 * -1.0     # -90.0 deg
    pitch: float = 0.0
    znear: float = 0.01
    zfar: float = 1000.0


def camera_update(cam: Camera) -> None:
    '''Update camera'''

    cam.front.x = glm.cos(cam.pitch) * glm.cos(cam.yaw)
    cam.front.y = glm.sin(cam.pitch)
    cam.front.z = glm.cos(cam.pitch) * glm.sin(cam.yaw)

    cam.front = glm.v3_unit(cam.front)
    cam.right = glm.v3_unit(glm.v3_cross(cam.front, glm.Vec3(y=1.0)))
    cam.up = glm.v3_unit(glm.v3_cross(cam.right, cam.front))


def camera_to_yaw(val: float) -> float:
    return glm.to_radians(val)


def camera_to_pitch(val: float) -> float:
    return glm.to_radians(glm.clamp(val, -89.0, 89.0))


def camera_to_fovy(val: float) -> float:
    return glm.to_radians(glm.clamp(val, 1.0, 45.0))


def camera_view_matrix(cam: Camera) -> glm.Mat4:
    '''Return Camera view matrix'''
    return glm.m4_look_at(
            cam.position,
            cam.position + cam.front,
            cam.up)


def camera_perspective_matrix(cam: Camera) -> glm.Mat4:
    '''Return camera projection matrix'''
    return glm.m4_perspective_fov(cam.fovy, cam.aspect, cam.znear, cam.zfar)


# --- KEYBOARD


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
    kb.states[key] = (kb.states[key] & 0xFFFFFF00) | value


def _keyboard_set_previous_at(kb: Keyboard, key: int, value: int) -> None:
    kb.states[key] = (kb.states[key] & 0xFFFF00FF) | (value << 8)


def _keyboard_get_current_at(kb: Keyboard, key: int) -> int:
    return 0xFF & kb.states[key]


def _keyboard_get_previous_at(kb: Keyboard, key: int) -> int:
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

    return KeyState.DEFAULT


# --- MOUSE


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
    mouse.states[key] = (mouse.states[key] & 0xFFFFFF00) | value


def _mouse_set_previous(mouse: Mouse, key: int, value: int) -> None:
    mouse.states[key] = (mouse.states[key] & 0xFFFF00FF) | (value << 8)


def _mouse_get_current(mouse: Mouse, key: int) -> int:
    return 0xFF & mouse.states[key]


def _mouse_get_previous(mouse: Mouse, key: int) -> int:
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

    return MouseState.DEFAULT


# --- MAIN


def main() -> None:
    ''' '''
    if not glfw.init():
        logger.error('failed to init glfw')
        return

    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

        glwin = GlWindow(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        glfw.make_context_current(glwin.window)
        glwin_center_screen_position(glwin)
        glwin_set_window_callback(glwin, cb_window_resize)

        clock = Clock()

        camera = Camera(
                position=glm.Vec3(z=3.0),
                aspect=SCREEN_WIDTH/SCREEN_HEIGHT)
        camera_s = 2.0

        sq = Square()
        sq_init(sq)

        # tr = Transform()

        keyboard = Keyboard()

        mouse = Mouse()
        first_move = True
        last_mp = glm.Vec3()
        mouse_sensitivity = 0.2

        while not glwin_should_close(glwin):
            clock_update(clock)

            gl.glClearColor(0.10, 0.10, 0.10, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_TRUE)
            # ---

            sq_draw(sq)

            # tr.rotation = glm.qt_from_axis(
            #        clock.ticks, glm.Vec3(x=0.1, y=0.5, z=0.2))

            # model = tr_get_translation(tr)
            # view = camera_view_matrix(camera)
            # projection = camera_perspective_matrix(camera)

            # mvp = model * view * projection
            # shader_set_m4(shader, 'mvp', mvp)

            # keyboard
            ks_w = glwin_key_state(glwin, glfw.KEY_W)
            if key_state(keyboard, ks_w) == KeyState.HELD:
                camera.position = camera.position + glm.v3_scale(
                        camera.front,
                        camera_s * clock.delta)

            ks_s = glwin_key_state(glwin, glfw.KEY_S)
            if key_state(keyboard, ks_s) == KeyState.HELD:
                camera.position = camera.position - glm.v3_scale(
                        camera.front,
                        camera_s * clock.delta)

            ks_a = glwin_key_state(glwin, glfw.KEY_A)
            if key_state(keyboard, ks_a) == KeyState.HELD:
                camera.position = camera.position + glm.v3_scale(
                        camera.right,
                        camera_s * clock.delta)

            ks_d = glwin_key_state(glwin, glfw.KEY_D)
            if key_state(keyboard, ks_d) == KeyState.HELD:
                camera.position = camera.position - glm.v3_scale(
                        camera.right,
                        camera_s * clock.delta)

            ks_space = glwin_key_state(glwin, glfw.KEY_Q)
            if key_state(keyboard, ks_space) == KeyState.HELD:
                camera.position = camera.position + glm.v3_scale(
                        camera.up,
                        camera_s * clock.delta)

            ks_ls = glwin_key_state(glwin, glfw.KEY_E)
            if key_state(keyboard, ks_ls) == KeyState.HELD:
                camera.position = camera.position - glm.v3_scale(
                        camera.up,
                        camera_s * clock.delta)

            # mouse
            ms = glwin_mouse_state(glwin, glfw.MOUSE_BUTTON_LEFT)
            current_mp = glwin_mouse_pos(glwin)
            if mouse_state(mouse, ms) == MouseState.HELD:
                if first_move:
                    last_mp = current_mp
                    first_move = False
                else:
                    new_mp = current_mp - last_mp
                    last_mp = current_mp

                    camera.yaw = camera.yaw - camera_to_yaw(
                            new_mp.x * mouse_sensitivity)

                    camera.pitch = camera.pitch + camera_to_pitch(
                            new_mp.y * mouse_sensitivity)

                    camera_update(camera)

            # ---
            glfw.swap_buffers(glwin.window)
            glfw.poll_events()

    except GlWindowError as window_err:
        logger.error(f'ERROR: {window_err}')
        glfw.terminate()

    except glm.Mat4Error as mat4_err:
        logger.error(f'ERROR: {mat4_err}')
        glfw.terminate()

    finally:
        logger.debug('CLOSED')

        sq_clean(sq)

        glfw.terminate()


if __name__ == '__main__':
    main()

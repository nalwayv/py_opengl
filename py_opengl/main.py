'''
'''
import glfw
import ctypes

from loguru import logger
from py_opengl import glm
from OpenGL import GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
from dataclasses import dataclass, field
from typing import Any, Final
from enum import Enum


# --- GLOBALS


SCREEN_WIDTH: Final[int] = 500
SCREEN_HEIGHT: Final[int] = 500


# --- HELPERS


def to_c_array(arr: list[float]):
    ''' '''
    # Example:
    # create c type array and unpack data into it
    # arr = (ctypes.c_float * 3) = [0, 0, 0]
    return (gl.GLfloat * len(arr))(*arr)


# --- C TYPES


NULL_PTR = ctypes.c_void_p(0)
FLOAT_SIZE = ctypes.sizeof(gl.GLfloat)

# --- CLOCK


@dataclass(eq=False, repr=False, slots=True)
class Clock:
    ticks: int = 0
    delta: float = 1.0 / 60.0
    last_time_step: float = 0.0
    accumalate: float = 0.0


def clock_update(clock: Clock) -> None:
    '''Update clock '''
    current_time_step: float = glfw.get_time()
    elapsed: float = current_time_step - clock.last_time_step
    clock.last_time_step = current_time_step
    clock.accumalate += elapsed

    while clock.accumalate >= clock.delta:
        clock.accumalate -= clock.delta
        clock.ticks += 1.0


# --- CUBE


@dataclass(eq=False, repr=False, slots=True)
class Cube:
    width: float = 1.0
    height: float = 1.0
    depth: float = 1.0
    verts: list[float] = field(default_factory=list)
    color: list[float] = field(default_factory=list)
    points: int = 8
    data_size: int = 108
    components: int = 3

    def __post_init__(self):
        wp: float = self.width / 2.0
        hp: float = self.height / 2.0
        dp: float = self.depth / 2.0

        wn: float = wp * -1.0
        hn: float = hp * -1.0
        dn: float = dp * -1.0

        self.verts = [
                wp, hn, dp, wp, hn, dn, wp, hp, dn,
                wp, hn, dp, wp, hp, dn, wp, hp, dp,
                wn, hn, dn, wn, hn, dp, wn, hp, dp,
                wn, hn, dn, wn, hp, dp, wn, hp, dn,
                wn, hp, dp, wp, hp, dp, wp, hp, dn,
                wn, hp, dp, wp, hp, dn, wn, hp, dn,
                wn, hn, dn, wp, hn, dn, wp, hn, dp,
                wn, hn, dn, wp, hn, dp, wn, hn, dp,
                wn, hn, dp, wp, hn, dp, wp, hp, dp,
                wn, hn, dp, wp, hp, dp, wn, hp, dp,
                wp, hn, dn, wn, hn, dn, wn, hp, dn,
                wp, hn, dn, wn, hp, dn, wp, hp, dn]

        self.color = [
            1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5,
            1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5,
            0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0,
            0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0,
            0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5,
            0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5,
            0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0,
            0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0,
            0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0,
            0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0,
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]

# --- SHADER


@dataclass(eq=False, repr=False, match_args=False, slots=True)
class Shader:
    program_id: int = 0

    def __post_init__(self):
        self.program_id = gl.glCreateProgram()


def shader_default(shader: Shader) -> None:
    '''Simple Shader'''

    vert: str = '''#version 430 core
    layout (location = 0) in vec3 a_position;
    layout (location = 1) in vec3 a_color;

    out vec3 b_col;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        b_col = a_color;
        mat4 mvp = projection * view * model;
        gl_Position = mvp * vec4(a_position, 1.0);
    }
    '''

    frag: str = '''#version 430 core

    in vec3 b_col;
    out vec4 c_col;

    void main () {
        c_col = vec4(b_col, 1.0);
    }
    '''

    shader.program_id = compileProgram(
        compileShader(vert, gl.GL_VERTEX_SHADER),
        compileShader(frag, gl.GL_FRAGMENT_SHADER)
    )


def shader_clean(shader: Shader) -> None:
    '''Delete shader program id'''
    gl.glDeleteProgram(shader.program_id)


def shader_use(shader: Shader) -> None:
    '''Use shader program id'''
    gl.glUseProgram(shader.program_id)


def shader_set_vec3(shader: Shader, var_name: str, data: glm.Vec3) -> None:
    '''Set a global uniform vec3 variable within shader program'''
    location_id = gl.glGetUniformLocation(shader.program_id, var_name)

    gl.glUniform3f(location_id, data.x, data.y, data.z)


def shader_set_m4(shader: Shader, var_name: str, data: glm.Mat4) -> None:
    '''Set a global uniform mat4 variable within shader program'''
    location_id = gl.glGetUniformLocation(shader.program_id, var_name)

    gl.glUniformMatrix4fv(
            location_id,
            1,
            gl.GL_FALSE,
            glm.m4_to_multi_array(data))


# --- VBO

@dataclass(eq=False, repr=False, slots=True)
class Vbo:
    vao: int = 0
    data_size: int = 1   # len of data passed in
    components: int = 3     # x y z
    normalized: bool = False
    vbos: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.vao = gl.glGenVertexArrays(1)


def vbo_clean(vbo: Vbo) -> None:
    '''Clean vbo'''
    gl.glDeleteVertexArrays(1, vbo.vao)
    for v in vbo.vbos:
        gl.glDeleteBuffers(1, v)


def vbo_add_data(vbo: Vbo, arr: list[float]) -> None:
    '''Add data to vbo'''
    v_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, v_buffer)
    gl.glBindVertexArray(vbo.vao)

    vbo.vbos.append(v_buffer)

    normal = gl.GL_TRUE if vbo.normalized else gl.GL_FALSE

    gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            len(arr) * FLOAT_SIZE,
            to_c_array(arr),
            gl.GL_STATIC_DRAW
    )

    gl.glVertexAttribPointer(
            len(vbo.vbos) - 1,
            vbo.components,
            gl.GL_FLOAT,
            normal,
            0,
            NULL_PTR
    )

    gl.glEnableVertexAttribArray(len(vbo.vbos) - 1)


def vbo_use(vbo: Vbo) -> None:
    '''Bind vbo to vertex Array'''

    count = vbo.data_size // vbo.components
    if count <= 0:
        return

    gl.glBindVertexArray(vbo.vao)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, count)


# --- GL WINDOW


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

    fovy: float = glm.PIOVER2 * 0.5     # 45 deg
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
            glm.v3_add(cam.position, cam.front),
            cam.up)


def camera_projection_matrix(cam: Camera) -> glm.Mat4:
    '''Return camera projection matrix'''
    return glm.m4_projection(cam.fovy, cam.aspect, cam.znear, cam.zfar)


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
        glwin = GlWindow(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        glfw.make_context_current(glwin.window)
        glwin_center_screen_position(glwin)

        clock = Clock()

        camera = Camera(
                position=glm.Vec3(z=3.0),
                aspect=SCREEN_WIDTH/SCREEN_HEIGHT)

        cube = Cube()

        shader: Shader = Shader()
        shader_default(shader)

        vbo: Vbo = Vbo(data_size=cube.data_size)
        vbo_add_data(vbo, cube.verts)
        vbo_add_data(vbo, cube.color)

        keyboard = Keyboard()
        camera_s = 2.0

        mouse = Mouse()
        first_move = True
        last_mp = glm.Vec3()
        mouse_speed = 0.2

        while not glwin_should_close(glwin):
            clock_update(clock)

            gl.glClearColor(0.10, 0.10, 0.10, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_TRUE)
            # ---

            shader_use(shader)
            vbo_use(vbo)

            model = glm.m4_from_axis(clock.ticks, glm.Vec3(x=0.5, y=0.5))
            view = camera_view_matrix(camera)
            proj = camera_projection_matrix(camera)

            shader_set_m4(shader, 'model', model)
            shader_set_m4(shader, 'view', view)
            shader_set_m4(shader, 'projection', proj)

            # keyboard
            ks_w = glwin_key_state(glwin, glfw.KEY_W)
            if key_state(keyboard, ks_w) == KeyState.HELD:
                speed = camera_s * clock.delta
                new_pos = glm.v3_scale(camera.front, speed)
                camera.position = glm.v3_add(camera.position, new_pos)

            ks_s = glwin_key_state(glwin, glfw.KEY_S)
            if key_state(keyboard, ks_s) == KeyState.HELD:
                speed = camera_s * clock.delta
                new_pos = glm.v3_scale(camera.front, speed)
                camera.position = glm.v3_sub(camera.position, new_pos)

            ks_a = glwin_key_state(glwin, glfw.KEY_A)
            if key_state(keyboard, ks_a) == KeyState.HELD:
                speed = camera_s * clock.delta
                new_pos = glm.v3_scale(camera.right, speed)
                camera.position = glm.v3_add(camera.position, new_pos)

            ks_d = glwin_key_state(glwin, glfw.KEY_D)
            if key_state(keyboard, ks_d) == KeyState.HELD:
                speed = camera_s * clock.delta
                new_pos = glm.v3_scale(camera.right, speed)
                camera.position = glm.v3_sub(camera.position, new_pos)

            ks_space = glwin_key_state(glwin, glfw.KEY_Q)
            if key_state(keyboard, ks_space) == KeyState.HELD:
                speed = camera_s * clock.delta
                new_pos = glm.v3_scale(camera.up, speed)
                camera.position = glm.v3_add(camera.position, new_pos)

            ks_ls = glwin_key_state(glwin, glfw.KEY_E)
            if key_state(keyboard, ks_ls) == KeyState.HELD:
                speed = camera_s * clock.delta
                new_pos = glm.v3_scale(camera.up, speed)
                camera.position = glm.v3_sub(camera.position, new_pos)

            # mouse move
            ms = glwin_mouse_state(glwin, glfw.MOUSE_BUTTON_LEFT)
            current_mp = glwin_mouse_pos(glwin)
            if mouse_state(mouse, ms) == MouseState.HELD:
                if first_move:
                    last_mp = current_mp
                    first_move = False
                else:
                    new_dir = glm.v3_sub(current_mp, last_mp)
                    last_mp = current_mp

                    camera.yaw -= camera_to_yaw(new_dir.x * mouse_speed)
                    camera.pitch += camera_to_pitch(new_dir.y * mouse_speed)
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
        vbo_clean(vbo)
        shader_clean(shader)
        glfw.terminate()


if __name__ == '__main__':
    main()

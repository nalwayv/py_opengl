'''PY OPENGL
'''
import ctypes
import glfw

from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from OpenGL import GL
from OpenGL.GL.shaders import compileShader, compileProgram
from py_opengl import glm
from typing import Any, Final


# --- GLOBALS


SCREEN_WIDTH: Final[int] = 500
SCREEN_HEIGHT: Final[int] = 500


# --- C HELPERS


def c_array(arr: list[float]):
    '''Convert list to ctype array'''
    # return (gl.GLfloat * len(arr))(*arr)
    return (ctypes.c_float * len(arr))(*arr)


def c_cast(offset: int):
    '''Cast to ctype void pointer (void*)(offset)'''
    return ctypes.c_void_p(offset)


FLOAT_SIZE = 4
UINT_SIZE = 4

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


def trans_get_transform(tr: Transform) -> glm.Mat4:
    '''Get mat4 transform'''

    t = glm.m4_create_identity()
    t = t * glm.m4_create_translation(tr.position)
    t = t * glm.qt_to_mat4(tr.rotation)
    t = t * glm.m4_create_scaler(tr.scale)

    return t

# --- SHADER


@dataclass(eq=False, repr=False, slots=True)
class Shader:
    program_id: int = 0

    def __post_init__(self):
        self.program_id = GL.glCreateProgram()


def shader_compile(shader: Shader, v_filepath: str, f_filepath: str) -> None:
    '''Simple shader'''
    shader.program_id = compileProgram(
            compileShader(v_filepath, GL.GL_VERTEX_SHADER),
            compileShader(f_filepath, GL.GL_FRAGMENT_SHADER))


def shader_clean(shader: Shader) -> None:
    '''Delete shader program id'''
    GL.glDeleteProgram(shader.program_id)


def shader_use(shader: Shader) -> None:
    '''Use shader program id'''
    GL.glUseProgram(shader.program_id)


def shader_set_vec3(shader: Shader, var_name: str, data: glm.Vec3) -> None:
    '''Set a global uniform vec3 variable within shader program'''
    location_id = GL.glGetUniformLocation(shader.program_id, var_name)
    GL.glUniform3f(location_id, data.x, data.y, data.z)


def shader_set_m4(shader: Shader, var_name: str, data: glm.Mat4) -> None:
    '''Set a global uniform mat4 variable within shader program'''
    location_id = GL.glGetUniformLocation(shader.program_id, var_name)
    GL.glUniformMatrix4fv(
            location_id,
            1,
            GL.GL_FALSE,
            glm.m4_multi_array(data))


# --- VBO

@dataclass(eq=False, repr=False, slots=True)
class Vbo:
    vao: int = 0
    components: int = 3     # x y z
    normalized: bool = False
    vbos: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.vao = GL.glGenVertexArrays(1)


def vbo_draw(vbo: Vbo) -> None:
    '''Bind vbo to vertex Array'''
    GL.glBindVertexArray(vbo.vao)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, vbo.components)


def vbo_clean(vbo: Vbo) -> None:
    '''Clean vbo'''
    GL.glDeleteVertexArrays(1, vbo.vao)
    for v in vbo.vbos:
        GL.glDeleteBuffers(1, v)


def vbo_add_data(vbo: Vbo, arr: list[float]) -> None:
    '''Add data to vbo'''
    v_buffer = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, v_buffer)
    GL.glBindVertexArray(vbo.vao)
    vbo.vbos.append(v_buffer)

    normal: bool = GL.GL_TRUE if vbo.normalized else GL.GL_FALSE
    size_of: int = len(arr) * FLOAT_SIZE

    GL.glBufferData(GL.GL_ARRAY_BUFFER, size_of, c_array(arr), GL.GL_STATIC_DRAW)

    at = len(vbo.vbos) - 1
    GL.glVertexAttribPointer(at, vbo.components, GL.GL_FLOAT, normal, 0, c_cast(0))
    GL.glEnableVertexAttribArray(at)


# --- SQUARE


# TODO()
@dataclass(eq=False, repr=False, slots=True)
class Triangle:
    vbo: Any = None
    shader: Any = None

    def __post_init__(self):
        self.vbo = Vbo()
        self.shader = Shader()


def tri_init(tri: Triangle) -> None:
    '''Initialize triangle
    Parameters
    ---
    tri: Triangle

    Returns
    ---
    None
    '''

    vert_src = '''
    # version 430 core

    layout (location = 0) in vec3 a_pos;
    layout (location = 1) in vec3 a_col;

    out vec3 b_col;

    uniform mat4 mvp;

    void main(void)
    {
        b_col = a_col;
        gl_Position = mvp * vec4(a_pos, 1.0);
    }
    '''

    frag_src = '''
    # version 430 core

    in vec3 b_col;

    out vec4 c_col;

    void main()
    {
        c_col = vec4(b_col, 1.0);
    }
    '''

    shader_compile(tri.shader, vert_src, frag_src)

    verts = [
            0.5, -0.5, 0.0,
            -0.5, -0.5, 0.0,
            0.0, 0.5, 0.0]

    color = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]

    vbo_add_data(tri.vbo, verts)
    vbo_add_data(tri.vbo, color)


def tri_draw(tri: Triangle) -> None:
    '''Render triangle to screen'''
    shader_use(tri.shader)
    vbo_draw(tri.vbo)


def tri_clean(tri: Triangle) -> None:
    '''Clean up triangle shader and vbo'''
    shader_clean(tri.shader)
    vbo_clean(tri.vbo)


# --- GL WINDOW


def cb_window_resize(window, width, height):
    ''' '''
    GL.glViewport(0, 0, width, height)


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
    ''' '''
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
    '''Get current cursor possition on current window

    Parameters
    ---
    glwin: GlWindow
        glfw window

    Returns
    ---
    Vec3: vec3(x, y, z)
    '''
    cx, cy = glfw.get_cursor_pos(glwin.window)
    return glm.Vec3(x=cx, y=cy)


def glwin_mouse_state(glwin: GlWindow, button: int) -> tuple[int, int]:
    '''Get glfw mouse button state

    Parameters
    ---
    glwin: GlWindow
        glfw window
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
    '''Update camera

    Parameters
    ---
    cam: Camera

    Returns
    ---
    None
    '''
    cam.front.x = glm.cos(cam.pitch) * glm.cos(cam.yaw)
    cam.front.y = glm.sin(cam.pitch)
    cam.front.z = glm.cos(cam.pitch) * glm.sin(cam.yaw)

    cam.front = glm.v3_unit(cam.front)
    cam.right = glm.v3_unit(glm.v3_cross(cam.front, glm.Vec3(y=1.0)))
    cam.up = glm.v3_unit(glm.v3_cross(cam.right, cam.front))


def camera_yaw(val: float) -> float:
    '''Helper function for camera yaw

    Parameters
    ---
    val: float
        value in degrees,
        will be converted into radians within function

    Example:
    ---
    camera.yaw += camera_yaw(45.5)

    Returns
    ---
    float: yaw value in radians
    '''
    return glm.to_radians(val)


def camera_pitch(val: float) -> float:
    '''Helper function for camera pitch

    Parameters
    ---
    val: float
        value in degrees,
        will be converted into radians within function

    Example
    ---
    camera.pitch += camera_pitch(45.5)

    Returns
    ---
    float: pitch value in radians
    '''

    return glm.to_radians(glm.clamp(val, -89.0, 89.0))


def camera_fovy(val: float) -> float:
    '''Helper function for camera fovy

    Parameters
    ---
    val: float
        value in degrees,
        will be converted into radians within function

    Example
    ---
    camera.fovy += camera_fovy(45.5)

    Returns
    ---
    float: fovy value in radians
    '''
    return glm.to_radians(glm.clamp(val, 1.0, 45.0))


def camera_view_matrix(cam: Camera) -> glm.Mat4:
    '''Return Camera view matrix

    Parameters
    ---
    cam: Camera

    Returns
    ---
    Mat4: camera view matrix 4x4
    '''
    return glm.m4_look_at(
            cam.position,
            cam.position + cam.front,
            cam.up)


def camera_perspective_matrix(cam: Camera) -> glm.Mat4:
    '''Return camera projection matrix

    Parameters
    ---
    cam: Camera

    Returns
    ---
    Mat4: camera projection matrix 4x4
    '''
    return glm.m4_projection(cam.fovy, cam.aspect, cam.znear, cam.zfar)
#    return glm.m4_perspective_fov(cam.fovy, cam.aspect, cam.znear, cam.zfar)


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
    ''' '''
    kb.states[key] = (kb.states[key] & 0xFFFFFF00) | value


def _keyboard_set_previous_at(kb: Keyboard, key: int, value: int) -> None:
    ''' '''
    kb.states[key] = (kb.states[key] & 0xFFFF00FF) | (value << 8)


def _keyboard_get_current_at(kb: Keyboard, key: int) -> int:
    ''' '''
    return 0xFF & kb.states[key]


def _keyboard_get_previous_at(kb: Keyboard, key: int) -> int:
    ''' '''
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
    ''' '''
    mouse.states[key] = (mouse.states[key] & 0xFFFFFF00) | value


def _mouse_set_previous(mouse: Mouse, key: int, value: int) -> None:
    ''' '''
    mouse.states[key] = (mouse.states[key] & 0xFFFF00FF) | (value << 8)


def _mouse_get_current(mouse: Mouse, key: int) -> int:
    ''' '''
    return 0xFF & mouse.states[key]


def _mouse_get_previous(mouse: Mouse, key: int) -> int:
    ''' '''
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
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)

        glwin = GlWindow(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        glfw.make_context_current(glwin.window)
        glwin_center_screen_position(glwin)
        glwin_set_window_callback(glwin, cb_window_resize)

        clock = Clock()

        camera = Camera(position=glm.Vec3(z=3.0), aspect=SCREEN_WIDTH/SCREEN_HEIGHT)

        camera_speed = 2.0

        tri = Triangle()
        tri_init(tri)

        transform = Transform()

        keyboard = Keyboard()

        mouse = Mouse()
        first_move = True
        last_mp = glm.Vec3()
        mouse_sensitivity = 0.2

        while not glwin_should_close(glwin):
            clock_update(clock)

            GL.glClearColor(0.2, 0.3, 0.3, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # ---

            tri_draw(tri)

            transform.rotation = glm.qt_from_axis(clock.ticks, glm.Vec3(x=0.1, y=0.5, z=0.1))

            model = trans_get_transform(transform)
            view = camera_view_matrix(camera)
            projection = camera_perspective_matrix(camera)

            mvp = model * view * projection
            shader_set_m4(tri.shader, 'mvp', mvp)

            # keyboard
            ks_w = glwin_key_state(glwin, glfw.KEY_W)
            if key_state(keyboard, ks_w) == KeyState.HELD:
                camera.position = camera.position + glm.v3_scale(camera.front, camera_speed * clock.delta)

            ks_s = glwin_key_state(glwin, glfw.KEY_S)
            if key_state(keyboard, ks_s) == KeyState.HELD:
                camera.position = camera.position - glm.v3_scale(camera.front, camera_speed * clock.delta)

            ks_a = glwin_key_state(glwin, glfw.KEY_A)
            if key_state(keyboard, ks_a) == KeyState.HELD:
                camera.position = camera.position + glm.v3_scale(camera.right, camera_speed * clock.delta)

            ks_d = glwin_key_state(glwin, glfw.KEY_D)
            if key_state(keyboard, ks_d) == KeyState.HELD:
                camera.position = camera.position - glm.v3_scale(camera.right, camera_speed * clock.delta)

            ks_space = glwin_key_state(glwin, glfw.KEY_Q)
            if key_state(keyboard, ks_space) == KeyState.HELD:
                camera.position = camera.position + glm.v3_scale(camera.up, camera_speed * clock.delta)

            ks_ls = glwin_key_state(glwin, glfw.KEY_E)
            if key_state(keyboard, ks_ls) == KeyState.HELD:
                camera.position = camera.position - glm.v3_scale(camera.up, camera_speed * clock.delta)

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

                    camera.yaw -= camera_yaw(new_mp.x * mouse_sensitivity)
                    camera.pitch += camera_pitch(new_mp.y * mouse_sensitivity)

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

        tri_clean(tri)

        glfw.terminate()


if __name__ == '__main__':
    main()

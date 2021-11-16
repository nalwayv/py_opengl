'''
'''
import glfw
import ctypes

from py_opengl import glm

from OpenGL import GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

# --- GLOBALS


SCREEN_WIDTH: int = 500
SCREEN_HEIGHT: int = 500


# --- HELPERS


def to_c_array(arr: list[float]):
    ''' '''
    # Example:
    # arr = (ctypes.c_float * 10)
    return (gl.GLfloat * len(arr))(*arr)


# --- C TYPES


NULL_PTR = ctypes.c_void_p(0)
FLOAT_SIZE = ctypes.sizeof(gl.GLfloat)


# --- TRANSFORM


@dataclass(eq=False, repr=False, slots=True)
class Transform:
    position: glm.Vec3 = glm.Vec3()
    scale: glm.Vec3 = glm.Vec3(1.0, 1.0, 1.0)
    angle_radians: float = 0.0


def transform_get_transform_m4(trans: Transform) -> glm.Mat4:
    '''Get transform matrix 4x4'''
    r: glm.Mat4 = glm.m4_from_axis(trans.angle_radians, glm.Vec3(z=1.0))
    t: glm.Mat4 = glm.m4_init_translate(trans.position)
    s: glm.Mat4 = glm.m4_init_scaler(trans.scale)

    return glm.m4_multiply_m4s(r, t, s)


def transform_get_inv_transform_m4(trans: Transform) -> glm.Mat4:
    '''Get inverse transform matrix 4x4'''
    return glm.m4_inverse(transform_get_transform_m4(trans))


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


# TODO(14/11/1021) ...
@dataclass(eq=False, repr=False, slots=True)
class Cube:
    width: float = 1.0
    height: float = 1.0
    depth: float = 1.0
    verts: list[float] = field(default_factory=list)

    points: int = 8
    data_size: int = 108
    components: int = 3

    def __post_init__(self):
        w: float = self.width / 2.0
        h: float = self.height / 2.0
        d: float = self.depth / 2.0

        # point data
        p0 = [-w, -h, -d]
        p1 = [w, -h, -d]
        p2 = [-w, h, -d]
        p3 = [w, h, -d]
        p4 = [-w, -h, d]
        p5 = [w, -h, d]
        p6 = [-w, h, d]
        p7 = [w, h, d]

        points = [
                p5, p1, p3,
                p5, p3, p7,
                p0, p4, p6,
                p0, p6, p2,
                p6, p7, p3,
                p6, p3, p2,
                p0, p1, p5,
                p0, p5, p4,
                p4, p5, p7,
                p4, p7, p6,
                p1, p0, p2,
                p1, p2, p3]

        # flattern array
        from functools import reduce
        from operator import iconcat
        self.verts = reduce(iconcat, points, [])


# --- SHADER


@dataclass(eq=False, repr=False, match_args=False, slots=True)
class Shader:
    program_id: int = 0

    def __post_init__(self):
        self.program_id = gl.glCreateProgram()


def shader_default(shader: Shader) -> None:
    '''Simple Shader'''

    vert: str = '''#version 330 core
    layout (location = 0) in vec3 a_position;

    out vec3 b_col;

    uniform vec3 color;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        b_col = color;
        mat4 mvp = projection * view * model;
        gl_Position = mvp * vec4(a_position, 1.0);
    }
    '''

    frag: str = '''#version 330 core

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


class GlWindowErr(Exception):
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
            raise GlWindowErr('failed to init glfw window')


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
    cx, cy = glfw.get_cursor_pos(glwin)
    return glm.Vec3(x=cx, y=cy)


def glwin_mouse_state(glwin: GlWindow, button: int) -> int:
    '''Get glfw mouse button state'''
    return glfw.get_mouse_button(glwin.window, button)


# --- CAMERA


@dataclass(eq=False, repr=False, slots=True)
class Camera:
    position: glm.Vec3 = glm.Vec3()
    front: glm.Vec3 = glm.Vec3(z=-1.0)
    up: glm.Vec3 = glm.Vec3(y=1.0)
    right: glm.Vec3 = glm.Vec3(x=1.0)
    aspect: float = 1.0

    fovy: float = glm.PIOVER2
    yaw: float = glm.PIOVER2 * -1.0
    pitch: float = 0.0
    znear: float = 0.01
    zfar: float = 1000.0


def camera_update_pitch(cam: Camera, angR: float) -> None:
    low: float = glm.to_radians(-89.0)
    high: float = glm.to_radians(89.0)
    cam.pitch = glm.clamp(angR, low, high)

    x: float = glm.cos(cam.yaw) * glm.cos(cam.pitch)
    y: float = glm.sin(cam.pitch)
    z: float = glm.sin(cam.yaw) * glm.cos(cam.pitch)

    cam.front = glm.v3_unit(glm.Vec3(x, y, z))
    cam.right = glm.v3_unit(glm.v3_cross(cam.front, glm.Vec3(y=1.0)))
    cam.up = glm.v3_unit(glm.v3_cross(cam.right, cam.front))


def camera_update_yaw(cam: Camera, angR: float) -> None:
    cam.yaw = angR

    x: float = glm.cos(cam.yaw) * glm.cos(cam.pitch)
    y: float = glm.sin(cam.pitch)
    z: float = glm.sin(cam.yaw) * glm.cos(cam.pitch)

    cam.front = glm.v3_unit(glm.Vec3(x, y, z))
    cam.right = glm.v3_unit(glm.v3_cross(cam.front, glm.Vec3(y=1.0)))
    cam.up = glm.v3_unit(glm.v3_cross(cam.right, cam.front))


def camera_update_fovy(cam: Camera, angR: float) -> None:
    low: float = glm.to_radians(1.0)
    high: float = glm.to_radians(45.0)
    cam.fovy = glm.clamp(angR, low, high)

    x: float = glm.cos(cam.yaw) * glm.cos(cam.pitch)
    y: float = glm.sin(cam.pitch)
    z: float = glm.sin(cam.yaw) * glm.cos(cam.pitch)

    cam.front = glm.v3_unit(glm.Vec3(x, y, z))
    cam.right = glm.v3_unit(glm.v3_cross(cam.front, glm.Vec3(y=1.0)))
    cam.up = glm.v3_unit(glm.v3_cross(cam.right, cam.front))


def camera_view_matrix(cam: Camera) -> glm.Mat4:
    '''Camerea get view matrix'''
    return glm.m4_look_at(
            cam.position,
            glm.v3_add(cam.position, cam.front),
            cam.up)


def camera_projection_matrix(cam: Camera) -> glm.Mat4:
    '''Camera get projection matrix'''
    return glm.m4_projection(cam.fovy, cam.aspect, cam.znear, cam.zfar)

# --- MOUSE
# TODO()


class MouseState(Enum):
    PRESSED = 0
    RELEASED = 1
    HELD = 2
    DEFAULT = 3


@dataclass(eq=False, repr=False, slots=True)
class Mouse:
    state: int = 0xFF


def _mouse_set_current(mouse: Mouse, value: int) -> None:
    mouse.state = (mouse.state & 0xFFFFFF00) | value


def _mouse_set_previous(mouse: Mouse, value: int) -> None:
    mouse.state = (mouse.state & 0xFFFF00FF) | (value << 8)


def _mouse_get_current(mouse: Mouse) -> int:
    return 0xFF & mouse.state


def _mouse_get_previous(mouse: Mouse) -> int:
    return 0xFF & (mouse.state >> 8)


def mouse_state(
        mouse: Mouse,
        button: int,
        glfw_mouse_state: int) -> MouseState:
    '''Mouse button pressed'''
    button = glm.clamp(button, 0, 3)
    glfw_mouse_state = glm.clamp(glfw_mouse_state, 0, 3)

    tmp = _mouse_get_current(mouse)
    _mouse_set_previous(mouse, tmp)
    _mouse_set_current(mouse, glfw_mouse_state)

    result = MouseState.DEFAULT

    if _mouse_get_previous(mouse) == 0:
        if _mouse_get_current(mouse) == 0:
            result = MouseState.DEFAULT
        else:
            # pressed
            result = MouseState.PRESSED
    else:
        if _mouse_get_current(mouse) == 0:
            # released
            result = MouseState.RELEASED
        else:
            # held
            result = MouseState.HELD

    return result


# --- MAIN


def main() -> None:
    ''' '''
    if not glfw.init():
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

        mouse = Mouse()

        while not glwin_should_close(glwin):
            clock_update(clock)

            gl.glClearColor(0.3, 0.2, 0.2, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            # ---

            shader_use(shader)
            vbo_use(vbo)

            model: glm.Mat4 = glm.m4_from_axis(
                                    glm.to_radians(clock.ticks),
                                    glm.Vec3(x=0.5, y=1.0))

            view = camera_view_matrix(camera)
            proj = camera_projection_matrix(camera)

            shader_set_vec3(shader, 'color', glm.Vec3(x=1.0, y=0.5))
            shader_set_m4(shader, 'model', model)
            shader_set_m4(shader, 'view', view)
            shader_set_m4(shader, 'projection', proj)

            state = glwin_mouse_state(glwin, 0)
            if mouse_state(mouse, 0, state) == MouseState.HELD:
                print('pressed')

            # ---
            glfw.swap_buffers(glwin.window)
            glfw.poll_events()

    except GlWindowErr as gl_window_error:
        print(f'ERROR: {gl_window_error}')
        glfw.terminate()

    except glm.Mat4Err as mat4_error:
        print(f'ERROR: {mat4_error}')
        glfw.terminate()

    finally:
        vbo_clean(vbo)
        shader_clean(shader)

        glfw.terminate()

        print('CLOSED')


if __name__ == '__main__':
    main()

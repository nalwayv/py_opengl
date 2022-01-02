'''PY OPENGL
'''
import glfw

from dataclasses import dataclass
from loguru import logger
from OpenGL import GL
from py_opengl import utils
from py_opengl import glm
from py_opengl import clock
from py_opengl import shader
from py_opengl import vbo
from py_opengl import keyboard
from py_opengl import mouse
from py_opengl import camera
from typing import Any

# --- TRANSFORM


@dataclass(eq=False, repr=False, slots=True)
class Transform:
    position: glm.Vec3 = glm.Vec3()
    scale: float = 0.0
    rotation: glm.Quaternion = glm.Quaternion(w=1.0)


def trans_get_matrix(tr: Transform) -> glm.Mat4:
    '''Get mat4 transform'''
    t = glm.m4_create_identity()
    t = t * glm.m4_create_translation(tr.position)
    t = t * glm.qt_to_mat4(tr.rotation)
    t = t * glm.m4_create_scaleF(tr.scale)
    return t


# --- SHAPE


@dataclass(eq=False, repr=False, slots=True)
class Triangle:
    vbo: Any = None
    shader: Any = None

    def __post_init__(self):
        self.vbo = vbo.Vbo()
        self.shader = shader.Shader()


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

    shader.shader_compile(tri.shader, vert_src, frag_src)

    verts = [
            0.5, -0.5, 0.0,
            -0.5, -0.5, 0.0,
            0.0, 0.5, 0.0]

    color = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]

    vbo.vbo_add_data(tri.vbo, verts)
    vbo.vbo_add_data(tri.vbo, color)


def tri_draw(tri: Triangle) -> None:
    '''Render triangle to screen'''
    shader.shader_use(tri.shader)
    vbo.vbo_draw(tri.vbo)


def tri_clean(tri: Triangle) -> None:
    '''Clean up triangle shader and vbo'''
    shader.shader_clean(tri.shader)
    vbo.vbo_clean(tri.vbo)


# --- GL WINDOW


class GlWindowError(Exception):
    '''Custom error for gl window'''

    def __init__(self, msg: str):
        super().__init__(msg)


def cb_window_resize(window, width, height):
    ''' '''
    GL.glViewport(0, 0, width, height)


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

        glwin = GlWindow(width=utils.SCREEN_WIDTH, height=utils.SCREEN_HEIGHT)
        glfw.make_context_current(glwin.window)
        glwin_center_screen_position(glwin)
        glwin_set_window_callback(glwin, cb_window_resize)

        time = clock.Clock()

        tri = Triangle()
        tri_init(tri)

        model = Transform()
        cam = camera.Camera(position=glm.Vec3(z=3.0), aspect=utils.SCREEN_WIDTH/utils.SCREEN_HEIGHT)

        kb = keyboard.Keyboard()
        ms = mouse.Mouse()
        first_move = True
        last_mp = glm.Vec3()
        mouse_sensitivity = 0.2

        while not glwin_should_close(glwin):
            clock.clock_update_time(time)

            GL.glClearColor(0.2, 0.3, 0.3, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # ---

            tri_draw(tri)

            model.rotation = glm.qt_from_axis(clock.ticks, glm.Vec3(x=0.1, y=0.5, z=0.2))
            m = trans_get_matrix(model)
            v = camera.camera_view_matrix(camera)
            p = camera.camera_perspective_matrix(camera)

            shader.shader_set_m4(tri.shader, 'mvp', m * v * p)

            # keyboard
            ks_w = glwin_key_state(glwin, glfw.KEY_W)
            if keyboard.key_state(keyboard, ks_w) == keyboard.KeyState.HELD:
                camera.position = camera.position + glm.v3_scale(camera.front, camera.speed * time.delta)

            ks_s = glwin_key_state(glwin, glfw.KEY_S)
            if keyboard.key_state(keyboard, ks_s) == keyboard.KeyState.HELD:
                camera.position = camera.position - glm.v3_scale(camera.front, camera.speed * time.delta)

            ks_a = glwin_key_state(glwin, glfw.KEY_A)
            if keyboard.key_state(keyboard, ks_a) == keyboard.KeyState.HELD:
                camera.position = camera.position + glm.v3_scale(camera.right, camera.speed * time.delta)

            ks_d = glwin_key_state(glwin, glfw.KEY_D)
            if keyboard.key_state(keyboard, ks_d) == keyboard.KeyState.HELD:
                camera.position = camera.position - glm.v3_scale(camera.right, camera.speed * time.delta)

            ks_space = glwin_key_state(glwin, glfw.KEY_Q)
            if keyboard.key_state(keyboard, ks_space) == keyboard.KeyState.HELD:
                camera.position = camera.position + glm.v3_scale(camera.up, camera.speed * time.delta)

            ks_ls = glwin_key_state(glwin, glfw.KEY_E)
            if keyboard.key_state(keyboard, ks_ls) == keyboard.KeyState.HELD:
                camera.position = camera.position - glm.v3_scale(camera.up, camera.speed * time.delta)

            # mouse
            ms = glwin_mouse_state(glwin, glfw.MOUSE_BUTTON_LEFT)
            current_mp = glwin_mouse_pos(glwin)

            if mouse.mouse_state(mouse, ms) == mouse.MouseState.HELD:
                if first_move:
                    last_mp = current_mp
                    first_move = False
                else:
                    new_mp = current_mp - last_mp
                    last_mp = current_mp

                    camera.yaw -= camera.camera_yaw(new_mp.x * mouse_sensitivity)
                    camera.pitch += camera.camera_pitch(new_mp.y * mouse_sensitivity)

                    camera.camera_update(camera)

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

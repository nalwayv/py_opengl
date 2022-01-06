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
    scale: float = 1.0
    rotation: glm.Quaternion = glm.Quaternion(w=1.0)

    def get_matrix(self) -> glm.Mat4:
        '''Get mat4 transform'''
        return (
            glm.Mat4.create_translation(self.position) *
            self.rotation.to_mat4() * 
            glm.Mat4.create_scaler(glm.Vec3(self.scale, self.scale, self.scale)))


# --- SHAPE


@dataclass(eq=False, repr=False, slots=True)
class Triangle:
    vbo_program: Any= None
    shader_program: Any = None

    def __post_init__(self):
        self.vbo_program = vbo.Vbo()
        self.shader_program = shader.Shader()

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
        self.shader_program.compile(vert_src, frag_src)


        verts = [
                0.5, -0.5, 0.0,
                -0.5, -0.5, 0.0,
                0.0, 0.5, 0.0]

        color = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0]

        self.vbo_program.add_data(verts)
        self.vbo_program.add_data(color)


    def draw(self) -> None:
        '''Render triangle to screen'''
        self.shader_program.use()
        self.vbo_program.draw()


    def clean(self) -> None:
        '''Clean up triangle shader and vbo'''
        self.shader_program.clean()
        self.vbo_program.clean()


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


    def set_window_callback(self, cb_func) -> None:
        ''' '''
        glfw.set_window_size_callback(self.window, cb_func)


    def should_close(self) -> bool:
        '''Close window'''
        return True if glfw.window_should_close(self.window) else False


    def center_screen_position(self) -> None:
        '''Center glwindow to center of screen'''
        video = glfw.get_video_mode(glfw.get_primary_monitor())

        x: float = (video.size.width // 2) - (self.width // 2)
        y: float = (video.size.height // 2) - (self.height // 2)

        glfw.set_window_pos(self.window, x, y)


    def get_mouse_pos(self) -> glm.Vec3:
        '''Get current cursor possition on current window

        Returns
        ---
        Vec3: vec3(x, y, z)
        '''
        cx, cy = glfw.get_cursor_pos(self.window)
        return glm.Vec3(x=cx, y=cy)


    def get_mouse_state(self, button: int) -> tuple[int, int]:
        '''Get glfw mouse button state

        Parameters
        ---
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
        return (button, glfw.get_mouse_button(self.window, button))


    def get_key_state(self, key: int) -> tuple[int, int]:
        '''Get glfw keybutton state

        Parameters
        ---
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
        return (key, glfw.get_key(self.window, key))


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
        glwin.center_screen_position()
        glwin.set_window_callback(cb_window_resize)

        time = clock.Clock()

        tri = Triangle()

        model = Transform()
        cam = camera.Camera(position=glm.Vec3(z=3.0), aspect=utils.SCREEN_WIDTH/utils.SCREEN_HEIGHT)

        kb = keyboard.Keyboard()
        ms = mouse.Mouse()
        first_move = True
        last_mp = glm.Vec3()
        mouse_sensitivity = 0.2

        while not glwin.should_close():
            time.update()

            GL.glClearColor(0.2, 0.3, 0.3, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # ---

            tri.draw()

            model.rotation = glm.Quaternion.from_axis(time.ticks, glm.Vec3(x=1.0, y=0.5))
            m = model.get_matrix()
            v = cam.view_matrix()
            p = cam.perspective_matrix()
            mvp = m*v*p
            tri.shader_program.set_m4('mvp', mvp)

            # keyboard
            if kb.get_state(glwin.get_key_state(glfw.KEY_W)) == keyboard.KeyState.HELD:
                cam.position = cam.position + cam.front * (cam.speed * time.delta)

            if kb.get_state(glwin.get_key_state(glfw.KEY_S)) == keyboard.KeyState.HELD:
                cam.position = cam.position - cam.front * (cam.speed * time.delta)

            if kb.get_state(glwin.get_key_state(glfw.KEY_A)) == keyboard.KeyState.HELD:
                cam.position = cam.position + cam.right * (cam.speed * time.delta)

            if kb.get_state(glwin.get_key_state(glfw.KEY_D)) == keyboard.KeyState.HELD:
                cam.position = cam.position - cam.right * (cam.speed * time.delta)

            if kb.get_state(glwin.get_key_state(glfw.KEY_Q)) == keyboard.KeyState.HELD:
                cam.position = cam.position + cam.up * (cam.speed * time.delta)

            if kb.get_state(glwin.get_key_state(glfw.KEY_E)) == keyboard.KeyState.HELD:
                cam.position = cam.position - cam.up * (cam.speed * time.delta)

            # mouse
   
            current_mp = glwin.get_mouse_pos()
            if ms.get_state(glwin.get_mouse_state(glfw.MOUSE_BUTTON_LEFT)) == mouse.MouseState.HELD:
                if first_move:
                    last_mp = current_mp
                    first_move = False
                else:
                    new_mp = current_mp - last_mp
                    last_mp = current_mp

                    cam.yaw -= camera.Camera.to_yaw(new_mp.x * mouse_sensitivity)
                    cam.pitch += camera.Camera.to_pitch(new_mp.y * mouse_sensitivity)

            cam.update()

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

        tri.clean()

        glfw.terminate()


if __name__ == '__main__':
    main()

"""PY OPENGL
"""
from dataclasses import dataclass
from textwrap import dedent
from typing import Optional

import glfw
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
from py_opengl import transform
from py_opengl import window
from py_opengl import texture


# --- SHAPE


@dataclass(eq=False, repr=False, slots=True)
class Triangle:
    vbo_: Optional[vbo.Vbo] = None
    shader_: Optional[shader.Shader] = None
    texture_: Optional[texture.Texture] = None
    transform_: Optional[transform.Transform] = None

    def __post_init__(self):
        self.vbo_ = vbo.Vbo(length=9)
        self.shader_ = shader.Shader()
        self.texture_ = texture.Texture()
        self.transform_ = transform.Transform()

        texture_src: str = 'wall.jpg'

        vert_src: str = dedent("""
        # version 430 core

        layout (location = 0) in vec3 a_pos;
        layout (location = 1) in vec3 a_col;
        layout (location = 2) in vec3 a_texture;

        out vec3 b_col;
        out vec2 b_texture;

        uniform mat4 mvp;

        void main(void)
        {
            b_col = a_col;
            b_texture = vec2(a_texture.x, a_texture.y);

            gl_Position = mvp * vec4(a_pos, 1.0);
        }
        """)

        frag_src: str = dedent("""
        # version 430 core

        in vec3 b_col;
        in vec2 b_texture;

        out vec4 c_col;

        uniform sampler2D c_texture;

        void main(void)
        {
            c_col = texture(c_texture, b_texture) * vec4(b_col, 1.0);
        }
        """)

        verts: list[float] = [
            0.5, -0.5, 0.0,
            -0.5, -0.5, 0.0,
            0.0, 0.5, 0.0
        ]

        color: list[float] = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        tex: list[float] = [
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.5, 1.0, 0.0,
        ]

        self.texture_.compile(texture_src)
        self.shader_.compile(vert_src, frag_src)
        self.vbo_.add_data(verts)
        self.vbo_.add_data(color)
        self.vbo_.add_data(tex)

    def draw(self) -> None:
        """Draw to screen
        """
        self.texture_.use()
        self.shader_.use()
        self.vbo_.draw()

    def clean(self) -> None:
        """Clean up
        """
        self.shader_.clean()
        self.texture_.clean()
        self.vbo_.clean()


@dataclass(eq=False, repr=False, slots=True)
class Cube:
    vbo_: Optional[vbo.Vbo] = None
    shader_: Optional[shader.Shader] = None
    texture_: Optional[texture.Texture] = None
    transform_: Optional[transform.Transform] = None

    def __post_init__(self):
        self.vbo_ = vbo.Vbo(length=36)
        self.shader_ = shader.Shader()
        self.texture_ = texture.Texture()
        self.transform_ = transform.Transform()

        texture_src: str = 'grid512.bmp'

        vert_src: str = dedent("""
        # version 430 core

        layout (location = 0) in vec3 a_pos;
        layout (location = 1) in vec3 a_col;
        layout (location = 2) in vec3 a_texture;

        out vec3 b_col;
        out vec2 b_texture;

        uniform mat4 mvp;

        void main(void)
        {
            b_col = a_col;
            b_texture = vec2(a_texture.x, a_texture.y);

            gl_Position = mvp * vec4(a_pos, 1.0);
        }
        """)

        frag_src: str = dedent("""
        # version 430 core

        in vec3 b_col;
        in vec2 b_texture;

        out vec4 c_col;

        uniform sampler2D c_texture;

        void main(void)
        {
            c_col = texture(c_texture, b_texture) * vec4(b_col, 1.0);
        }
        """)

        verts: list[float] = [
            0.5, 0.5, 0.5,
            -0.5, 0.5, 0.5,
            -0.5, -0.5, 0.5,
            -0.5, -0.5, 0.5,
            0.5, -0.5, 0.5,
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
            0.5, -0.5, 0.5,
            0.5, -0.5, -0.5,
            0.5, -0.5, -0.5,
            0.5, 0.5, -0.5,
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
            0.5, 0.5, -0.5,
            -0.5, 0.5, -0.5,
            -0.5, 0.5, -0.5,
            -0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
            -0.5, 0.5, 0.5,
            -0.5, 0.5, -0.5,
            -0.5, -0.5, -0.5,
            -0.5, -0.5, -0.5,
            -0.5, -0.5, 0.5,
            -0.5, 0.5, 0.5,
            -0.5, -0.5, -0.5,
            0.5, -0.5, -0.5,
            0.5, -0.5, 0.5,
            0.5, -0.5, 0.5,
            -0.5, -0.5, 0.5,
            -0.5, -0.5, -0.5,
            0.5, -0.5, -0.5,
            -0.5, -0.5, -0.5,
            -0.5, 0.5, -0.5,
            -0.5, 0.5, -0.5,
            0.5, 0.5, -0.5,
            0.5, -0.5, -0.5
        ]

        color: list[float] = [
            1.0, 1.0, 1.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0
        ]

        tex: list[float] = [
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]

        self.texture_.compile(texture_src)
        self.shader_.compile(vert_src, frag_src)
        self.vbo_.add_data(verts)
        self.vbo_.add_data(color)
        self.vbo_.add_data(tex)

    def draw(self) -> None:
        """Draw to screen
        """
        self.texture_.use()
        self.shader_.use()
        self.vbo_.draw()

    def clean(self) -> None:
        """Clean up
        """
        self.texture_.clean()
        self.shader_.clean()
        self.vbo_.clean()


# --- CALLBACKS


def cb_window_resize(window, width, height):
    """Window callback resize function

    Parameters
    ---
    window : GLFWwindow*
        glfw window
    width : float
        its width
    height : float
        its height
    """
    GL.glViewport(0, 0, width, height)


# --- MAIN

def main() -> None:
    """Main
    """
    if not glfw.init():
        logger.error('failed to init glfw')
        return

    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)

        glwin = window.GlWindow(
            width=utils.SCREEN_WIDTH,
            height=utils.SCREEN_HEIGHT
        )
        glfw.make_context_current(glwin.window)
        glwin.center_screen_position()
        glwin.set_window_resize_callback(cb_window_resize)

        time = clock.Clock()
        cam = camera.Camera(
            position=glm.Vec3(z=3.0),
            aspect=utils.SCREEN_WIDTH/utils.SCREEN_HEIGHT
        )

        kb = keyboard.Keyboard()
        ms = mouse.Mouse()
        first_move = True
        last_mp = glm.Vec3()

        shape = Cube()

        while not glwin.should_close():
            time.update()

            GL.glClearColor(0.2, 0.3, 0.3, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_CULL_FACE)

            # ---

            shape.draw()

            shape.transform_.rotation = glm.Quaternion.from_axis(
                time.ticks,
                glm.Vec3(x=1.0, y=0.5)
            )
            m: glm.Mat4 = shape.transform_.get_matrix()
            v: glm.Mat4 = cam.view_matrix()
            p: glm.Mat4 = cam.perspective_matrix()
            mvp: glm.Mat4 = m * v * p
            shape.shader_.set_m4('mvp', mvp)

            # keyboard
            if kb.is_key_held(glwin.get_key_state(glfw.KEY_W)):
                cam.move_by(camera.CameraDirection.IN, 1.4, time.delta)

            if kb.is_key_held(glwin.get_key_state(glfw.KEY_S)):
                cam.move_by(camera.CameraDirection.OUT, 1.4, time.delta)

            if kb.is_key_held(glwin.get_key_state(glfw.KEY_A)):
                cam.move_by(camera.CameraDirection.LEFT, 1.4, time.delta)

            if kb.is_key_held(glwin.get_key_state(glfw.KEY_D)):
                cam.move_by(camera.CameraDirection.RIGHT, 1.4, time.delta)

            if kb.is_key_held(glwin.get_key_state(glfw.KEY_E)):
                cam.move_by(camera.CameraDirection.UP, 1.4, time.delta)

            if kb.is_key_held(glwin.get_key_state(glfw.KEY_Q)):
                cam.move_by(camera.CameraDirection.DOWN, 1.4, time.delta)

            # mouse
            if ms.is_button_held(
                    glwin.get_mouse_state(glfw.MOUSE_BUTTON_LEFT)
            ):
                if first_move:
                    mx, my = glwin.get_mouse_pos()
                    last_mp.x = mx
                    last_mp.y = my
                    first_move = False
                else:
                    mx, my = glwin.get_mouse_pos()
                    new_mp = glm.Vec3(x=mx, y=my) - last_mp
                    last_mp.x = mx
                    last_mp.y = my

                    cam.rotate_by(camera.CameraRotation.YAW, new_mp.x, 0.2)
                    cam.rotate_by(camera.CameraRotation.PITCH, new_mp.y, 0.2)

            cam.update()

            # ---
            glfw.swap_buffers(glwin.window)
            glfw.poll_events()

    except (glm.Vec3Error, glm.Mat4Error, glm.QuatError) as math_err:
        logger.error(f'MATH ERROR: {math_err}')

    except texture.TextureError as texture_err:
        logger.error(f'TEXTURE ERROR: {texture_err}')

    except camera.CameraError as camera_err:
        logger.error(f'CAMERA ERROR: {camera_err}')

    except window.GlWindowError as window_err:
        logger.error(f'WINDOW ERROR: {window_err}')

    finally:
        logger.debug('CLOSED')
        shape.clean()
        glfw.terminate()


if __name__ == '__main__':
    main()


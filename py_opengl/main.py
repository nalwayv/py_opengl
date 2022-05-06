"""Main
"""
from dataclasses import dataclass
# from abc import ABC, abstractmethod
from typing import Optional

import glfw
from loguru import logger
from OpenGL import GL

from py_opengl import utils
from py_opengl import maths
from py_opengl import clock
# from py_opengl import transform
from py_opengl import shader
# from py_opengl import texture
from py_opengl import keyboard
from py_opengl import mouse
from py_opengl import camera
from py_opengl import window
from py_opengl import color
from py_opengl import mesh


# ---


@dataclass(eq= False, repr= False, slots= True)
class Shape:
    _mesh: Optional[mesh.Mesh]= None
    _shader: Optional[shader.Shader]= None

    def __post_init__(self):
        verts: list[mesh.Vertex]= [
            mesh.Vertex(maths.Vec3( 0.5,  0.5, 0.5), maths.Vec3(1.0, 1.0, 1.0)),
            mesh.Vertex(maths.Vec3(-0.5,  0.5, 0.5), maths.Vec3(1.0, 1.0, 0.0)),
            mesh.Vertex(maths.Vec3(-0.5, -0.5, 0.5), maths.Vec3(1.0, 0.0, 0.0)),
            mesh.Vertex(maths.Vec3( 0.5, -0.5, 0.5), maths.Vec3(1.0, 0.0, 1.0)),

            mesh.Vertex(maths.Vec3( 0.5,  0.5,  0.5), maths.Vec3(1.0, 1.0, 1.0)),
            mesh.Vertex(maths.Vec3( 0.5, -0.5,  0.5), maths.Vec3(1.0, 0.0, 1.0)),
            mesh.Vertex(maths.Vec3( 0.5, -0.5, -0.5), maths.Vec3(0.0, 0.0, 1.0)),
            mesh.Vertex(maths.Vec3( 0.5,  0.5, -0.5), maths.Vec3(0.0, 1.0, 1.0)),

            mesh.Vertex(maths.Vec3( 0.5,  0.5,  0.5), maths.Vec3(1.0, 1.0, 1.0)),
            mesh.Vertex(maths.Vec3( 0.5,  0.5, -0.5), maths.Vec3(0.0, 1.0, 1.0)),
            mesh.Vertex(maths.Vec3(-0.5,  0.5, -0.5), maths.Vec3(0.0, 1.0, 0.0)),
            mesh.Vertex(maths.Vec3(-0.5,  0.5,  0.5), maths.Vec3(1.0, 1.0, 0.0)),

            mesh.Vertex(maths.Vec3(-0.5,  0.5,  0.5), maths.Vec3(1.0, 1.0, 0.0)),
            mesh.Vertex(maths.Vec3(-0.5,  0.5, -0.5), maths.Vec3(0.0, 1.0, 0.0)),
            mesh.Vertex(maths.Vec3(-0.5, -0.5, -0.5), maths.Vec3(0.0, 0.0, 0.0)),
            mesh.Vertex(maths.Vec3(-0.5, -0.5,  0.5), maths.Vec3(1.0, 0.0, 0.0)),

            mesh.Vertex(maths.Vec3(-0.5, -0.5, -0.5), maths.Vec3(0.0, 0.0, 0.0)),
            mesh.Vertex(maths.Vec3( 0.5, -0.5, -0.5), maths.Vec3(0.0, 0.0, 1.0)),
            mesh.Vertex(maths.Vec3( 0.5, -0.5,  0.5), maths.Vec3(1.0, 0.0, 1.0)),
            mesh.Vertex(maths.Vec3(-0.5, -0.5,  0.5), maths.Vec3(1.0, 0.0, 0.0)),

            mesh.Vertex(maths.Vec3( 0.5, -0.5, -0.5), maths.Vec3(0.0, 0.0, 1.0)),
            mesh.Vertex(maths.Vec3(-0.5, -0.5, -0.5), maths.Vec3(0.0, 0.0, 0.0)),
            mesh.Vertex(maths.Vec3(-0.5,  0.5, -0.5), maths.Vec3(0.0, 1.0, 0.0)),
            mesh.Vertex(maths.Vec3( 0.5,  0.5, -0.5), maths.Vec3(0.0, 1.0, 1.0))
        ]

        indices: list[int]= [
             0,  1,  2,
             2,  3,  0,
             4,  5,  6,
             6,  7,  4,
             8,  9, 10,
            10, 11,  8,
            12, 13, 14,
            14, 15, 12,
            16, 17, 18,
            18, 19, 16,
            20, 21, 22,
            22, 23, 20
        ]

        self._mesh = mesh.Mesh(vertices= verts, indices= indices)
        self._mesh.setup()
        self._shader= shader.Shader(vshader= 'debug_shader.vert', fshader= 'debug_shader.frag')
        self._shader.setup()
    
    def draw(self) -> None:
        self._shader.use()
        self._mesh.use()

    def clean(self) -> None:
        self._shader.clean()
        self._mesh.clean()


# --- CALLBACKS


def cb_window_resize(window, width, height):
    """Window callback resize function

    Parameters
    ---
    window : GLFWwindow*

    width : float
    
    height : float
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

        glwin= window.GlWindow(
            width= utils.SCREEN_WIDTH,
            height= utils.SCREEN_HEIGHT
        )

        glfw.make_context_current(glwin.window)
        glwin.center_screen_position()
        glwin.set_window_resize_callback(cb_window_resize)


        bg_col= color.Color.from_rgba(75, 75, 75, 255)
        
        time= clock.Clock()

        cam= camera.Camera(
            position= maths.Vec3(z=3.0),
            aspect= utils.SCREEN_WIDTH/utils.SCREEN_HEIGHT
        )

        kb: keyboard.Keyboard= keyboard.Keyboard()

        ms: mouse.Mouse= mouse.Mouse()
        first_move: bool= True
        last_mp:maths.Vec3= maths.Vec3.zero()

        shape= Shape()

        while not glwin.should_close():
            time.update()

            GL.glClearColor(*bg_col.get_data_norm())
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_CULL_FACE)

            # ---

            # shape
            shape.draw()

            # shape.transform_.rotated_xyz(maths.Vec3(x= 25.0, y= 15.0) * time.delta)

            shape._shader.set_mat4('m_matrix', maths.Mat4.identity())
            shape._shader.set_mat4('v_matrix', cam.view_matrix())
            shape._shader.set_mat4('p_matrix', cam.projection_matrix())

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
            if ms.is_button_held(glwin.get_mouse_state(glfw.MOUSE_BUTTON_LEFT)):
                if first_move:
                    mx, my= glwin.get_mouse_pos()
                    last_mp.x= mx
                    last_mp.y= my
                    first_move = False
                else:
                    mx, my= glwin.get_mouse_pos()
                    new_mp= maths.Vec3(x=mx, y=my) - last_mp
                    last_mp.x= mx
                    last_mp.y= my
                    
                    cam.rotate_by(camera.CameraRotation.YAW, new_mp.x, 0.2)
                    cam.rotate_by(camera.CameraRotation.PITCH, new_mp.y, 0.2)

            # ---
            glfw.poll_events()
            glfw.swap_buffers(glwin.window)

    except Exception as err:
        logger.error(f"ERROR: {err}")

    finally:
        logger.debug('CLOSED')
        shape.clean()
        glfw.terminate()


if __name__ == '__main__':
    main()

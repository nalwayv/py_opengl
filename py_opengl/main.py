"""Main
"""
from dataclasses import dataclass, field
# from abc import ABC, abstractmethod
from typing import Optional

import glfw
from loguru import logger
from OpenGL import GL

from py_opengl import utils
from py_opengl import maths
from py_opengl import transform
from py_opengl import clock
from py_opengl import shader
from py_opengl import buffer
from py_opengl import keyboard
from py_opengl import mouse
from py_opengl import camera
from py_opengl import window
from py_opengl import texture
from py_opengl import color


# --- SHAPE

@dataclass(eq= False, repr= False, slots= True)
class Cube:
    size: maths.Vec3= maths.Vec3(1.0, 1.0, 1.0)

    shader_: Optional[shader.Shader]= None
    texture_: Optional[texture.Texture]= None
    transform_: Optional[transform.Transform]= None

    verts: list[float]= field(default_factory=list)
    colors: list[float]= field(default_factory=list)
    tex_coords: list[float]= field(default_factory=list)
    indices: list[int]= field(default_factory=list)

    cube_id: Optional[buffer.Vao]= None
    vert_vbo: Optional[buffer.Vbo]= None
    color_vbo: Optional[buffer.Vbo]= None
    texture_vbo: Optional[buffer.Vbo]= None
    indices_ibo: Optional[buffer.Ibo]= None

    def __post_init__(self):

        hw: float= self.size.x * 0.5
        hh: float= self.size.y * 0.5
        hd: float= self.size.z * 0.5

        self.verts= [
             hw, hh, hd, 
            -hw, hh, hd,
            -hw,-hh, hd, 
             hw,-hh, hd,
             hw, hh, hd, 
             hw,-hh, hd, 
             hw,-hh,-hd, 
             hw, hh,-hd,
             hw, hh, hd,
             hw, hh,-hd,
            -hw, hh,-hd,
            -hw, hh, hd,
            -hw, hh, hd,
            -hw, hh,-hd,
            -hw,-hh,-hd,
            -hw,-hh, hd,
            -hw,-hh,-hd,
             hw,-hh,-hd,
             hw,-hh, hd,
            -hw,-hh, hd,
             hw,-hh,-hd,
            -hw,-hh,-hd,
            -hw, hh,-hd,
             hw, hh,-hd
        ]

        self.colors = [
            1.0, 1.0, 1.0, 
            1.0, 1.0, 0.0, 
            1.0, 0.0, 0.0, 
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 
            1.0, 0.0, 1.0, 
            0.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 
            0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 
            1.0, 0.0, 1.0, 
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 
            0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 
            0.0, 1.0, 1.0
        ]

        self.tex_coords= [
            1.0, 0.0,
            0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
            1.0, 0.0,
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            0.0, 1.0,
            1.0, 1.0,
            1.0, 0.0,
            0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            1.0, 0.0,
            0.0, 0.0
        ]
   
        self.indices= [
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

        self.shader_= shader.Shader() 
        self.texture_= texture.Texture()
        self.transform_= transform.Transform()

        self.texture_.compile('grid512.bmp')
        self.shader_.compile('shader.vert', 'shader.frag')
    
        self.cube_id= buffer.Vao()
        self.vert_vbo= buffer.Vbo(index=0)
        self.color_vbo= buffer.Vbo(index=1)
        self.texture_vbo= buffer.Vbo(index=2, components=2)
        self.indices_ibo= buffer.Ibo()

        self.vert_vbo.setup(self.verts)
        self.color_vbo.setup(self.colors)
        self.texture_vbo.setup(self.tex_coords)
        self.indices_ibo.setup(self.indices)
    

    def draw(self) -> None:
        self.texture_.use()
        self.shader_.use()

        GL.glDrawElements(GL.GL_TRIANGLES, self.indices_ibo.length, GL.GL_UNSIGNED_INT, utils.C_VOID_POINTER)

    def clean(self) -> None:
        self.texture_.clean()
        self.shader_.clean()
        
        self.cube_id.clean()
        self.vert_vbo.clean()
        self.color_vbo.clean()
        self.texture_vbo.clean()
        self.indices_ibo.clean()


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

        shape: Cube= Cube()

        while not glwin.should_close():
            time.update()

            GL.glClearColor(*bg_col.get_data_norm())
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_CULL_FACE)

            # ---

            # shape
            shape.draw()

            shape.transform_.rotated_xyz(maths.Vec3(x= 25.0, y= 15.0) * time.delta)

            shape.shader_.set_mat4('m_matrix', shape.transform_.model_matrix())
            shape.shader_.set_mat4('v_matrix', cam.view_matrix())
            shape.shader_.set_mat4('p_matrix', cam.projection_matrix())

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

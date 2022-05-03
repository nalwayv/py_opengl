"""Main
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional

import glfw
from loguru import logger
from OpenGL import GL

from py_opengl import utils
from py_opengl import maths
from py_opengl import transform
from py_opengl import clock
from py_opengl import shader
from py_opengl import vbo
from py_opengl import keyboard
from py_opengl import mouse
from py_opengl import camera
from py_opengl import window
from py_opengl import texture
from py_opengl import color


# --- SHAPE


class IObject(ABC):
    @abstractmethod
    def draw() -> None:
        """Draw to screen"""

    @abstractmethod
    def clean() -> None:
        """Clean up data used by object"""

@dataclass(eq= False, repr= False, slots= True)
class Triangle(IObject):
    vbo_: Optional[vbo.Vbo]= None
    shader_: Optional[shader.Shader]= None
    texture_: Optional[texture.Texture]= None
    transform_: Optional[transform.Transform]= None

    def __post_init__(self):
        verts_pts: list[maths.Pt3]=  [
            maths.Pt3( 0.5, -0.5, 0.0),
            maths.Pt3(-0.5, -0.5, 0.0),
            maths.Pt3( 0.0,  0.5, 0.0)
        ]

        color_pts: list[maths.Pt3]=  [
            maths.Pt3(1.0, 0.0, 0.0),
            maths.Pt3(0.0, 1.0, 0.0),
            maths.Pt3(0.0, 0.0, 1.0)
        ]

        tex_coords_pts: list[maths.Pt2]=  [
            maths.Pt2(0.0, 0.0),
            maths.Pt2(1.0, 0.0),
            maths.Pt2(0.5, 1.0),
        ]

        indices_pts: list[maths.Pt3Int]= [
            maths.Pt3Int(0, 1, 2),
            maths.Pt3Int(3, 4, 5),
            maths.Pt3Int(6, 7, 8)
        ]

        self.vbo_= vbo.Vbo(length=len(indices_pts) * 3)
        self.shader_= shader.Shader()
        self.texture_= texture.Texture()
        self.transform_= transform.Transform()

        self.texture_.compile('grid512.bmp')
        self.shader_.compile('shader.vert', 'shader.frag')
        self.vbo_.setup(verts_pts, color_pts, tex_coords_pts, indices_pts)

    def draw(self) -> None:
        self.texture_.use()
        self.shader_.use()
        self.vbo_.use(vbo.VboDrawMode.TRIANGLES)

    def clean(self) -> None:
        self.shader_.clean()
        self.texture_.clean()
        self.vbo_.clean()


@dataclass(eq= False, repr= False, slots= True)
class Cube(IObject):
    vbo_: Optional[vbo.Vbo]= None
    shader_: Optional[shader.Shader]= None
    texture_: Optional[texture.Texture]= None
    transform_: Optional[transform.Transform]= None
    scale_: maths.Vec3= maths.Vec3(1.0, 1.0, 1.0)

    verts: list[maths.Pt3]= field(default_factory=list)
    colors: list[maths.Pt3]= field(default_factory=list)
    tex_coords: list[maths.Pt2]= field(default_factory=list)
    indices: list[maths.Pt3Int]= field(default_factory=list)

    def __post_init__(self):

        hw: float= self.scale_.x * 0.5
        hh: float= self.scale_.y * 0.5
        hd: float= self.scale_.z * 0.5

        self.verts: list[maths.Pt3]=  [
            maths.Pt3( hw, hh, hd), 
            maths.Pt3(-hw, hh, hd),
            maths.Pt3(-hw,-hh, hd), 
            maths.Pt3( hw,-hh, hd),

            maths.Pt3( hw, hh, hd), 
            maths.Pt3( hw,-hh, hd), 
            maths.Pt3( hw,-hh,-hd), 
            maths.Pt3( hw, hh,-hd),

            maths.Pt3( hw, hh, hd),
            maths.Pt3( hw, hh,-hd),
            maths.Pt3(-hw, hh,-hd),
            maths.Pt3(-hw, hh, hd),

            maths.Pt3(-hw, hh, hd),
            maths.Pt3(-hw, hh,-hd),
            maths.Pt3(-hw,-hh,-hd),
            maths.Pt3(-hw,-hh, hd),

            maths.Pt3(-hw,-hh,-hd),
            maths.Pt3( hw,-hh,-hd),
            maths.Pt3( hw,-hh, hd),
            maths.Pt3(-hw,-hh, hd),

            maths.Pt3( hw,-hh,-hd),
            maths.Pt3(-hw,-hh,-hd),
            maths.Pt3(-hw, hh,-hd),
            maths.Pt3( hw, hh,-hd)
        ]

        self.colors: list[maths.Pt3]=  [
            maths.Pt3(1.0, 1.0, 1.0), 
            maths.Pt3(1.0, 1.0, 0.0), 
            maths.Pt3(1.0, 0.0, 0.0), 
            maths.Pt3(1.0, 0.0, 1.0),

            maths.Pt3(1.0, 1.0, 1.0), 
            maths.Pt3(1.0, 0.0, 1.0), 
            maths.Pt3(0.0, 0.0, 1.0),
            maths.Pt3(0.0, 1.0, 1.0),

            maths.Pt3(1.0, 1.0, 1.0),
            maths.Pt3(0.0, 1.0, 1.0),
            maths.Pt3(0.0, 1.0, 0.0),
            maths.Pt3(1.0, 1.0, 0.0),

            maths.Pt3(1.0, 1.0, 0.0), 
            maths.Pt3(0.0, 1.0, 0.0), 
            maths.Pt3(0.0, 0.0, 0.0), 
            maths.Pt3(1.0, 0.0, 0.0),

            maths.Pt3(0.0, 0.0, 0.0), 
            maths.Pt3(0.0, 0.0, 1.0), 
            maths.Pt3(1.0, 0.0, 1.0), 
            maths.Pt3(1.0, 0.0, 0.0),

            maths.Pt3(0.0, 0.0, 1.0), 
            maths.Pt3(0.0, 0.0, 0.0), 
            maths.Pt3(0.0, 1.0, 0.0), 
            maths.Pt3(0.0, 1.0, 1.0)
        ]

        self.tex_coords: list[maths.Pt2]=  [
            maths.Pt2(1.0, 0.0),
            maths.Pt2(0.0, 0.0),
            maths.Pt2(0.0, 1.0),
            maths.Pt2(1.0, 1.0),

            maths.Pt2(0.0, 0.0),
            maths.Pt2(0.0, 1.0),
            maths.Pt2(1.0, 1.0),
            maths.Pt2(1.0, 0.0),

            maths.Pt2(1.0, 1.0),
            maths.Pt2(1.0, 0.0),
            maths.Pt2(0.0, 0.0),
            maths.Pt2(0.0, 1.0),

            maths.Pt2(1.0, 0.0),
            maths.Pt2(0.0, 0.0),
            maths.Pt2(0.0, 1.0),
            maths.Pt2(1.0, 1.0),

            maths.Pt2(0.0, 1.0),
            maths.Pt2(1.0, 1.0),
            maths.Pt2(1.0, 0.0),
            maths.Pt2(0.0, 0.0),

            maths.Pt2(0.0, 1.0),
            maths.Pt2(1.0, 1.0),
            maths.Pt2(1.0, 0.0),
            maths.Pt2(0.0, 0.0)
        ]
   
        self.indices: list[maths.Pt3Int]= [
            maths.Pt3Int( 0,  1,  2),
            maths.Pt3Int( 2,  3,  0),
            maths.Pt3Int( 4,  5,  6),
            maths.Pt3Int( 6,  7,  4),
            maths.Pt3Int( 8,  9, 10),
            maths.Pt3Int(10, 11,  8),
            maths.Pt3Int(12, 13, 14),
            maths.Pt3Int(14, 15, 12),
            maths.Pt3Int(16, 17, 18),
            maths.Pt3Int(18, 19, 16),
            maths.Pt3Int(20, 21, 22),
            maths.Pt3Int(22, 23, 20)
        ]

        self.vbo_= vbo.Vbo(length=len(self.indices) * 3)
        self.shader_= shader.Shader() 
        self.texture_= texture.Texture()
        self.transform_= transform.Transform()

        self.texture_.compile('grid512.bmp')
        self.shader_.compile('shader.vert', 'shader.frag')
        self.vbo_.setup(
            self.verts,
            self.colors,
            self.tex_coords,
            self.indices
        )

    def draw(self) -> None:
        self.texture_.use()
        self.shader_.use()
        self.vbo_.use(vbo.VboDrawMode.TRIANGLES)

    def clean(self) -> None:
        self.texture_.clean()
        self.shader_.clean()
        self.vbo_.clean()



# --- FUNCS


# def aabb_from_verts(t: transform.Transform, verts: list[maths.Vec3]):
#     if not verts:
#         return geometry.AABB3()

#     pt= t.get_transformed(maths.Vec3.create_from_pt(verts[0]))

#     min_x: float= pt.x
#     max_x: float= pt.x
#     min_y: float= pt.y
#     max_y: float= pt.y
#     min_z: float= pt.z
#     max_z: float= pt.z

#     for vert in verts[1:]:
#         tvert: maths.Vec3= t.get_transformed(maths.Vec3.create_from_pt(vert))

#         min_x= maths.minf(min_x, tvert.x)
#         max_x= maths.maxf(max_x, tvert.x)
#         min_y= maths.minf(min_y, tvert.y)
#         max_y= maths.maxf(max_y, tvert.y)
#         min_z= maths.minf(min_z, tvert.z)
#         max_z= maths.maxf(max_z, tvert.z)

#     return geometry.AABB3.from_min_max(
#         maths.Vec3(min_x, min_y, min_z),
#         maths.Vec3(max_x, max_y, max_z)
#     )


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

            m: maths.Mat4= shape.transform_.model_matrix()
            v: maths.Mat4= cam.view_matrix()
            p: maths.Mat4= cam.projection_matrix()
            
            shape.shader_.set_m4('mvp', m * v * p)

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

"""Main
"""
from dataclasses import dataclass

import glfw
from loguru import logger
from OpenGL import GL

from py_opengl import utils
from py_opengl import maths
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



@dataclass(eq= False, repr= False, slots= True)
class Triangle:
    vbo_: vbo.Vbo|None= None
    shader_: shader.Shader|None= None
    texture_: texture.Texture|None= None
    transform_: maths.Transform|None= None

    def __post_init__(self):
        self.vbo_= vbo.Vbo(length=9)
        self.shader_= shader.Shader()
        self.texture_= texture.Texture()
        self.transform_= maths.Transform()

        texture_src: str= 'wall.jpg'
        vert_src: str= 'shader.vert'
        frag_src: str= 'shader.frag'

        verts: list[float]=  [
            0.5, -0.5, 0.0,
            -0.5, -0.5, 0.0,
            0.0, 0.5, 0.0
        ]

        color: list[float]=  [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        tex_coords: list[float]=  [
            0.0, 0.0,
            1.0, 0.0,
            0.5, 1.0,
        ]

        indices: list[int]= [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8
        ]

        self.texture_.compile(texture_src)
        self.shader_.compile(vert_src, frag_src)
        self.vbo_.setup(verts, color, tex_coords, indices)

    def draw(self) -> None:
        """Draw to screen
        """
        self.texture_.use()
        self.shader_.use()
        self.vbo_.use(vbo.VboDrawMode.TRIANGLES)

    def clean(self) -> None:
        """Clean up
        """
        self.shader_.clean()
        self.texture_.clean()
        self.vbo_.clean()


@dataclass(eq= False, repr= False, slots= True)
class Cube:
    vbo_: vbo.Vbo|None= None
    shader_: shader.Shader|None= None
    texture_: texture.Texture|None= None
    transform_: maths.Transform|None= None
    size: maths.Vec3 = maths.Vec3(1.0, 1.0, 1.0)

    def __post_init__(self):
        self.vbo_= vbo.Vbo(length=36)
        self.shader_= shader.Shader()
        self.texture_= texture.Texture()
        self.transform_= maths.Transform()

        self.texture_.compile('grid512.bmp')
        self.shader_.compile('shader.vert', 'shader.frag')

        hw: float= self.size.x * 0.5
        hh: float= self.size.y * 0.5
        hd: float= self.size.z * 0.5

        verts: list[float]=  [
             hw, hh, hd,  -hw, hh, hd,  -hw,-hh, hd,  hw,-hh, hd,
             hw, hh, hd,   hw,-hh, hd,   hw,-hh,-hd,  hw, hh,-hd,
             hw, hh, hd,   hw, hh,-hd,  -hw, hh,-hd, -hw, hh, hd,
            -hw, hh, hd,  -hw, hh,-hd,  -hw,-hh,-hd, -hw,-hh, hd,
            -hw,-hh,-hd,   hw,-hh,-hd,   hw,-hh, hd, -hw,-hh, hd,
             hw,-hh,-hd,  -hw,-hh,-hd,  -hw, hh,-hd,  hw, hh,-hd
        ]

        color: list[float]=  [
            1.0, 1.0, 1.0,   1.0, 1.0, 0.0,   1.0, 0.0, 0.0,   1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,   1.0, 0.0, 1.0,   0.0, 0.0, 1.0,   0.0, 1.0, 1.0,
            1.0, 1.0, 1.0,   0.0, 1.0, 1.0,   0.0, 1.0, 0.0,   1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,   0.0, 1.0, 0.0,   0.0, 0.0, 0.0,   1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,   0.0, 0.0, 1.0,   1.0, 0.0, 1.0,   1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,   0.0, 0.0, 0.0,   0.0, 1.0, 0.0,   0.0, 1.0, 1.0
        ]

        tex_coords: list[float]=  [
            1.0, 0.0,   0.0, 0.0,   0.0, 1.0,   1.0, 1.0,
            0.0, 0.0,   0.0, 1.0,   1.0, 1.0,   1.0, 0.0,
            1.0, 1.0,   1.0, 0.0,   0.0, 0.0,   0.0, 1.0,
            1.0, 0.0,   0.0, 0.0,   0.0, 1.0,   1.0, 1.0,
            0.0, 1.0,   1.0, 1.0,   1.0, 0.0,   0.0, 0.0,
            0.0, 1.0,   1.0, 1.0,   1.0, 0.0,   0.0, 0.0
        ]
   
        indices: list[int]= [
             0,  1,  2,    2,  3,  0,
             4,  5,  6,    6,  7,  4,
             8,  9, 10,   10, 11,  8,
            12, 13, 14,   14, 15, 12,
            16, 17, 18,   18, 19, 16,
            20, 21, 22,   22, 23, 20
        ]

        self.vbo_.setup(verts, color, tex_coords, indices)

    def draw(self) -> None:
        """Draw to screen
        """
        self.texture_.use()
        self.shader_.use()
        self.vbo_.use(vbo.VboDrawMode.TRIANGLES)

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
        offset: float= 1.0
        speed: float= 1.0
        ang: float= 0.0

        while not glwin.should_close():
            time.update()

            GL.glClearColor(*bg_col.get_data_norm())
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_CULL_FACE)

            # ---

            shape.draw()

            shape.transform_.angle = float(time.ticks)
            shape.transform_.axis = maths.Vec3(1.0, 0.5, 0.8)

            shape.transform_.position= maths.Vec3(
                x= maths.sin(ang) * offset,
                y= maths.cos(ang) * offset
            )
            ang += (speed * time.delta)

            m: maths.Mat4= shape.transform_.get_matrix()
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

            cam.update()

            # ---
            glfw.poll_events()
            glfw.swap_buffers(glwin.window)

    except (maths.Vec3Error, maths.Mat4Error, maths.QuatError) as math_err:
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

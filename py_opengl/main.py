"""Main
"""
import glfw
from loguru import logger
from OpenGL import GL

from py_opengl import utils
from py_opengl import maths
from py_opengl import clock
from py_opengl import shader
from py_opengl import keyboard
from py_opengl import mouse
from py_opengl import camera
from py_opengl import window
from py_opengl import color
from py_opengl import mesh
from py_opengl import transform
from py_opengl import geometry


# ---


class CubeModel():

    __slots__= ('_mesh', '_shader','_transform')

    def __init__(self, scale: float) -> None:
        self._mesh= mesh.CubeMesh(maths.Vec3.create_from_value(scale))
        self._transform= transform.Transform()
    
    def set_position(self, v3: maths.Vec3) -> None:
        self._transform.origin.set_from(v3)

    def translate(self, v3: maths.Vec3) -> None:
        self._transform.translated(v3)

    def rotate(self, v3: maths.Vec3) -> None:
        self._transform.rotated_xyz(v3)
    
    def position(self) -> maths.Vec3:
        self._transform.origin

    def compute(self) -> geometry.AABB3:
        return self._mesh.compute_aabb(self._transform)

    def draw(self, _shader: shader.Shader,  cam: camera.Camera) -> None:
        _shader.use()
        _shader.set_mat4('m_matrix', self._transform.model_matrix())
        _shader.set_mat4('v_matrix', cam.view_matrix())
        _shader.set_mat4('p_matrix', cam.projection_matrix())
        self._mesh.render()

    def delete(self) -> None:
        self._mesh.delete()


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
        width= utils.SCREEN_WIDTH
        height= utils.SCREEN_HEIGHT
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glwin= window.GlWindow(width, height, 'glfw')
        glfw.make_context_current(glwin.window)
        glwin.center_screen_position()
        glwin.set_window_resize_callback(cb_window_resize)

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)

        # ---

        time= clock.Clock()

        cam= camera.Camera(maths.Vec3(z= 3.0),float(width / height))

        kb: keyboard.Keyboard= keyboard.Keyboard()

        ms: mouse.Mouse= mouse.Mouse()
        first_move: bool= True
        last_mp: maths.Vec3= maths.Vec3.zero()

        shader1= shader.Shader('debug_shader.vert', 'debug_shader.frag')

        shape1= CubeModel(0.5)
        shape2= CubeModel(0.3)

        bgcolor= color.Color.create_from_rgba(75, 75, 75, 255)

        while not glwin.should_close():
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glClearColor(*bgcolor.unit_values())
            GL.glEnable(GL.GL_DEPTH_TEST)
            
            # ---
            
            # time
            time.update()

            # shape
            shape1.rotate(maths.Vec3(x= 10.0, y= 10.0) * (1.4 * time.delta))
            shape1.draw(shader1, cam)

            shape2.rotate(maths.Vec3(y=10.0, z=5.0) * (1.4 * time.delta))
            shape2.draw(shader1, cam)


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

            GL.glDisable(GL.GL_DEPTH_TEST)
            
            # ---
            
            glfw.swap_buffers(glwin.window)
            glfw.poll_events()

    except Exception as err:
        logger.error(f"ERROR: {err}")

    finally:
        logger.debug('CLOSED')
        
        shape1.delete()
        shape2.delete()
        shader1.delete()

        glfw.terminate()


if __name__ == '__main__':
    main()

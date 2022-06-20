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
from py_opengl import model
from py_opengl import geometry
from py_opengl import abtree


# ---

# TMP
class Scene:

    __slots__= ('objects', 'tree')

    def __init__(self) -> None:
        self.objects: list[model.Model|None]= []
        self.tree: abtree.ABTree= abtree.ABTree()

    def add_obj(self, obj: model.Model) -> None:
        if obj in self.objects:
            return

        self.objects.append(obj)
        self.tree.add(obj)

    def remove_obj(self, obj: model.Model) -> None:
        if obj not in self.objects:
            return
        self.objects.remove(obj)
        self.tree.remove(obj)

    def update_obj(self, obj: model.Model) -> None:
        if obj not in self.objects:
            return
        self.tree.update(obj)

    def raycast(self, ray: geometry.Ray3) -> model.Model|None:
        return self.tree.raycast(ray)
    
    def query(self, ab: geometry.AABB3) -> list[model.Model|None]:
        return self.tree.query(ab)

    def draw_debug(self, s: shader.Shader, view:maths.Mat4, projection:maths.Mat4) -> None:
        self.tree.debug(s, view, projection)


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

        # ---

        time= clock.Clock()

        cam= camera.Camera(maths.Vec3(z= 5.0), float(width / height))

        kb: keyboard.Keyboard= keyboard.Keyboard(glwin)
        ms: mouse.Mouse= mouse.Mouse(glwin)
        first_move: bool= True
        last_mp: maths.Vec3= maths.Vec3.zero()

        shader0= shader.Shader('debug_shader.vert', 'debug_shader.frag')

        shape0= model.CubeModel(maths.Vec3.create_from_value(0.5))
        shape1= model.PyramidModel(0.3)
        shape2= model.CubeModel(maths.Vec3(0.2, 0.5, 0.2))
        shape3= model.CubeModelAABB(geometry.AABB3(maths.Vec3(), maths.Vec3.create_from_value(0.3)))
        shape4= model.CubeModel(maths.Vec3.create_from_value(0.5))

        shape0.translate(maths.Vec3(0.0, 0.0, 0.0))
        shape1.translate(maths.Vec3(1.5, 1.5, -2.5))
        shape2.translate(maths.Vec3(2.5, -2.0, 2.0))
        shape3.translate(maths.Vec3(1.5, 1.0, 1.0))
        shape4.translate(maths.Vec3(-1.0, -1.0, 1.0))

        bgcolor= color.Color.create_from_rgba(75, 75, 75, 255)

        scene: Scene= Scene()
        scene.add_obj(shape0)
        scene.add_obj(shape1)
        scene.add_obj(shape2)
        scene.add_obj(shape3)
        scene.add_obj(shape4)

        fshape= model.FrustumModel(cam.get_frustum_corners(True))
        fshape.translate(maths.Vec3(z=3))

        while not glwin.should_close():
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glClearColor(*bgcolor.unit_values())
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_CULL_FACE)
        
            # --

            time.update()

            # --
            
            v_matrix= cam.get_view_matrix()
            p_matrix= cam.get_projection_matrix()

            shape1.rotate(maths.Vec3(x= 30.0, z= 25.0) * (1.2 * time.delta))
            shape0.draw(shader0, v_matrix, p_matrix)
            shape1.draw(shader0, v_matrix, p_matrix)
            shape2.draw(shader0, v_matrix, p_matrix)
            shape3.draw(shader0, v_matrix, p_matrix)
            shape4.draw(shader0, v_matrix, p_matrix)

            fshape.draw(shader0, v_matrix, p_matrix, True)
            scene.draw_debug(shader0, v_matrix, p_matrix)

            # if kb.is_key_held(glfw.KEY_I):
            #     shape1.translate(maths.Vec3(y= 1.5) * (1.4 * time.delta))
        
            # if kb.is_key_held(glfw.KEY_K):
            #     shape1.translate(maths.Vec3(y= -1.5) * (1.4 * time.delta))

            # if kb.is_key_held(glfw.KEY_J):
            #     shape1.translate(maths.Vec3(x= -1.5) * (1.4 * time.delta))

            # if kb.is_key_held(glfw.KEY_L):
            #     shape1.translate(maths.Vec3(x= 1.5) * (1.4 * time.delta))

            # if kb.is_key_held(glfw.KEY_O):
            #     shape1.translate(maths.Vec3(z= 1.5) * (1.4 * time.delta))

            # if kb.is_key_held(glfw.KEY_U):
            #     shape1.translate(maths.Vec3(z= -1.5) * (1.4 * time.delta))

            # --
        
            if kb.is_key_held(glfw.KEY_W):
                cam.translate(camera.CameraDirection.IN, time.delta)

            if kb.is_key_held(glfw.KEY_S):
                cam.translate(camera.CameraDirection.OUT, time.delta)

            if kb.is_key_held(glfw.KEY_A):
                cam.translate(camera.CameraDirection.LEFT, time.delta)

            if kb.is_key_held(glfw.KEY_D):
                cam.translate(camera.CameraDirection.RIGHT, time.delta)

            if kb.is_key_held(glfw.KEY_E):
                cam.translate(camera.CameraDirection.UP, time.delta)

            if kb.is_key_held(glfw.KEY_Q):
                cam.translate(camera.CameraDirection.DOWN, time.delta)

            if ms.is_button_held(glfw.MOUSE_BUTTON_LEFT):
                if first_move:
                    mx, my= glwin.get_mouse_pos()
                    last_mp.x= mx
                    last_mp.y= my
                    first_move = False
                else:
                    mx, my= glwin.get_mouse_pos()
                    new_mp= maths.Vec3(x= mx, y= my) - last_mp
                    last_mp.x= mx
                    last_mp.y= my

                    cam.rotate(camera.CameraRotation.YAW, new_mp.x, time.delta)
                    cam.rotate(camera.CameraRotation.PITCH, new_mp.y, time.delta)

            # --

            glfw.swap_buffers(glwin.window)
            glfw.poll_events()

    except Exception as err:
        logger.error(f'ERROR: {err}')

    finally:
        logger.debug('CLOSED')
        shape0.delete()
        shape1.delete()
        shape2.delete()
        shape3.delete()
        shape4.delete()
        fshape.delete()
        
        shader0.delete()
        
        glfw.terminate()


if __name__ == '__main__':
    main()

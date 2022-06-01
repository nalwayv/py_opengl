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
from py_opengl import gjk
from py_opengl import tmp
from py_opengl import geometry

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

        cam= camera.Camera(maths.Vec3(z= 3.0), float(width / height))

        kb: keyboard.Keyboard= keyboard.Keyboard()

        ms: mouse.Mouse= mouse.Mouse()
        first_move: bool= True
        last_mp: maths.Vec3= maths.Vec3.zero()

        shader0= shader.Shader('debug_shader.vert', 'debug_shader.frag')

        shape0= model.CubeModel(maths.Vec3.create_from_value(0.5))
        shape1= model.PyramidModel(0.3)
        shape2= model.CubeModel(maths.Vec3(0.2, 0.5, 0.2))
        shape3= model.CubeModel(maths.Vec3.create_from_value(0.4))

        shape1.translate(maths.Vec3(1.5, 1.5, 0.0))
        shape2.translate(maths.Vec3(1.0, -1.0, 1.0))
        shape3.translate(maths.Vec3(0.0, 1.0, -1.0))

        bgcolor= color.Color.create_from_rgba(75, 75, 75, 255)

        # tr= tmp.Tree0()
        # tr.add(shape0)
        # tr.add(shape1)
        # tr.add(shape2)
        # tr.add(shape3)
        # ray= geometry.Ray3(maths.Vec3(1.5, 1.5, 5), maths.Vec3(z= -1))
        # ray_shape= model.LineModel(ray.origin, ray.origin + ray.direction * 10)
        # if tr.ray_cast(ray):
        #     print('rc= True')
        
        tr= tmp.Tree1(4, 1)
        tr.add(shape0)
        tr.add(shape1)
        tr.add(shape2)
        tr.add(shape3)

        ray= geometry.Ray3(maths.Vec3(0.0, 1.0, 5), maths.Vec3(z= -1))
        ray_shape= model.LineModel(ray.origin, ray.origin + ray.direction * 10)
        if tr.raycast_check(ray):
            print('ray cast= true')
            

        while not glwin.should_close():
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glClearColor(*bgcolor.unit_values())
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_CULL_FACE)
        
            # --

            time.update()

            # --

            shape1.rotate(1, maths.Vec3(x=0.2, y=0.2) * (1.4 * time.delta))

            shape0.draw(shader0, cam)
            shape1.draw(shader0, cam)
            shape2.draw(shader0, cam)
            shape3.draw(shader0, cam)

            ray_shape.draw(shader0, cam, True)
            tr.debug(shader0, cam)


            if kb.is_key_pressed(glwin.get_key_state(glfw.KEY_G)):
                mksum= gjk.Minkowskisum(shape0, shape1)
                gjk_check= gjk.GJK()
                check= gjk_check.detect(mksum)
                if check:
                    print('GJK= True')
                else:
                    print(' ')

            # --

            if kb.is_key_held(glwin.get_key_state(glfw.KEY_I)):
                shape1.translate(maths.Vec3(y= 1.5) * (1.4 * time.delta))
        
            if kb.is_key_held(glwin.get_key_state(glfw.KEY_K)):
                shape1.translate(maths.Vec3(y= -1.5) * (1.4 * time.delta))

            if kb.is_key_held(glwin.get_key_state(glfw.KEY_J)):
                shape1.translate(maths.Vec3(x= -1.5) * (1.4 * time.delta))

            if kb.is_key_held(glwin.get_key_state(glfw.KEY_L)):
                shape1.translate(maths.Vec3(x= 1.5) * (1.4 * time.delta))

            if kb.is_key_held(glwin.get_key_state(glfw.KEY_O)):
                shape1.translate(maths.Vec3(z= 1.5) * (1.4 * time.delta))

            if kb.is_key_held(glwin.get_key_state(glfw.KEY_U)):
                shape1.translate(maths.Vec3(z= -1.5) * (1.4 * time.delta))

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


            if ms.is_button_held(
                glwin.get_mouse_state(glfw.MOUSE_BUTTON_LEFT)
            ):
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

            # --

            glfw.swap_buffers(glwin.window)
            glfw.poll_events()

    # except Exception as err:
    #     logger.error(f"ERROR: {err}")

    finally:
        logger.debug('CLOSED')
        shape0.delete()
        shape1.delete()
        shape2.delete()
        shape3.delete()
        ray_shape.delete()
        shader0.delete()
        glfw.terminate()


if __name__ == '__main__':
    main()

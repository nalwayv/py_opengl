
from dataclasses import dataclass, field
from py_opengl import glm

@dataclass(eq=False, repr=False, slots=True)
class Camera:
    position: glm.Vec3 = glm.Vec3()
    front: glm.Vec3 = glm.Vec3(z=-1.0)
    up: glm.Vec3 = glm.Vec3(y=1.0)
    right: glm.Vec3 = glm.Vec3(x=1.0)
    aspect: float = 1.0

    speed: float= 1.4
    fovy: float = glm.PIOVER2
    yaw: float = glm.PIOVER2 * -1.0     # -90.0 deg
    pitch: float = 0.0
    znear: float = 0.01
    zfar: float = 1000.0


def camera_update(cam: Camera) -> None:
    '''Update camera

    Parameters
    ---
    cam: Camera

    Returns
    ---
    None
    '''
    cam.front.x = glm.cos(cam.pitch) * glm.cos(cam.yaw)
    cam.front.y = glm.sin(cam.pitch)
    cam.front.z = glm.cos(cam.pitch) * glm.sin(cam.yaw)

    cam.front = glm.v3_unit(cam.front)
    cam.right = glm.v3_unit(glm.v3_cross(cam.front, glm.Vec3(y=1.0)))
    cam.up = glm.v3_unit(glm.v3_cross(cam.right, cam.front))


def camera_yaw(val: float) -> float:
    '''Helper function for camera yaw

    Parameters
    ---
    val: float
        value in degrees,
        will be converted into radians within function

    Example
    ---
    camera.yaw += camera_yaw(45.5)

    Returns
    ---
    float: yaw value in radians
    '''
    return glm.to_radians(val)


def camera_pitch(val: float) -> float:
    '''Helper function for camera pitch

    Parameters
    ---
    val: float
        value in degrees,
        will be converted into radians within function

    Example
    ---
    camera.pitch += camera_pitch(45.5)

    Returns
    ---
    float: pitch value in radians
    '''

    return glm.to_radians(glm.clamp(val, -89.0, 89.0))


def camera_fovy(val: float) -> float:
    '''Helper function for camera fovy

    Parameters
    ---
    val: float
        value in degrees,
        will be converted into radians within function

    Example
    ---
    camera.fovy += camera_fovy(45.5)

    Returns
    ---
    float: fovy value in radians
    '''
    return glm.to_radians(glm.clamp(val, 1.0, 45.0))


def camera_view_matrix(cam: Camera) -> glm.Mat4:
    '''Return Camera view matrix

    Parameters
    ---
    cam: Camera

    Returns
    ---
    Mat4: camera view matrix 4x4
    '''
    return glm.m4_look_at(cam.position, cam.position + cam.front, cam.up)


def camera_perspective_matrix(cam: Camera) -> glm.Mat4:
    '''Return camera projection matrix

    Parameters
    ---
    cam: Camera

    Returns
    ---
    Mat4: camera projection matrix 4x4
    '''
    return glm.m4_projection(cam.fovy, cam.aspect, cam.znear, cam.zfar)


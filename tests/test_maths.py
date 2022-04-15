from py_opengl import maths
import pytest

def test_vec_unit():
    """test convertion to unit"""
    a: maths.Vec2 = maths.Vec2(1.0, 1.0)
    a.to_unit()
    assert a.is_unit()

    b: maths.Vec3 = maths.Vec3(1.0, 1.0, 1.0)
    b.to_unit()
    assert b.is_unit()

    c: maths.Vec4 = maths.Vec4(1.0, 1.0, 1.0, 1.0)
    c.to_unit()
    assert c.is_unit()


def test_vec_equil():
    """test vec equil"""
    a1: maths.Vec2 = maths.Vec2(1.0, 1.0)
    a2: maths.Vec2 = maths.Vec2(1.0, 1.0)
    assert a1.is_equil(a2)

    b1: maths.Vec3 = maths.Vec3(1.0, 1.0, 1.0)
    b2: maths.Vec3 = maths.Vec3(1.0, 1.0, 1.0)
    assert b1.is_equil(b2)

    c1: maths.Vec4 = maths.Vec4(1.0, 1.0, 1.0, 1.0)
    c2: maths.Vec4 = maths.Vec4(1.0, 1.0, 1.0, 1.0)
    assert c1.is_equil(c2)
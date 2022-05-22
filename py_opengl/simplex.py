"""Simplex
"""
# TODO

from typing import TypeVar, Optional
from py_opengl import maths
from py_opengl import model


# ---


MT= TypeVar('MT', bound= model.Model)


# ---


class SimplexError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class Simplex:

    __slots__= ('_pts', '_count')

    def __init__(self) -> None:
        self._pts: list[maths.Vec3]= [maths.Vec3.zero()]*4
        self._count: int= 0

    def _line(self, dir: maths.Vec3) -> bool:
        a: maths.Vec3= self.get_at(0)
        b: maths.Vec3= self.get_at(1)

        ab: maths.Vec3= b - a
        ao: maths.Vec3= a * -1.0

        if ab.dot(ao) > 0.0:
            dir.set_from(ab.cross(ao).cross(ab))
        else:
            self.simplex.set(a)

        return False

    def _triangle(self, dir: maths.Vec3) -> bool:
        a: maths.Vec3= self.get_at(0)
        b: maths.Vec3= self.get_at(1)
        c: maths.Vec3= self.get_at(2)

        ab: maths.Vec3= b - a
        ac: maths.Vec3= c - a
        ao: maths.Vec3= a * -1.0

        abc: maths.Vec3= ab.cross(ac)

        if abc.cross(ac).dot(ao) > 0.0:
            if ac.dot(ao) > 0.0:
                self.set_pts(a, c)
                dir.set_from(ac.cross(ao).cross(ac))
            else:
                self.set_pts(a, b)
                return self._line(dir)
        else:
            if ab.cross(abc).dot(ao) > 0.0:
                self.set_pts(a, b)
                return self._line(dir)
            else:
                if abc.dot(ao) > 0.0:
                    dir.set_from(abc)
                else:
                    self.set_pts(a, c, b)
                    dir.set_from(abc * -1.0)
        return False

    def _tetrahedron(self, dir: maths.Vec3) -> bool:
        a: maths.Vec3= self.get_at(0)
        b: maths.Vec3= self.get_at(1)
        c: maths.Vec3= self.get_at(2)
        d: maths.Vec3= self.get_at(3)

        ab: maths.Vec3= b - a
        ac: maths.Vec3= c - a
        ad: maths.Vec3= d - a
        ao: maths.Vec3= a * -1.0

        abc= ab.cross(ac)
        acd= ac.cross(ad)
        adb= ad.cross(ab)

        if abc.dot(ao) > 0.0:
            self.set_pts(a, b, c)
            return self._triangle(dir)

        if acd.dot(ao) > 0.0:
            self.set_pts(a, c, d)
            return self._triangle(dir)

        if adb.dot(ao) > 0.0:
            self.set_pts(a, d, b)
            return self._triangle(dir)

        return True

    def next_simplex(self, dir: maths.Vec3) -> bool:
        match self.simplex.size():
            case 2:
                return self._line(dir)
            case 3:
                return self._triangle(dir)
            case 4:
                return self._tetrahedron(dir)
        return False

    def size(self) -> int:
        return self._count

    def set_pts(self, *pts: tuple[maths.Vec3]) -> None:
        n: int= len(pts)
        if n > 4:
            return

        self._count= n
        for idx, pt in enumerate(pts):
            self._pts[idx].set_from(pt)

    def clear(self) -> None:
        self._count= 0
        self.set_at(0, maths.Vec3.zero())
        self.set_at(1, maths.Vec3.zero())
        self.set_at(2, maths.Vec3.zero())
        self.set_at(3, maths.Vec3.zero())

    def get_at(self, idx: int) -> maths.Vec3:
        match idx:
            case 0:
                return self._pts[0]
            case 1:
                return self._pts[1]
            case 2:
                return self._pts[2]
            case 3:
                return self._pts[3]

        raise SimplexError('out of range')

    def set_at(self, idx: int, val: maths.Vec3) -> None:
        match idx:
            case 0:
                self._pts[0].set_from(val)
            case 1:
                self._pts[1].set_from(val)
            case 2:
                self._pts[2].set_from(val)
            case 3:
                self._pts[3].set_from(val)

    def push_pt(self, pt: maths.Vec3):
        self.set_at(3, self.get_at(2))
        self.set_at(2, self.get_at(1))
        self.set_at(1, self.get_at(0))
        self.set_at(0, pt)


# ---


class Minkowskisum:

    __slots__=  ('m0', 'm1')

    def __init__(
        self,
        m0: Optional[MT]= None,
        m1: Optional[MT]= None
    ) -> None:
        self.m0: Optional[MT]= m0
        self.m1: Optional[MT]= m1

    def get_support(self, dir: maths.Vec3) -> maths.Vec3:
        if (self.m0 is None) or (self.m1 is None):
            return maths.Vec3.zero()

        p0: maths.Vec3= self.m0.get_furthest_pt(dir)
        p1: maths.Vec3= self.m1.get_furthest_pt(dir * -1.0)

        return p0 - p1

    def get_dir(self) -> maths.Vec3:
        if (self.m0 is None) or (self.m1 is None):
            return maths.Vec3(x= 1.0)

        p0: maths.Vec3= self.m0.get_position()
        p1: maths.Vec3= self.m1.get_position()
        dir: maths.Vec3= p1 - p0

        if not dir.is_unit():
            dir.to_unit()

        return dir


# ---


class GJK:

    __slots__= ('simplex',)

    def __init__(self) -> None:
        self.simplex= Simplex()


    def detect(self, ms: Minkowskisum, dir: maths.Vec3= maths.Vec3()) -> bool:
        if dir.is_zero():
            dir.set_from(ms.get_dir())

        self.simplex.push_pt(ms.get_support(dir))

        if self.simplex.get_at(0).dot(dir) <= 0.0:
            return False
        
        dir.set_from(dir * -1.0)

        while True:
            sp: maths.Vec3= ms.get_support(dir)
            self.simplex.push_pt(sp)

            if sp.dot(dir) <= 0.0:
                return False
            else:
                if self.simplex.next_simplex(dir):
                    return True
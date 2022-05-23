"""Simplex
"""
# TODO

from typing import TypeVar, Optional, Final
from py_opengl import maths
from py_opengl import model


# ---


MT= TypeVar('MT', bound= model.Model)
OPTIMAL: Final[float]= 1e-3

# ---


class SimplexError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class Simplex:

    __slots__= ('_pts', '_count')

    def __init__(self) -> None:
        self._pts: list[maths.Vec3]= []
        self._count: int= 0

    def _solve2(self, dir: maths.Vec3) -> bool:
        if self.length() < 2:
            return False
    
        a: maths.Vec3= self.get_at(1)
        b: maths.Vec3= self.get_at(0)

        ab: maths.Vec3= b - a
        ao: maths.Vec3= a * -1.0

        if ab.dot(ao)> 0.0:
            dir.set_from(ab.cross(ao).cross(ab))
        else:
            self._pts= [a]
            dir.set_from(ao)

        return False

    def _solve3(self, dir: maths.Vec3) -> bool:
        if self.length() < 3:
            return False
    
        a: maths.Vec3= self.get_at(2)
        b: maths.Vec3= self.get_at(1)
        c: maths.Vec3= self.get_at(0)

        ab: maths.Vec3= b - a
        ac: maths.Vec3= c - a
        ao: maths.Vec3= a * -1.0

        abc: maths.Vec3= ab.cross(ac)

        t0 = abc.cross(ac)
        t1= ab.cross(abc)

        if t0.dot(ao) > 0.0:
            if ac.dot(ao) > 0.0:
                self.remove_at(1)
                dir.set_from(ac.cross(ao).cross(ac))
            else:
                self.remove_at(0)
                return self._solve2(dir)
        else:
            if t1.dot(ao) > 0.0:
                self.remove_at(0)
                return self._solve2(dir)
            else:
                if abc.dot(ao):
                    dir.set_from(abc)
                else:
                    self._pts= [b, c, a]
                    dir.set_from(abc * -1.0)
    
        return False

    def _solve4(self, dir: maths.Vec3) -> bool:
        if self.length() < 4:
            return False

        a: maths.Vec3= self.get_at(3)
        b: maths.Vec3= self.get_at(2)
        c: maths.Vec3= self.get_at(1)
        d: maths.Vec3= self.get_at(0)

        ab: maths.Vec3= b - a
        ac: maths.Vec3= c - a
        ad: maths.Vec3= d - a
        ao: maths.Vec3= a * -1.0

        abc= ab.cross(ac)
        acd= ac.cross(ad)
        adb= ad.cross(ab)

        if abc.dot(ao) > 0.0:
            self._pts= [c, b, a]
            return self._solve3(dir)

        if acd.dot(ao) > 0.0:
            self._pts= [d, c, a]
            return self._solve3(dir)

        if adb.dot(ao) > 0.0:
            self._pts= [b, d, a]
            return self._solve3(dir)

        return True

    def check_next(self, dir: maths.Vec3) -> bool:
        match self.length():
            case 2:
                return self._solve2(dir)
            case 3:
                return self._solve3(dir)
            case 4:
                return self._solve4(dir)
        return False

    def length(self) -> int:
        return len(self._pts)

    def get_at(self, idx: int) -> maths.Vec3:
        if idx < 0 or idx >= self.length():
            raise SimplexError('out of range')
        return self._pts[idx]

    def remove_at(self, idx: int) -> None:
        if idx < 0 or idx >= self.length():
            return
        self._pts.pop(idx)

    def push(self, pt: maths.Vec3):
        self._pts.append(pt)
    
    def clear(self) -> None:
        """Clear current pts
        """
        self._pts.clear()


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
        dir: maths.Vec3= p0 - p1

        return dir


# ---


class GJK:

    __slots__= ('simplex', 'iterations')

    def __init__(self) -> None:
        self.iterations: int= 30

    # def _check_is_optimal(self, next_pt: maths.Vec3, dir: maths.Vec3):
    #     """
    #     """
    #     d0: float= next_pt.dot(dir)

    #     for i in range(self.simplex.length()):
    #         d1: float= self.simplex.get_at(i).dot(dir)
    #         if (d0 - d1) > OPTIMAL:
    #             return False
    #     return True

    def detect(self, mksum: Minkowskisum, dir: maths.Vec3= maths.Vec3()) -> bool:
        """
        """

        if dir.is_zero():
            dir.set_from(mksum.get_dir())

        simp= Simplex()
        npt: maths.Vec3= mksum.get_support(dir)
        simp.push(npt)
        
        dir.set_from(npt * -1.0)

        for _ in range(self.iterations):
            npt= mksum.get_support(dir)

            if npt.dot(dir) <= 0.0:
                return False

            simp.push(npt)

            if simp.check_next(dir):
                return True

        # return False
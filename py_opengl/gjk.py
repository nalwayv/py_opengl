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
        self._pts: list[maths.Vec3]= []

    def _solve2(self, dir: maths.Vec3) -> bool:
        if self.length() < 2:
            return False
    
        a: maths.Vec3= self._pts[1]
        b: maths.Vec3= self._pts[0]

        ab: maths.Vec3= b - a
        a0: maths.Vec3= a * -1.0

        if ab.dot(a0)> 0.0:
            dir.set_from(ab.cross(a0).cross(ab))
        else:
            self._pts= [a]
            dir.set_from(a0)

        return False

    def _solve3(self, dir: maths.Vec3) -> bool:
        if self.length() < 3:
            return False
    
        a: maths.Vec3= self._pts[2]
        b: maths.Vec3= self._pts[1]
        c: maths.Vec3= self._pts[0]

        ab: maths.Vec3= b - a
        ac: maths.Vec3= c - a
        a0: maths.Vec3= a * -1.0

        abc: maths.Vec3= ab.cross(ac)

        t0 = abc.cross(ac)
        t1= ab.cross(abc)

        if t0.dot(a0) > 0.0:
            if ac.dot(a0) > 0.0:
                self._pts= [a, c]
                dir.set_from(ac.cross(a0).cross(ac))
            else:
                self._pts= [a, b]
                return self._solve2(dir)
        else:
            if t1.dot(a0) > 0.0:
                self._pts= [a, b]
                return self._solve2(dir)
            else:
                if abc.dot(a0) > 0.0:
                    dir.set_from(abc)
                else:
                    self._pts= [a, c, b]
                    dir.set_from(abc * -1.0)
    
        return False

    def _solve4(self, dir: maths.Vec3) -> bool:
        if self.length() < 4:
            return False

        a: maths.Vec3= self._pts[3]
        b: maths.Vec3= self._pts[2]
        c: maths.Vec3= self._pts[1]
        d: maths.Vec3= self._pts[0]

        ab: maths.Vec3= b - a
        ac: maths.Vec3= c - a
        ad: maths.Vec3= d - a
        a0: maths.Vec3= a * -1.0

        acb= ac.cross(ab)
        d_acb: float= acb.dot(a0)

        abd = ab.cross(ad)
        d_abd: float= abd.dot(a0)

        adc = ad.cross(ac)
        d_adc: float= adc.dot(a0)

        neg: int= 0
        pos: int= 0


        if d_adc > 0:
            pos += 1
        else:
            neg += 1

        if d_abd > 0:
            pos += 1
        else:
            neg += 1

        if d_acb > 0:
            pos += 1
        else:
            neg += 1

        if pos == 3 or neg == 3:
            return True

        if neg == 2 and pos == 1:
            if d_adc > 0.0:
                self._pts= [a, b, c]
                dir.set_from(adc)
            elif d_abd > 0:
                self._pts= [a, b, d]
                dir.set_from(abd)
            else:
                self._pts= [a, c, d]
                dir.set_from(acb)
        elif neg == 1 and pos == 2:
            if d_adc < 0:
                self._pts= [a, b, c]
                dir.set_from(adc * -1.0)
            elif d_abd < 0:
                self._pts= [a, b, d]
                dir.set_from(abd * -1.0)
            else:
                self._pts= [a, c, d]                
                dir.set_from(acb * -1.0)

        return self._solve3(dir)

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

        if not dir.is_unit():
            dir.to_unit()
        return dir


# ---


class GJK:

    __slots__= ('simplex', 'iterations')

    def __init__(self) -> None:
        self.iterations: int= 30

    def detect(self, mksum: Minkowskisum, dir: maths.Vec3= maths.Vec3()) -> bool:
        """
        """

        if dir.is_zero():
            dir.set_from(mksum.get_dir())

        simp= Simplex()
        spt: maths.Vec3= mksum.get_support(dir)
        simp.push(spt)
        
        dir.set_from(spt * -1.0)

        for _ in range(self.iterations):
            spt= mksum.get_support(dir)

            if spt.dot(dir) <= 0.0:
                return False

            simp.push(spt)

            if simp.check_next(dir):
                return True

        return False
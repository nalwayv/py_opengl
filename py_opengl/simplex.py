"""Simplex
"""
# TODO

from py_opengl import maths


# ---


class SimplexError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


# ---


class Simplex:

    __slots__= ('_pts', '_count')

    def __init__(self) -> None:
        self._pts: list[maths.Vec3]= [maths.Vec3.zero()]*4
        self._count: int= 0

    def size(self) -> int:
        return self._count

    def set(self, *pts: tuple[maths.Vec3]) -> None:
        n= len(pts)
        if n > 4:
            return

        self._count= n
        for idx, pt in enumerate(pts):
            self._pts[idx].set_from(pt)

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

    def push(self, pt: maths.Vec3):
        self._pts[3].set_from(self._pts[2])
        self._pts[2].set_from(self._pts[1])
        self._pts[1].set_from(self._pts[0])
        self._pts[0].set_from(pt)


# ---


class Foo:

    __slots__= ('simplex',)

    def __init__(self) -> None:
        self.simplex= Simplex()

    def _line(self, dir: maths.Vec3) -> bool:
        a: maths.Vec3= self.simplex.get_at(0)
        b: maths.Vec3= self.simplex.get_at(1)

        ab: maths.Vec3= b - a
        ao: maths.Vec3= a * -1.0

        if ab.dot(ao) > 0.0:
            dir.set_from(ab.cross(ao).cross(ab))
        else:
            self.simplex.set(a)

        return False

    def _triangle(self, dir: maths.Vec3) -> bool:
        a: maths.Vec3= self.simplex.get_at(0)
        b: maths.Vec3= self.simplex.get_at(1)
        c: maths.Vec3= self.simplex.get_at(2)

        ab: maths.Vec3= b - a
        ac: maths.Vec3= c - a
        ao: maths.Vec3= a * -1.0

        abc: maths.Vec3= ab.cross(ac)

        if abc.cross(ac).dot(ao) > 0.0:
            if ac.dot(ao) > 0.0:
                self.simplex.set(a, c)
                dir.set_from(ac.cross(ao).cross(ac))
            else:
                self.simplex.set(a, b)
                return self.line(dir)
        else:
            if ab.cross(abc).dot(ao) > 0.0:
                self.simplex.set(a, b)
                return self.line(dir)
            else:
                if abc.dot(ao) > 0.0:
                    dir.set_from(abc)
                else:
                    self.simplex.set(a, c, b)
                    dir.set_from(abc * -1.0)
        return False

    def _tetrahedron(self, dir: maths.Vec3) -> bool:
        a: maths.Vec3= self.simplex.get_at(0)
        b: maths.Vec3= self.simplex.get_at(1)
        c: maths.Vec3= self.simplex.get_at(2)
        d: maths.Vec3= self.simplex.get_at(3)

        ab: maths.Vec3= b - a
        ac: maths.Vec3= c - a
        ad: maths.Vec3= d - a
        ao: maths.Vec3= a * -1.0

        abc= ab.cross(ac)
        acd= ac.cross(ad)
        adb= ad.cross(ab)

        if abc.dot(ao) > 0.0:
            self.simplex.set(a, b, c)
            return self.triangle(dir)

        if acd.dot(ao) > 0.0:
            self.simplex.set(a, c, d)
            return self.triangle(dir)

        if adb.dot(ao) > 0.0:
            self.simplex.set(a, d, b)
            return self.triangle(dir)

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
""""""
from dataclasses import dataclass
from py_opengl import glm


# --- AABB


@dataclass(eq=False, repr=False, slots=True)
class Aabb:
    position: glm.Vec3 = glm.Vec3()
    size: glm.Vec3 = glm.Vec3()

    @staticmethod
    def from_min_max(
            min_pt: glm.Vec3,
            max_pt: glm.Vec3
        ) -> 'Aabb':
        return Aabb(
            (min_pt + max_pt) * 0.5,
            (max_pt - min_pt) * 0.5
        )

    def copy(self) -> 'Aabb':
        """Return a copy of self

        Returns
        ---
        Abbb
        """
        return Aabb(self.position.copy(), self.size.copy())

    def closest_pt(self, pt: glm.Vec3) -> glm.Vec3:
        pt_min: glm.Vec3 = self.get_min()
        pt_max: glm.Vec3 = self.get_max()

        x: float = glm.clamp(pt.x, pt_min.x, pt_max.x)
        y: float = glm.clamp(pt.y, pt_min.y, pt_max.y)
        z: float = glm.clamp(pt.z, pt_min.z, pt_max.z)

        return glm.Vec3(x, y, z)

    def get_min(self) -> glm.Vec3:
        pt1: glm.Vec3 = self.position + self.size
        pt2: glm.Vec3 = self.position - self.size
        
        x: float = glm.minf(pt1.x, pt2.x)
        y: float = glm.minf(pt1.y, pt2.y)
        z: float = glm.minf(pt1.z, pt2.z)

        return glm.Vec3(x, y, z)

    def get_max(self) -> glm.Vec3:
        pt1: glm.Vec3 = self.position + self.size
        pt2: glm.Vec3 = self.position - self.size
        
        x: float = glm.maxf(pt1.x, pt2.x)
        y: float = glm.maxf(pt1.y, pt2.y)
        z: float = glm.maxf(pt1.z, pt2.z)

        return glm.Vec3(x, y, z)

    def intersect_aabb(self, other: 'Aabb') -> bool:
        amin: glm.Vec3 = self.get_min()
        amax: glm.Vec3 = self.get_max()

        bmin: glm.Vec3 = other.get_min()
        bmax: glm.Vec3 = other.get_max()

        check_x: bool = amin.x <= bmax.x and amax.x >= bmin.x
        check_y: bool = amin.y <= bmax.y and amax.y >= bmin.y
        check_z: bool = amin.z <= bmax.z and amax.z >= bmin.z

        return check_x and check_y and check_z

    def intersect_pt(self, pt: glm.Vec3) -> bool:
        amin = self.get_min()
        amax = self.get_max()

        check = (
            (pt.x < amin.x) or
            (pt.y < amin.y) or
            (pt.x > amax.x) or
            (pt.y > amax.y)
        )

        return not check

    def intersect_sphere(self, sph: 'Sphere') -> bool:
        close_pt: glm.Vec3 = self.closest_pt(sph.position)
        dis: float = (sph.position - close_pt).length_sqr()
        r2 = glm.sqr(sph.radius)
        return dis < r2

    def intersect_plain(self, plain: 'Plain') -> bool:
        len_sq: float = (
            self.size.x * glm.absf(plain.normal.x) +
            self.size.y * glm.absf(plain.normal.y) +
            self.size.z * glm.absf(plain.normal.z)
        )

        dot: float = plain.normal.dot(self.position)
        dis: float = dot - plain.direction

        return glm.absf(dis) <= len_sq


# --- SPHERE


@dataclass(eq=False, repr=False, slots=True)
class Sphere:
    position: glm.Vec3 = glm.Vec3()
    radius: float = 1.0

    def copy(self) -> 'Sphere':
        """Return a copy of self

        Returns
        ---
        Sphere
        """
        return Sphere(self.position.copy(), self.radius)

    def closest_pt(self, pt: glm.Vec3) -> glm.Vec3:
        p = pt - self.position
        p.to_unit()
        return self.position + (p * self.radius)

    def intersect_sphere(self, other: 'Sphere') -> bool:
        dis: float = (self.position - other.position).length_sqr()
        r2: float = glm.sqr(self.radius + other.radius)
        return dis < r2

    def intersect_pt(self, pt: glm.Vec3) -> bool:
        dis: float = (self.position - pt).length_sqr()
        r2: float = glm.sqr(self.radius)
        return dis < r2

    def intersect_aabb(self, aabb: 'Aabb') -> bool:
        close_pt: glm.Vec3 = aabb.closest_pt(self.position)
        dis: float = (self.position - close_pt).length_sqr()
        r2 = glm.sqr(self.radius)
        return dis < r2

    def intersect_plain(self, plain: 'Plain'):
        close_pt: glm.Vec3 = plain.closest_pt(self.position)
        dis: float = (self.position - close_pt).length_sqr()
        r2: float = glm.sqr(self.radius)
        return dis < r2


# --- PLAIN


class PlainError(Exception):
    '''Custom error for plain'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Plain:
    normal: glm.Vec3 = glm.Vec3(x=1.0)
    direction: float = 0.0

    def __post_init__(self):
        if not self.normal.is_unit():
            self.normal.to_unit()

    def copy(self) -> 'Plain':
        """Return a copy of self

        Returns
        ---
        Plain
        """
        return Plain(self.normal.copy(), self.dir)

    def dot(self, v4: glm.Vec4) -> float:
        x: float = self.normal.x * v4.x
        y: float = self.normal.y * v4.y
        z: float = self.normal.z * v4.z
        w: float = self.direction * v4.w
        return x + y + z + w

    def dot_coord(self, v3: glm.Vec3) -> float:
        x: float = self.normal.x * v3.x
        y: float = self.normal.y * v3.y
        z: float = self.normal.z * v3.z
        w: float = self.direction
        return x + y + z + w

    def dot_normal(self, v3: glm.Vec3) -> float:
        x: float = self.normal.x * v3.x
        y: float = self.normal.y * v3.y
        z: float = self.normal.z * v3.z
        return x + y + z

    def closest_pt(self, pt: glm.Vec3) -> glm.Vec3:
        dot: float = self.normal.dot(pt)
        dis: float = dot - self.direction
        return pt - (self.normal * dis)

    def unit(self) -> 'Plain':
        """Return a copy of self that has been normalized

        Returns
        ---
        Plain
        """
        len_sqr: float = self.normal.length_sqr()

        if glm.is_one(len_sqr):
            return self.copy()

        inv: float = glm.inv_sqrt(len_sqr)
        return Plain(
            self.normal * inv,
            self.direction * inv
        )

    def to_unit(self) -> None:
        """Normalize the length of self
        """
        len_sqr: float = self.normal.length_sqr()
        if glm.is_zero(len_sqr):
            raise PlainError('length of plain was zero')

        inv = glm.inv_sqrt(len_sqr)

        self.normal.x *= inv
        self.normal.y *= inv
        self.normal.z *= inv
        self.direction *= inv

    def intersect_plain(self, other: 'Plain') -> bool:
        dis: float = (self.normal.cross(other.nomal)).length_sqr()
        return not glm.is_zero(dis)

    def intersect_sphere(self, sph: 'Sphere') -> bool:
        close_pt = self.closest_pt(sph.position)
        len_sq = (sph.position - close_pt).length_sqr()
        r2 = glm.sqr(sph.radius)
        return len_sq < r2

    def intersect_aabb(self, aabb: 'Aabb') -> bool:
        len_sq: float = (
            aabb.size.x * glm.absf(self.normal.x) +
            aabb.size.y * glm.absf(self.normal.y) +
            aabb.size.z * glm.absf(self.normal.z)
        )

        dot: float = self.normal.dot(aabb.position)
        dis: float = dot - self.direction

        return glm.absf(dis) <= len_sq
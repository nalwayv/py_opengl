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

    def intersect(self, other: 'Aabb') -> bool:
        amin: glm.Vec3 = self.get_min()
        amax: glm.Vec3 = self.get_max()
        bmin: glm.Vec3 = other.get_min()
        bmax: glm.Vec3 = other.get_max()

        check_x: bool = amin.x <= bmax.x and amax.x >= bmin.x
        check_y: bool = amin.y <= bmax.y and amax.y >= bmin.y
        check_z: bool = amin.z <= bmax.z and amax.z >= bmin.z

        return check_x and check_y and check_z


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

    @staticmethod
    def from_verts(
            p1: glm.Vec3,
            p2: glm.Vec3,
            p3: glm.Vec3
        ) -> 'Plain':
        """Create from three points

        Returns
        ---
        Plain
        """
        a: glm.Vec3 = p2 - p1
        b: glm.Vec3 = p3 - p1
        n: glm.Vec3 = a.cross(b)
        n.to_unit()
        d: float = n.x * p1.x + n.y * p1.y + n.z * p1.z
        return Plain(n, -d)

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

    def is_equil(self, other: 'Plain') -> bool:
        """Check if self and other have the same component values

        Parameters
        ---
        other : Plain

        Returns
        ---
        bool
        """
        check_n: bool = self.normal.is_equil(other.normal)
        check_d: bool = glm.is_equil(self.direction, other.direction)
        return check_n and check_d


"""Geometry
"""
from audioop import cross
from enum import Enum, auto
from re import M

from py_opengl import maths
from py_opengl import transform


# --- IDs


class GeometryType(Enum):
    AABB3= auto()
    LINE= auto()
    TRIANGLE= auto()
    SPHERE= auto()
    PLAIN= auto()
    FRUSTUM= auto()
    RAY= auto()


# --- AABB


class AABB3:
    """AABB using center and extents


    Aabb(center= Vec3(0, 0, 0), extents= Vec3(1, 1, 1))

    get_min == Vec3(-1, -1, -1)\n
    get_max == Vec3(1, 1, 1)
    """

    __slots__= ('center', 'extents', 'TYPE')

    def __init__(
        self,
        center: maths.Vec3= maths.Vec3(),
        extents: maths.Vec3= maths.Vec3(),
    ) -> None:
        self.center: maths.Vec3= center
        self.extents: maths.Vec3= extents
        self.TYPE: GeometryType= GeometryType.AABB3

    def __hash__(self) -> int:
        data: tuple[float]= (
            self.center.x, self.center.y, self.center.z,
            self.extents.x, self.extents.y, self.extents.z
        )
        return hash(data)

    def __eq__(self, other: 'AABB3') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.center.is_equil(other.center) and
                self.extents.is_equil(other.extents) and
                self.TYPE == other.TYPE
            ):
                return True
        return False

    def __str__(self) -> str:
        pmin= self.get_min()
        pmax= self.get_max()
        return f'AABB\n\tMIN_V3({pmin.x}, {pmin.y}, {pmin.z})\n\tMAX_V3({pmax.x}, {pmax.y}, {pmax.z})'

    @staticmethod
    def create_from_min_max(min_pt: maths.Vec3, max_pt: maths.Vec3) -> 'AABB3':
        return AABB3(
            center= (min_pt + max_pt) * 0.5,
            extents= (min_pt - max_pt) * 0.5
        )

    @staticmethod
    def create_combined_from(a: 'AABB3', b: 'AABB3') -> 'AABB3':
        return AABB3.create_from_min_max(
            maths.Vec3.create_from_min(a.get_min(), b.get_min()),
            maths.Vec3.create_from_max(a.get_max(), b.get_max())
        )

    def combined_with(self, a: 'AABB3') -> None:
        """Set self to the union of self and a"""
        aabb= AABB3.create_from_min_max(
            maths.Vec3.create_from_min(self.get_min(), a.get_min()),
            maths.Vec3.create_from_max(self.get_max(), a.get_max())
        )
        self.center.set_from(aabb.center)
        self.extents.set_from(aabb.extents)

    def combine_with(self, a: 'AABB3') -> 'AABB3':
        """Set self to the union of self and a"""
        return AABB3.create_from_min_max(
            maths.Vec3.create_from_min(self.get_min(), a.get_min()),
            maths.Vec3.create_from_max(self.get_max(), a.get_max())
        )

    def combined_from(self, a: 'AABB3', b: 'AABB3') -> None:
        """Set self to the union of a and b"""
        aabb= AABB3.create_from_min_max(
            maths.Vec3.create_from_min(a.get_min(), b.get_min()),
            maths.Vec3.create_from_max(a.get_max(), b.get_max())
        )
        self.center.set_from(aabb.center)
        self.extents.set_from(aabb.extents)

    def expand(self, by: float) -> 'AABB3':
        """Expanded by float value"""
        if by < 0.0:
            by = maths.absf(by)

        p0: maths.Vec3= self.get_min() - maths.Vec3.create_from_value(by)
        p1: maths.Vec3= self.get_max() + maths.Vec3.create_from_value(by)

        return AABB3.create_from_min_max(p0, p1)

    def expanded(self, by: float) -> None:
        expand= self.expand(by)
        self.center.set_from(expand.center)
        self.extents.set_from(expand.extents)

    def translate(self, v3: maths.Vec3) -> None:
        """Translate by
        """
        self.center.set_from(self.center + v3)

    def copy(self) -> 'AABB3':
        """Return a copy of self
        """
        return AABB3(
            center= self.center.copy(),
            extents= self.extents.copy()
        )

    def perimeter(self) -> float:
        pmin: maths.Vec3= self.get_min()
        pmax: maths.Vec3= self.get_max()

        return 4.0 * (pmax - pmin).sum()

    def get_area(self) -> float:
        pmin= self.get_min()
        pmax= self.get_max()

        return(pmax - pmin).sum()

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        pmin: maths.Vec3= self.get_min()
        pmax: maths.Vec3= self.get_max()
        pt: maths.Vec3= maths.Vec3.zero()

        for i in range(3):
            val: float= pt.get_at(i)
            val= maths.minf(val, pmin.get_at(i))
            val= maths.maxf(val, pmax.get_at(i))
            pt.set_at(i, val)

        return pt

    def get_min(self) -> maths.Vec3:
        p0: maths.Vec3= self.center + self.extents
        p1: maths.Vec3= self.center - self.extents

        return maths.Vec3.create_from_min(p0, p1)

    def get_max(self) -> maths.Vec3:
        p0: maths.Vec3= self.center + self.extents
        p1: maths.Vec3= self.center - self.extents

        return maths.Vec3.create_from_max(p0, p1)

    def is_degenerate(self) -> bool:
        """Return true if self aabb is degenerate

        check if its min points equil its max points
        """
        pmin= self.get_min()
        pmax= self.get_max()

        check_x= maths.is_equil(pmin.x, pmax.x)
        check_y= maths.is_equil(pmin.y, pmax.y)
        check_z= maths.is_equil(pmin.z, pmax.z)

        return check_x and check_y and check_z

    def get_corners(self) -> list[maths.Vec3]:
        ptmin: maths.Vec3= self.get_min()
        ptmax: maths.Vec3= self.get_max()

        corners: list[maths.Vec3]= [
            maths.Vec3(ptmin.x, ptmin.y, ptmin.z),
            maths.Vec3(ptmax.x, ptmin.y, ptmin.z),
            maths.Vec3(ptmax.x, ptmax.y, ptmin.z),
            maths.Vec3(ptmin.x, ptmax.y, ptmin.z),
            maths.Vec3(ptmax.x, ptmin.y, ptmax.z),
            maths.Vec3(ptmax.x, ptmin.y, ptmax.z),
            maths.Vec3(ptmax.x, ptmax.y, ptmax.z),
            maths.Vec3(ptmin.x, ptmax.y, ptmax.z)
        ]
        return corners

    def contains_aabb(self, other: 'AABB3') -> bool:
        amin: maths.Vec3= self.get_min()
        amax: maths.Vec3= self.get_max()
        bmin: maths.Vec3= other.get_min()
        bmax: maths.Vec3= other.get_max()

        return (
            (amin.x <= bmin.x and amax.x >= bmax.x) and 
            (amin.y <= bmin.y and amax.y >= bmax.y) and 
            (amin.z <= bmin.z and amax.z >= bmax.z)
        )

    def intersect_aabb(self, other: 'AABB3') -> bool:
        amin: maths.Vec3= self.get_min()
        amax: maths.Vec3= self.get_max()
        bmin: maths.Vec3= other.get_min()
        bmax: maths.Vec3= other.get_max()
        
        return (
            (amin.x <= bmax.x and amax.x >= bmin.x) and
            (amin.y <= bmax.y and amax.y >= bmin.y) and
            (amin.z <= bmax.z and amax.z >= bmin.z)
        )

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        pmin: maths.Vec3= self.get_min()
        pmax: maths.Vec3= self.get_max()

        if pt.x < pmin.x or pt.y < pmin.y or pt.z < pmin.z:
            return False

        if pt.x > pmax.x or pt.y > pmax.y or pt.z > pmax.z:
            return False

        return True


# --- Line


class Line3:

    __slots__= ('start', 'end', 'TYPE')

    def __init__(
        self,
        start: maths.Vec3= maths.Vec3(),
        end: maths.Vec3= maths.Vec3()
    ) -> None:
        self.start: maths.Vec3= start
        self.end: maths.Vec3= end
        self.TYPE: GeometryType= GeometryType.LINE

    def __hash__(self) -> int:
        data: tuple[float]= (
            self.start.x, self.start.y, self.start.z,
            self.end.x, self.end.y, self.end.z
        )
        return hash(data)

    def __eq__(self, other: 'Line3') -> bool:
        if isinstance(other, self.__class__):
            if(
            self.start.is_equil(other.start) and
            self.end.is_equil(other.end) and
            self.TYPE == other.TYPE
            ):
                return True
        return False

    def edge(self) -> maths.Vec3:
        return (self.end - self.start)


# --- Triangle


class Triangle3:

    __slots__= ('p0', 'p1', 'p2', 'TYPE')

    def __init__(
        self,
        p0: maths.Vec3,
        p1: maths.Vec3,
        p2: maths.Vec3
    ) -> None:
        self.p0= p0.copy()
        self.p1= p1.copy()
        self.p2= p2.copy()
        self.TYPE: GeometryType= GeometryType.TRIANGLE

    def __hash__(self) -> int:
        data: tuple[float]= (
            self.p0.x, self.p0.y, self.p0.z,
            self.p1.x, self.p1.y, self.p1.z,
            self.p2.x, self.p2.y, self.p2.z
        )
        return hash(data)

    def __eq__(self, other: 'Triangle3') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.p0.is_equil(other.p0) and
                self.p1.is_equil(other.p1) and
                self.p2.is_equil(other.p2) and
                self.TYPE == other.TYPE
            ):
                return True
        return False


    def intersect_pt(self, pt: maths.Vec3):
        bc= maths.Vec3.barycentric(
            self.p0,
            self.p1,
            self.p2,
            pt
        )
        # u: float= bc.x
        v: float= bc.y
        w: float= bc.z
        return (
            v >= 0.0 and
            w >= 0.0 and
            (v+w) <= 1.0
        )


# --- SPHERE


class Sphere3:

    __slots__= ('center', 'radius', 'TYPE')

    def __init__(
        self,
        center: maths.Vec3= maths.Vec3(),
        radius: float= 1.0
    ) -> None:
        self.center: maths.Vec3= center
        self.radius: float= radius
        self.TYPE: GeometryType= GeometryType.SPHERE

    def __hash__(self) -> int:
        data: tuple[float]= (
            self.center.x, self.center.y, self.center.z,
            self.radius
        )
        return hash(data)

    def __eq__(self, other: 'Sphere3') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.center.is_equil(other.center) and
                maths.is_equil(self.radius, other.radius) and
                self.TYPE == other.TYPE
            ):
                return True
        return False

    def area(self) -> float:
        """Return area
        """
        return maths.PI * maths.sqr(self.radius)

    def copy(self) -> 'Sphere3':
        """Return a copy of self
        """
        return Sphere3(self.center.copy(), self.radius)

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        point: maths.Vec3= pt - self.center
        
        if not point.is_unit():
            point.to_unit()

        return self.center + (point * self.radius)

    def furthest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        if not pt.is_unit():
            pt.to_unit()
        
        return self.center + (pt * self.radius)

    def intersect_sphere(self, other: 'Sphere3') -> bool:
        dis: float= (self.center - other.center).length_sqr()
        r2: float= maths.sqr(self.radius + other.radius)

        return dis < r2

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        dis: float= (self.center - pt).length_sqr()
        r2: float= maths.sqr(self.radius)

        return dis < r2

    def intersect_aabb(self, aabb: 'AABB3') -> bool:
        close_pt: maths.Vec3= aabb.closest_pt(self.center)
        dis: float= (self.center - close_pt).length_sqr()
        r2: float= maths.sqr(self.radius)

        return dis < r2

    def intersect_plain(self, plain: 'Plane3'):
        close_pt: maths.Vec3= plain.closest_pt(self.center)
        dis: float= (self.center - close_pt).length_sqr()
        r2: float= maths.sqr(self.radius)

        return dis < r2


# --- PLAIN


class PlainError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class Plane3:

    __slots__= ('normal', 'direction', 'TYPE')

    def __init__(
        self,
        normal: maths.Vec3= maths.Vec3(x= 1.0),
        direction: float= 0.0
    ) -> None:
        self.normal: maths.Vec3= normal
        self.direction: float= direction
        self.TYPE: GeometryType= GeometryType.PLAIN

    def __hash__(self) -> int:
        data: tuple[float]= (
            self.normal.x, self.normal.y, self.normal.z,
            self.direction
        )
        return hash(data)

    def __eq__(self, other: 'Plane3') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.normal.is_equil(other.normal) and
                maths.is_equil(self.direction, other.direction) and
                self.TYPE == other.TYPE 
            ):
                return True
        return False

    def __post_init__(self):
        if not self.normal.is_unit():
            self.normal.to_unit()

    @staticmethod
    def create_from_v4(v4: maths.Vec4) -> 'Plane3':
        n: maths.Vec3= v4.xyz()
        w: float= v4.w
        if not n.is_unit():
            n.to_unit()
        return Plane3(n, v4.w)

    @staticmethod
    def create_from_normal_and_point(unit_v3: maths.Vec3, pt: maths.Vec3):
        if not unit_v3.is_unit():
            unit_v3.to_unit()

        n: maths.Vec3= unit_v3.copy()
        d: float= n.dot(pt)

        return Plane3(normal=n, direction=d)

    @staticmethod
    def create_from_points(
        a: maths.Vec3,
        b: maths.Vec3,
        c: maths.Vec3
    ) -> 'Plane3':
        v0: maths.Vec3= b - a
        v1: maths.Vec3= c - a

        n: maths.Vec3= v0.cross(v1)
        if not n.is_unit():
            n.to_unit()

        d: float= n.dot(a)

        return Plane3(normal= n, direction= d)

    def __str__(self) -> str:
        return f'''
        NORMAL: (X: {self.normal.x}, Y: {self.normal.y}, Z: {self.normal.z})
        DIRECTION: ({self.direction})
        '''
    def copy(self) -> 'Plane3':
        """Return a copy of self
        """
        return Plane3(
            normal= self.normal.copy(),
            direction= self.direction
        )

    def dot(self, v4: maths.Vec4) -> float:
        x: float= self.normal.x * v4.x
        y: float= self.normal.y * v4.y
        z: float= self.normal.z * v4.z
        w: float= self.direction * v4.w

        return x + y + z + w

    def dot_normal(self, v3: maths.Vec3) -> float:
        x: float= self.normal.x * v3.x
        y: float= self.normal.y * v3.y
        z: float= self.normal.z * v3.z

        return x + y + z

    def closest_point(self, point: maths.Vec3) -> maths.Vec3:
        scale: float= (self.normal.dot(point) - self.direction) / self.normal.length_sqr()
        return point - (self.normal * scale)

    def project_point_onto_plain(self, pt: maths.Vec3) -> maths.Vec3:
        scale: float= pt.dot(self.normal) - self.direction
        return pt - (self.normal * scale)

    def unit(self) -> 'Plane3':
        """Return a copy of self with unit length
        """
        len_sqr: float= self.normal.length_sqr()

        normal: maths.Vec3= maths.Vec3.create_unit_x()
        direction: float= 0.0

        if maths.is_one(len_sqr):
            normal.set_from(self.normal)
            direction= self.direction
        else:
            inv: float= maths.inv_sqrt(len_sqr)
            normal= self.normal * inv
            direction= self.direction * inv

        return Plane3(normal, direction)

    def to_unit(self) -> None:
        """Convert to unit length
        """
        len_sqr: float= self.normal.length_sqr()
        
        if maths.is_zero(len_sqr):
            return

        inv: float= maths.inv_sqrt(len_sqr)
        self.normal.x *= inv
        self.normal.y *= inv
        self.normal.z *= inv
        self.direction *= inv

    def pt_from_intersect_plane(self, other: 'Plane3') -> maths.Vec3:
        d: maths.Vec3= self.normal.cross(other.nomal)
        denom: float= d.length_sqr()

        if denom < maths.EPSILON:
            return maths.Vec3.zero()
        
        pt: maths.Vec3= (other.normal * self.direction) - (self.normal * other.direction)
        
        return pt.cross(d) * (1.0 / denom)

    def pt_from_intersect_planes(self, p2: 'Plane3', p3: 'Plane3') -> maths.Vec3:
        u: maths.Vec3= p2.normal.cross(p3.normal)
        denom: float= self.normal.dot(u)

        if maths.absf(denom) < maths.EPSILON:
            return maths.Vec3.zero()
        
        pt: maths.Vec3= (p2.normal * p3.direction) - (p3.normal * p2.direction)
        
        return (u * self.direction) + self.normal.cross(pt) * (1.0 / denom)

    def intersect_pane(self, other: 'Plane3') -> bool:
        dis: float= (self.normal.cross(other.nomal)).length_sqr()
        return not maths.is_zero(dis)

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        dot: float= pt.dot(self.normal)
        return maths.is_zero(dot - self.direction)

    def intersect_sphere(self, sph: 'Sphere3') -> bool:
        close_pt: maths.Vec3= self.closest_point(sph.center)
        len_sq: float= (sph.center - close_pt).length_sqr()

        return len_sq < maths.sqr(sph.radius)

    def intersect_aabb(self, aabb: 'AABB3') -> bool:
        len_sq: float = (
            aabb.extents.x * maths.absf(self.normal.x) +
            aabb.extents.y * maths.absf(self.normal.y) +
            aabb.extents.z * maths.absf(self.normal.z)
        )

        dot: float= self.normal.dot(aabb.center)
        dis: float= dot - self.direction

        return maths.absf(dis) <= len_sq


# --- FRUSTUM


class Frustum:

    __slots__=(
        'top',
        'bottom',
        'left',
        'right',
        'near',
        'far',
        'TYPE',
    )

    def __init__(
        self,
        top: Plane3= Plane3(),
        bottom: Plane3= Plane3(),
        left: Plane3= Plane3(),
        right: Plane3= Plane3(),
        near: Plane3= Plane3(),
        far: Plane3= Plane3()
    ) -> None:
        self.top: Plane3= top
        self.bottom: Plane3= bottom
        self.left: Plane3= left
        self.right: Plane3= right
        self.near: Plane3= near
        self.far: Plane3= far
        self.TYPE = GeometryType.FRUSTUM

    def __hash__(self) -> int:
        data: tuple[float]= (
            self.top.normal.x, self.top.normal.y, self.top.normal.z,
            self.top.direction,
            self.bottom.normal.x, self.bottom.normal.y, self.bottom.normal.z,
            self.bottom.direction,
            self.left.normal.x, self.left.normal.y, self.left.normal.z,
            self.left.direction,
            self.right.normal.x, self.right.normal.y, self.right.normal.z,
            self.right.direction,
            self.near.normal.x, self.near.normal.y, self.near.normal.z,
            self.near.direction,
            self.far.normal.x, self.far.normal.y, self.far.normal.z,
            self.far.direction,
        )
        return hash(data)

    def __eq__(self, other: 'Frustum') -> bool:
        if isinstance(other, self.__class__):
            if( 
                self.top == other.top and
                self.bottom == other.bottom and 
                self.left == other.left and
                self.right == other.right and
                self.near == other.near and
                self.far == other.far and
                self.TYPE == other.TYPE
            ):
                return True
        return False

    def __str__(self) -> str:
        return f'''
        TOP: {self.top}
        BOTTOM: {self.bottom}
        LEFT: {self.left}
        RIGHT: {self.right}
        NEAR: {self.near}
        FAR: {self.far}
        '''

# --- RAY3D


class Ray3:

    __slots__= ('origin', 'direction', 'TYPE')

    def __init__(
        self,
        origin: maths.Vec3= maths.Vec3(),
        direction: maths.Vec3= maths.Vec3()
    ) -> None:
        self.origin: maths.Vec3= origin
        self.direction: maths.Vec3= direction
        self.TYPE: GeometryType= GeometryType.RAY

        if not self.direction.is_unit():
            self.direction.to_unit()

    def __hash__(self) -> int:
        data: tuple[float]= (
            self.origin.x, self.origin.y, self.origin.z,
            self.direction.x, self.direction.y, self.direction.z,
        )
        return hash(data)

    def __eq__(self, other: 'Ray3') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.origin.is_equil(other.origin) and
                self.direction.is_equil(other.direction) and
                self.TYPE == other.TYPE 
            ):
                return True
        return False

    @staticmethod
    def from_points(origin: maths.Vec3, target: maths.Vec3) -> 'Ray3':
        o: maths.Vec3= origin.copy()
        d: maths.Vec3= target - origin

        if not d.is_unit():
            d.to_unit()

        return Ray3(
            origin= o,
            direction= d
        )

    def set_direction(self, unit_dir: maths.Vec3) -> None:
        if unit_dir.is_zero():
            return

        if not unit_dir.is_unit():
            unit_dir.to_unit()

        self.direction.x = unit_dir.x
        self.direction.y = unit_dir.y

    def copy(self) -> 'Ray3':
        return Ray3(
            origin= self.origin.copy(),
            direction= self.direction.copy()
        )

    def get_hit(self, t: float) -> maths.Vec3:
        """Return point along ray
        """
        return self.origin + (self.direction * t)

    def cast_aabb(self, aabb: AABB3) -> float:
        pmin: maths.Vec3= aabb.get_min()
        pmax: maths.Vec3= aabb.get_max()
        tmin: float= maths.MIN_FLOAT
        tmax: float= maths.MAX_FLOAT

        for idx in range(3):
            if maths.is_zero(self.direction.get_at(idx)):
                if(
                    self.origin.get_at(idx) < pmin.get_at(idx) or 
                    self.origin.get_at(idx) > pmax.get_at(idx)
                ):
                    return -1.0
            else:
                inv: float= 1.0 / self.direction.get_at(idx)

                t1: float= (pmin.get_at(idx) - self.origin.get_at(idx)) * inv
                t2: float= (pmax.get_at(idx) - self.origin.get_at(idx)) * inv

                if t1 > t2:
                    t1, t2= maths.swapf(t1, t2)

                if t1 > tmin:
                    tmin= t1

                if t2 > tmax:
                    tmax= t2

                if tmin > tmax:
                    return -1.0

        return tmin

    def cast_sphere(self, sph: Sphere3) -> float:
        a: maths.Vec3= sph.center - self.origin
        b: float= a.dot(self.direction)
        c: float= a.length_sqr() - maths.sqr(sph.radius)

        if c > 0.0 and b > 0.0:
            return -1.0

        d: float= maths.sqr(b) - c
        if d < 0.0:
            return -1.0

        t: float= -b - maths.sqrt(d)
        if t < 0.0:
            t= 0.0

        return t

    def cast_plain(self, pl: Plane3) -> float:
        nd: float= self.direction.dot(pl.normal)
        pn: float= self.origin.dot(pl.normal)

        if nd >= 0.0:
            return -1.0

        t: float= (pl.direction - pn) / nd

        if t >= 0.0:
            return t

        return -1.0

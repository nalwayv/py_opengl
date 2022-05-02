"""Geometry
"""
from dataclasses import dataclass
# from typing import Final, Optional
from enum import Enum, auto
# from abc import ABC, abstractmethod

from py_opengl import maths


# --- ENUM


class GeometryID(Enum):
    LINE= auto()
    AABB3= auto()
    SPHERE= auto()
    PLAIN= auto()
    RAY= auto()


# ---


# class ITranslate(ABC):
#     @abstractmethod
#     def translate(v3: maths.Vec3):
#         pass

# class IRotate(ABC):
#     @abstractmethod
#     def rotate(angle_deg: float, unit_axis: maths.Vec3):
#         pass

# class ITransformable(ITranslate, IRotate):
#     pass


# --- LINE


@dataclass(eq= False, repr= False, slots= True)
class Line:
    start: maths.Vec3= maths.Vec3()
    end: maths.Vec3= maths.Vec3()
    id: GeometryID= GeometryID.LINE

    def length(self) -> float:
        return self.start.distance_sqrt(self.end)


# --- AABB


@dataclass(eq= False, repr= True, slots= True)
class AABB3:
    """AABB using center and extents
    

    Aabb(center= Vec3(0, 0, 0), extents= Vec3(1, 1, 1))

    get_min == Vec3(-1, -1, -1)\n
    get_max == Vec3(1, 1, 1)
    
    """
    center: maths.Vec3= maths.Vec3()
    extents: maths.Vec3= maths.Vec3()

    id: GeometryID= GeometryID.AABB3

    @staticmethod
    def from_min_max(min_pt: maths.Vec3, max_pt: maths.Vec3) -> 'AABB3':
        return AABB3(
            center= (min_pt + max_pt) * 0.5,
            extents= ((min_pt - max_pt) * 0.5).abs()
        )

    def union(self, other: 'AABB3') -> 'AABB3':
        return AABB3.from_min_max(
            maths.Vec3.from_min(self.get_min(), other.get_min()),
            maths.Vec3.from_max(self.get_max(), other.get_max())
        )

    def union_one(self, a: 'AABB3') -> None:
        """Set self to the union of self and a"""
        aabb= AABB3.from_min_max(
            maths.Vec3.from_min(self.get_min(), a.get_min()),
            maths.Vec3.from_max(self.get_max(), a.get_max())
        )
        self.center= aabb.center
        self.extents= aabb.extents

    def union_two(self, a:'AABB3', b:'AABB3') -> None:
        """Set self to the union of a and b"""
        aabb= AABB3.from_min_max(
            maths.Vec3.from_min(a.get_min(), b.get_min()),
            maths.Vec3.from_max(a.get_max(), b.get_max())
        )
        self.center= aabb.center
        self.extents= aabb.extents

    def expand(self, by: float) -> 'AABB3':
        if by < 0.0:
            by = maths.absf(by)

        p0: maths.Vec3= self.get_min() - maths.Vec3(by,by,by)
        p1: maths.Vec3= self.get_max() + maths.Vec3(by,by,by)

        return AABB3.from_min_max(p0, p1)

    def get_size_x(self) -> float:
        p0= self.get_min()
        p1= self.get_max()
        return p1.x - p0.x

    def get_size_y(self) -> float:
        p0= self.get_min()
        p1= self.get_max()
        return p1.y - p0.y

    def get_size_z(self) -> float:
        p0= self.get_min()
        p1= self.get_max()
        return p1.z - p0.z

    def copy(self) -> 'AABB3':
        """Return a copy of self

        Returns
        ---
        Abbb
        """
        return AABB3(
            center= self.center.copy(),
            extents= self.extents.copy()
        )

    def perimeter(self) -> float:
        p0: maths.Vec3= self.get_min()
        p1: maths.Vec3= self.get_max()

        p2: maths.Vec3= p1 - p0

        return 4.0 * p2.sum()

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        p0: maths.Vec3= self.get_min()
        p1: maths.Vec3= self.get_max()
        p2: list[float]= [0.0, 0.0, 0.0]

        for i in range(3):
            val: float= pt[i]
            val= maths.minf(val, p0[i])
            val= maths.maxf(val, p1[i])
            p2[i]= val

        return maths.Vec3(p2[0], p2[1], p2[2])

    def get_min(self) -> maths.Vec3:
        p0: maths.Vec3= self.center + self.extents
        p1: maths.Vec3= self.center - self.extents

        return maths.Vec3.from_min(p0, p1)

    def get_max(self) -> maths.Vec3:
        p0: maths.Vec3= self.center + self.extents
        p1: maths.Vec3= self.center - self.extents

        return maths.Vec3.from_max(p0, p1)

    def intersect_aabb(self, other: 'AABB3') -> bool:
        amin: maths.Vec3= self.get_min()
        amax: maths.Vec3= self.get_max()

        bmin: maths.Vec3= other.get_min()
        bmax: maths.Vec3= other.get_max()

        check_x: bool= amin.x <= bmax.x and amax.x >= bmin.x
        check_y: bool= amin.y <= bmax.y and amax.y >= bmin.y
        check_z: bool= amin.z <= bmax.z and amax.z >= bmin.z

        return check_x and check_y and check_z

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        amin: maths.Vec3= self.get_min()
        amax: maths.Vec3= self.get_max()

        if pt.x < amin.x or pt.y < amin.y or pt.z < amin.z:
            return False

        if pt.x > amax.x or pt.y > amax.y or pt.z > amax.z:
            return False

        return True

    def intersect_sphere(self, sph: 'Sphere3') -> bool:
        close_pt: maths.Vec3= self.closest_pt(sph.center)
        dis: float= (sph.center - close_pt).length_sqr()
        r2= maths.sqr(sph.radius)

        return dis < r2

    def intersect_plain(self, plain: 'Plain') -> bool:
        len_sq: float = (
            self.extents.x * maths.absf(plain.normal.x) +
            self.extents.y * maths.absf(plain.normal.y) +
            self.extents.z * maths.absf(plain.normal.z)
        )

        dot: float= plain.normal.dot(self.center)
        dis: float= dot - plain.direction

        return maths.absf(dis) <= len_sq



# --- SPHERE



@dataclass(eq= False, repr= False, slots= True)
class Sphere3:
    center: maths.Vec3= maths.Vec3()
    radius: float= 1.0
    id: GeometryID= GeometryID.SPHERE

    def __hash__(self):
        return hash((self.radius, self.center.x, self.center.y, self.center.z))
    
    def __eq__(self, other: 'Sphere3') -> bool:
        check_r= maths.is_equil(self.radius, other.radius)
        check_p= self.center.is_equil(other.center)
        check_id= self.id == other.id
        return check_r and check_p and check_id

    def compute_aabb(self) -> AABB3:
        p0: maths.Vec3= self.center + maths.Vec3(self.radius, self.radius, self.radius)
        p1: maths.Vec3= self.center - maths.Vec3(self.radius, self.radius, self.radius)
        return AABB3.from_min_max(p0, p1)

    def area(self) -> float:
        """Return area

        Returns
        ---
        float
        """
        return maths.PI * maths.sqr(self.radius)

    def copy(self) -> 'Sphere3':
        """Return a copy of self

        Returns
        ---
        Sphere
        """
        return Sphere3(self.center.copy(), self.radius)

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        point: maths.Vec3= pt - self.center
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

    def intersect_plain(self, plain: 'Plain'):
        close_pt: maths.Vec3= plain.closest_pt(self.center)
        dis: float= (self.center - close_pt).length_sqr()
        r2: float= maths.sqr(self.radius)

        return dis < r2



# --- PLAIN



class PlainError(Exception):
    '''Custom error for plain'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Plain:
    normal: maths.Vec3= maths.Vec3(x=1.0)
    direction: float= 0.0
    id: GeometryID= GeometryID.PLAIN

    def __hash__(self):
        return hash((self.direction, self.normal.x, self.normal.y, self.normal.z))
    
    def __eq__(self, other: 'Plain') -> bool:
        check_d= maths.is_equil(self.direction, other.direction)
        check_n= self.normal.is_equil(other.normal)
        check_id= self.id == other.id
        return check_d and check_n and check_id

    def __post_init__(self):
        if not self.normal.is_unit():
            self.normal.to_unit()

    @staticmethod
    def from_points(a: maths.Vec3, b: maths.Vec3, c: maths.Vec3) -> 'Plain':
        v0: maths.Vec3= b - a
        v1: maths.Vec3= c - a
        
        normal: maths.Vec3= v0.cross(v1)
        if not normal.is_unit():
            normal.to_unit()

        direction: float= normal.dot(a)
        
        return Plain(normal, direction)

    def copy(self) -> 'Plain':
        """Return a copy of self

        Returns
        ---
        Plain
        """
        return Plain(
            normal= self.normal.copy(),
            direction= self.direction
        )

    def dot(self, v4: maths.Vec4) -> float:
        x: float= self.normal.x * v4.x
        y: float= self.normal.y * v4.y
        z: float= self.normal.z * v4.z
        w: float= self.direction * v4.w

        return x + y + z + w

    def dot_coord(self, v3: maths.Vec3) -> float:
        x: float= self.normal.x * v3.x
        y: float= self.normal.y * v3.y
        z: float= self.normal.z * v3.z
        w: float= self.direction

        return x + y + z + w

    def dot_normal(self, v3: maths.Vec3) -> float:
        x: float= self.normal.x * v3.x
        y: float= self.normal.y * v3.y
        z: float= self.normal.z * v3.z

        return x + y + z

    def closest_point(self, point: maths.Vec3) -> maths.Vec3:
        scale: float= (self.normal.dot(point) - self.direction) / self.normal.length_sqr()
        return point - (self.normal * scale)

    def unit(self) -> 'Plain':
        """Return a copy of self that has been normalized

        Returns
        ---
        Plain
        """
        len_sqr: float= self.normal.length_sqr()

        normal: maths.Vec3= maths.Vec3.unit_x()
        direction: float= 0.0

        if maths.is_one(len_sqr):
            normal.x= self.normal.x
            normal.y= self.normal.y
            normal.z= self.normal.z
            direction= self.direction
        else:
            inv: float= maths.inv_sqrt(len_sqr)

            normal= self.normal * inv
            direction= self.direction * inv

        return Plain(normal, direction)

    def to_unit(self) -> None:
        """Normalize the length of self
        """
        len_sqr: float= self.normal.length_sqr()
        if maths.is_zero(len_sqr):
            raise PlainError('length of plain was zero')

        inv: float= maths.inv_sqrt(len_sqr)

        self.normal.x *= inv
        self.normal.y *= inv
        self.normal.z *= inv
        self.direction *= inv

    def intersect_plain(self, other: 'Plain') -> bool:
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



# --- RAY3D



@dataclass(eq=False, repr=False, slots=True)
class Ray3:
    origin: maths.Vec3= maths.Vec3()
    direction: maths.Vec3= maths.Vec3(z=1.0)
    id: GeometryID= GeometryID.RAY

    def __post_init__(self):
        if not self.direction.is_unit():
            self.direction.to_unit()

    @staticmethod
    def from_points(origin: maths.Vec3, target:maths.Vec3) -> 'Ray3':
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

        Parameters
        ---
        t : float

        Returns
        ---
        Vec3
        """
        return self.origin + (self.direction * t)
    
    def cast_aabb(self, aabb: AABB3) -> tuple[bool, maths.Vec3]:
        amin: maths.Vec3= aabb.get_min()
        amax: maths.Vec3= aabb.get_max()
        tmin: float= maths.MIN_FLOAT
        tmax: float= maths.MAX_FLOAT

        for idx in range(3):
            if maths.is_zero(self.direction[idx]):
                if self.origin[idx] < amin[idx] or self.origin[idx] > amax[idx]:
                    return False, maths.Vec3.zero()
            else:
                inv: float= 1.0 / self.direction[idx]

                t1: float= (amin[idx] - self.origin[idx]) * inv
                t2: float= (amax[idx] - self.origin[idx]) * inv

                if t1 > t2:
                    t1, t2= t2, t1
        
                if t1 > tmin:
                    tmin= t1

                if t2 > tmax:
                    tmax= t2

                if tmin > tmax:
                    return False, maths.Vec3.zero()
    
        return True, self.get_hit(tmin)

    def cast_sphere(self, sph: Sphere3) -> tuple[bool, maths.Vec3]:
        a: maths.Vec3= sph.center - self.origin
        b: float= a.dot(self.direction)
        c: float= a.length_sqr() - maths.sqr(sph.radius)

        if c > 0.0 and b > 0.0:
            return False, maths.Vec3.zero()

        d: float= maths.sqr(b) - c
        if d < 0.0:
            return False, maths.Vec3.zero()
        
        t: float= -b - maths.sqrt(d)
        if t < 0.0:
            t= 0.0

        return True, self.get_hit(t)

    def cast_plain(self, pl: Plain) -> tuple[bool, maths.Vec3]:
        nd: float= self.direction.dot(pl.normal)
        pn: float= self.origin.dot(pl.normal)
 
        if nd >= 0.0:
            return False, maths.Vec3.zero()
        
        t: float= (pl.direction - pn) / nd
        
        if t >= 0.0:
            return False, maths.Vec3.zero()

        return True, self.get_hit(t)


# # ---
# MARGIN: Final[float] = 2.0

# @dataclass(eq= False, repr= False, slots= True)
# class AABBNode:
#     left: Optional['AABBNode']= None
#     right: Optional['AABBNode']= None
#     parent: Optional['AABBNode']= None
#     height: int= 0
#     aabb: AABB3= AABB3()

#     todo_item: int= 0

#     def is_leaf(self) -> bool:
#         return self.left is None

# @dataclass(eq= False, repr= False, slots= True)
# class AABBTree:
#     root: Optional[AABBNode]= None
#     leaves: dict[Sphere, AABBNode]= field(default_factory=dict)
#     aabb: AABB3= AABB3()

#     def add(self, obj: Sphere):
#         # TODO
#         self.aabb= obj.compute_aabb()
#         self.aabb.expand(MARGIN)

#         node= AABBNode()
#         node.aabb.union_one(self.aabb)

#         self.leaves[obj] = node
#         self._insert(node)

#     def _balance(self, item: AABBNode) -> AABBNode:
#         a= item

#         if a.is_leaf() or a.height < 2:
#             return a

#         b= a.left
#         c= a.right

#         balance= c.height - b.height

#         # rotate c up
#         if balance > 1:
#             f= c.left
#             g= c.right

#             # swap
#             c.left= a
#             c.parent= a.parent
#             a.parent= c

#             if c.parent is not None:
#                 if c.parent.left is a:
#                     c.parent.left = c
#                 else:
#                     c.parent.right = c
#             else:
#                 self.root = c

#             if f.height > g.height:
#                 c.right= f
#                 a.right= g
#                 g.parent= a

#                 a.aabb.union_two(b.aabb, g.aabb)
#                 c.aabb.union_two(a.aabb, f.aabb)

#                 a.height = 1 + maths.maxi(b.height, g.height)
#                 c.height = 1 + maths.maxi(a.height, f.height)
#             else:
#                 c.right= g
#                 a.right= f
#                 f.parent= a

#                 a.aabb.union_two(b.aabb, f.aabb)
#                 c.aabb.union_two(a.aabb, g.aabb)

#                 a.height = 1 + maths.maxi(b.height, f.height)
#                 c.height = 1 + maths.maxi(a.height, g.height)

#             return c

#         # rotate b up
#         if balance < -1:
#             d= b.left
#             e= b.right

#             # swap
#             b.left= a
#             b.parent= a.parent
#             a.parent= b

#             if b.parent is not None:
#                 if b.parent.left is a:
#                     b.parent.left = b
#                 else:
#                     b.parent.right = b
#             else:
#                 self.root = b

#             if d.height > e.height:
#                 b.right= d
#                 a.left= e
#                 e.parent= a

#                 a.aabb.union_two(c.aabb, e.aabb)
#                 b.aabb.union_two(a.aabb, d.aabb)

#                 a.height = 1 + maths.maxi(c.height, e.height)
#                 b.height = 1 + maths.maxi(a.height, d.height)
#             else:
#                 b.right= e
#                 a.left= d
#                 d.parent= a

#                 a.aabb.union_two(c.aabb, d.aabb)
#                 b.aabb.union_two(a.aabb, e.aabb)

#                 a.height = 1 + maths.maxi(c.height, d.height)
#                 b.height = 1 + maths.maxi(a.height, e.height)

#             return b

#         return a

#     def _insert(self, item: AABBNode) -> None:
#         if self.root is None:
#             self.root = item
#             return

#         tmp= AABB3()
#         item_aabb= item.aabb

#         node= self.root
#         while not node.is_leaf():
#             aabb= node.aabb
#             per= aabb.perimeter()
            
#             tmp= aabb
#             union_per= tmp.union(item_aabb).perimeter()

#             cost= 2.0 * union_per
#             d_cost = 2.0 * (union_per - per)

#             left= node.left
#             right= node.right

#             cost_left= 0.0
#             if left.is_leaf():
#                 tmp.union_two(left.aabb, item_aabb)
#                 cost_left= tmp.perimeter() + d_cost
#             else:
#                 old_cost= left.aabb.perimeter()
#                 tmp.union_two(left.aabb, item_aabb)
#                 new_cost= tmp.perimeter()
#                 cost_left = new_cost - old_cost + d_cost

#             cost_right= 0.0
#             if right.is_leaf():
#                 tmp.union_two(right.aabb, item_aabb)
#                 cost_right= tmp.perimeter() + d_cost
#             else:
#                 old_cost= right.aabb.perimeter()
#                 tmp.union_two(right.aabb, item_aabb)
#                 new_cost= tmp.perimeter()
#                 cost_right = new_cost - old_cost + d_cost

#             if cost < cost_left and cost < cost_right:
#                 break
#             if cost_left < cost_right:
#                 node= left
#             else:
#                 node= right
        
#         parent= node.parent
#         new_parent= AABBNode(
#             parent= node.parent,
#             aabb= node.aabb.union(item_aabb),
#             height= node.height + 1
#         )

#         if parent is not None:
#             if parent.left is node:
#                 parent.left= new_parent
#             else:
#                 parent.right= new_parent

#             new_parent.left= node
#             new_parent.right= item
#             node.parent= new_parent
#             item.parent= new_parent
#         else:
#             new_parent.left= node
#             new_parent.right= item
#             node.parent= new_parent
#             item.parent= new_parent
#             self.root= new_parent
        

#         node= item.parent
#         while node is not None:
#             node= self._balance(node)

#             left= node.left
#             right= node.right

#             node.height = 1 + maths.maxi(left.height, right.height)
#             node.aabb.union_two(left.aabb, right.aabb)
#             node= node.parent

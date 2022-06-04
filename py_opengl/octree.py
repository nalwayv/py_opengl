"""Octree
"""
from typing import TypeVar, Final

from py_opengl import geometry
from py_opengl import maths
from py_opengl import model
from py_opengl import shader
from py_opengl import camera


# ---


T= TypeVar('T', bound= model.Model)
MAX_OBJS: Final[int]= 3
PADDING: Final[float]= 1.0


# ---


class Node:

    __slots__= (
        'aabb',
        'children',
        'objs',
        'min_size',
        'max_size',
        'center'
    )
    
    def __init__(self) -> None:
        self.aabb: geometry.AABB3= geometry.AABB3()
        self.children: list['Node'|None]= []
        self.objs: list[T|None]= []
        self.min_size: float= 0.0
        self.max_size: float= 0.0
        self.center: maths.Vec3= maths.Vec3()

    @staticmethod
    def create_from_values(
        max_size: float,
        min_size: float,
        center: maths.Vec3
    ) -> 'Node':
        node= Node()
        node.max_size= max_size
        node.min_size= min_size
        node.center= center
        node.aabb= geometry.AABB3(
            center,
            maths.Vec3.create_from_value(PADDING * max_size)
        )
        return node

    def has_children(self) -> bool:
        return len(self.children) != 0

    def has_objs(self) -> bool:
        return len(self.objs) != 0

    def best_fit(self, v3: maths.Vec3) -> int:
        result: int= 0
        if v3.x <= self.center.x:
            result += 1
        if v3.y >= self.center.y:
            result += 4
        if v3.z >= self.center.z:
            result += 2
        return result


# ---


class Octree:

    __slots__= ('root', )

    def __init__(self, max_size: float, min_size: float, position: maths.Vec3) -> None:
        self.root= Node.create_from_values(
            max_size,
            min_size,
            position
        )

    def _split(self, node: Node) -> None:
        quarter: float= node.max_size * 0.25
        half: float= node.max_size * 0.5
        min_size: float= node.min_size
        center: maths.Vec3= node.center

        node.children= [None]*8
        node.children[0]= Node.create_from_values(
            half,
            min_size,
            center + maths.Vec3(-quarter, quarter, -quarter)
        )
        node.children[1]= Node.create_from_values(
            half,
            min_size,
            center + maths.Vec3(quarter, quarter, -quarter)
        )
        node.children[2]= Node.create_from_values(
            half,
            min_size,
            center + maths.Vec3(-quarter, quarter, quarter)
        )
        node.children[3]= Node.create_from_values(
            half,
            min_size,
            center + maths.Vec3(quarter, quarter, quarter)
        )
        node.children[4]= Node.create_from_values(
            half,
            min_size,
            center + maths.Vec3(-quarter, -quarter, -quarter)
        )
        node.children[5]= Node.create_from_values(
            half,
            min_size,
            center + maths.Vec3(quarter, -quarter, -quarter)
        )
        node.children[6]= Node.create_from_values(
            half,
            min_size,
            center + maths.Vec3(-quarter, -quarter, quarter)
        )
        node.children[7]= Node.create_from_values(
            half,
            min_size,
            center + maths.Vec3(quarter, -quarter, quarter)
        )

    def _should_merge(self, node: Node) -> bool:
        count:int= len(node.objs)
        if node.children:
            for child in node.children:
                if child.children:
                    return False
                count += len(child.objs)
        return count <= MAX_OBJS

    def _merge(self, node: Node) -> None:
        if not node.has_children():
            return

        for child in node.children:
            for n in range(len(child.objs)-1, 0, -1):
                obj= node.objs[n]
                node.objs.append(obj)
        node.children.clear()

    def _has_objects(self, node: Node) -> bool:
        if node.has_objs():
            return True
        for child in node.children:
            if self._has_objects(child):
                return True
        return False

    def _grow(self, v3: maths.Vec3) -> None:
        xd: int= 1 if v3.x >= 0 else -1
        yd: int= 1 if v3.y >= 0 else -1
        zd: int= 1 if v3.z >= 0 else -1
        old_root= self.root
        half: float= self.root.max_size * 0.5
        new_max_size: float= self.root.max_size * 2.0
        new_min_size: float= self.root.min_size
        new_center: maths.Vec3= self.root.center + maths.Vec3(xd*half, yd*half, zd*half)

        self.root= Node.create_from_values(new_max_size, new_min_size, new_center)
        if self._has_objects(old_root):
            bf: int= self.root.best_fit(old_root.aabb.center)
            children=[None]*8
            for i in range(8):
                if i == bf:
                    children[i]= old_root
                else:
                    xd= -1 if (i % 2 == 0) else 1
                    yd= -1 if (i > 3) else 1
                    zd= -1 if (i < 2 or (i > 3 and i < 6)) else 1
                    children[i]= Node.create_from_values(
                        old_root.max_size,
                        old_root.min_size,
                        new_center + maths.Vec3(xd*half, yd*half, zd*half)
                    )
            self.root.children= children

    def _add_obj(self, node: Node, obj:T, obj_aabb: geometry.AABB3):
        if not node.has_children():
            if(len(node.objs) < MAX_OBJS or (node.max_size * 0.5) < node.min_size):
                node.objs.append(obj)
                return

            if not node.has_children():
                self._split(node)
                if not node.children:
                    raise Exception('error octree')

                for n in range(len(node.objs) - 1, 0, -1):
                    current: T|None= node.objs[n]
                    current_aabb= current.compute_aabb()
                    bf: int= node.best_fit(current_aabb.center)

                    if node.children[bf].aabb.contains_aabb(current_aabb):
                        self._add_obj(node.children[bf], current, current_aabb)
                        node.objs.remove(current)

        bf: int= node.best_fit(obj_aabb.center)
        if node.children[bf].aabb.contains_aabb(obj_aabb):
            self._add_obj(node.children[bf], obj, obj_aabb)
        else:
            node.objs.append(obj)

    def _add(self, obj: T, aabb: geometry.AABB3) -> bool:
        if not self.root.aabb.contains_aabb(aabb):
            return False
        self._add_obj(self.root, obj, aabb)
        return True

    def add(self, obj: T) -> None:
        tryes: int= 0
        aabb= obj.compute_aabb()
        while not self._add(obj, aabb):
            self._grow(aabb.center - self.root.aabb.center)
            tryes += 1
            if tryes > 20:
                raise Exception('error with add to octree')

    def _shrink(self, node: Node) -> Node:
        if node.max_size < (node.min_size * 2.0):
            return node

        if not node.has_objs() and not node.has_children():
            return node

        aabbs: list[geometry.AABB3|None]= [None]*8
        q: float = node.max_size * 0.25
        e: maths.Vec3=maths.Vec3.create_from_value((node.max_size * 0.5) * PADDING)
        aabbs[0]= geometry.AABB3(node.center + maths.Vec3(-q,  q, -q), e)
        aabbs[1]= geometry.AABB3(node.center + maths.Vec3( q,  q, -q), e)
        aabbs[2]= geometry.AABB3(node.center + maths.Vec3(-q,  q,  q), e)
        aabbs[3]= geometry.AABB3(node.center + maths.Vec3( q,  q,  q), e)
        aabbs[4]= geometry.AABB3(node.center + maths.Vec3(-q, -q, -q), e)
        aabbs[5]= geometry.AABB3(node.center + maths.Vec3( q, -q, -q), e)
        aabbs[6]= geometry.AABB3(node.center + maths.Vec3(-q, -q,  q), e)
        aabbs[7]= geometry.AABB3(node.center + maths.Vec3( q, -q,  q), e)

        bf: int= -1
        for i, obj in enumerate(node.objs):
            obj_aabb= obj.compute_aabb()
            new_bf: int= node.best_fit(obj_aabb.center)

            if i == 0 or new_bf == bf:
                if aabbs[new_bf].contains_aabb(obj_aabb):
                    if bf < 0:
                        bf= new_bf
                else:
                    return node
            else:
                return node

        if node.has_children():
            has_objs: bool= False
            for i, child in enumerate(node.children):
                if self._has_objects(child):
                    if has_objs:
                        return node

                    if bf >= 0 and bf != i:
                        return node

                    has_objs= True
                    bf= i

        if not node.has_children():
            new_max_size: float= node.max_size * 0.5
            new_center: maths.Vec3= node.children[bf].aabb.center
            new_aabb: geometry.AABB3= geometry.AABB3(   
                new_center,
                maths.Vec3.create_from_value(PADDING * new_max_size)
            )

            node.max_size= new_max_size
            node.center= new_center
            node.aabb= new_aabb
            return node

        if bf == -1:
            return node
        return node.children[bf]

    def _remove_obj(self, node: Node, obj: T, obj_aabb: geometry.AABB3):
        removed_item: bool= False
        for item in node.objs:
            if item is obj:
                node.objs.remove(item)
                removed_item= True
                break
        
        if not removed_item and node.children:
            bf: int= node.best_fit(obj_aabb.center)
            self._remove_obj(node.children[bf], obj, obj_aabb)
        
        if removed_item and node.children:
            if self._should_merge(node):
                self._merge(node)

    def _remove(self, obj: T) -> bool:
        aabb= obj.compute_aabb()
        if not self.root.aabb.contains_aabb(aabb):
            return False
        self._remove_obj(self.root, obj, aabb)
        return True

    def remove(self, obj: T) -> None:
        if self._remove(obj):
            self.root= self._shrink(self.root)

    def debug(self, s: shader.Shader, c: camera.Camera) -> None:
        que: list[Node|None]= [self.root]

        while que:
            current= que.pop(0)
            if current == None:
                continue

            if current.has_children():
                for child in current.children:
                    que.append(child)

            if current.has_objs():
                for obj in current.objs:
                    obj_aabb= obj.compute_aabb()
                    obj_aabb.expanded(0.1)
                    m= model.CubeModelAABB(obj_aabb)
                    m.draw(s, c, True)
                    m.delete()

            dbg_model= model.CubeModelAABB(current.aabb)
            dbg_model.draw(s, c, True)
            dbg_model.delete()

    def raycast(self,ray: geometry.Ray3) -> list[T|None]:
        result: list[T|None]= []
        que: list[Node|None]= [self.root]

        while que:
            current= que.pop(0)
            if current == None:
                continue
            if current.has_objs():
                for obj in current.objs:
                    aabb= obj.compute_aabb()
                    t= ray.cast_aabb(aabb)
                    if t >= 0:
                        result.append(obj)
            if current.has_children():
                for child in current.children:
                    que.append(child)

        return result
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
        return True if self.children else False

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

    def _should_combine(self, node: Node) -> bool:
        count:int= len(node.objs)
        if node.children:
            for child in node.children:
                if child.children:
                    return False
                count += len(child.objs)
        return count <= node.max_objs

    def _add_obj(self, node: Node, obj:T, obj_aabb: geometry.AABB3):
        if not node.has_children():
            if(
                len(node.objs) < MAX_OBJS or
                (node.max_size * 0.5) < node.min_size
            ):
                node.objs.append(obj)
                return

            if not node.children:
                self._split(node)

                n: int= len(node.objs)-1
                while n >= 0:
                    n -= 1
                    current: T|None= node.objs[n]
                    current_aabb= current.compute_aabb()
                    bf: int= node.best_fit(current_aabb.center)
                    child= node.children[bf]

                    if child.aabb.contains_aabb(current_aabb):
                        self._add_obj(child, current, current_aabb)
                        node.objs.remove(current)

        bf: int= node.best_fit(obj_aabb.center)
        child= node.children[bf]
        if child.aabb.contains_aabb(obj_aabb):
            self._add_obj(child, obj, obj_aabb)
        else:
            node.objs.append(obj)

    def _add(self, node: Node, obj: T) -> bool:
        aabb: geometry.AABB3= obj.compute_aabb()
        if not node.aabb.contains_aabb(aabb):
            return False
        self._add_obj(node, obj, aabb)
        return True

    def add(self, obj: T) -> bool:
        return self._add(self.root, obj)

    def _remove_obj(self, node: Node, obj: T, obj_aabb: geometry.AABB3):
        removed_item: bool= False
        for item in node.objs:
            if item is obj:
                node.objs.remove(item)
                removed_item= True
                break
        
        if not removed_item and node.children:
            bf: int= self._best_fit(node, obj_aabb.center)
            child= node.children[bf]
            self._remove_obj(child, obj, obj_aabb)
        
        if removed_item and node.children:
            if self._should_combine(node):
                for child in node.children:
                    n: int= len(child.objs)-1
                    while n >= 0:
                        n -= 1
                        obj= node.objs[n]
                        node.objs.append(obj)
                node.children.clear()

    def _remove(self, node: Node, obj: T) -> bool:
        aabb= obj.compute_aabb()
        if not node.aabb.contains_aabb(aabb):
            return False
        self._remove_obj(node, obj, aabb)
        return True

    def remove(self, obj: T) -> bool:
        return self._remove(self.root, obj)

    def raycast_hit(self, ray: geometry.Ray3) -> bool:
        if ray.cast_aabb(self.root.aabb) < 0:
            return False

        que: list[Node|None]= [self.root]
        while que:
            current: Node|None= que.pop(0)
            
            if current == None:
                continue

            if current.has_children():
                for child in current.children:
                    que.append(child)
            for obj in current.objs:
                if ray.cast_aabb(obj.compute_aabb()) >= 0:
                    return True
        return False

    def debug(self, s: shader.Shader, c: camera.Camera) -> None:
        que: list[Node|None]= [self.root]

        while que:
            current= que.pop(0)
            if current == None:
                continue

            if current.has_children():
                for child in current.children:
                    que.append(child)

            dbg_model= model.CubeModelAABB(current.aabb)
            dbg_model.draw(s, c, True)
            dbg_model.delete()
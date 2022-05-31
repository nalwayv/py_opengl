"""
"""
from typing import TypeVar

from py_opengl import geometry
from py_opengl import maths
from py_opengl import model
from py_opengl import camera
from py_opengl import shader


T= TypeVar('T', bound= model.Model)

# bvh

class Node0:
    
    __slots__= (
        'left',
        'right',
        'parent',
        'height',
        'aabb',
        'obj'
    )

    def __init__(self) -> None:
        self.left: 'Node0'|None= None
        self.right: 'Node0'|None= None
        self.parent: 'Node0'|None= None
        self.height: int= 0
        self.aabb: geometry.AABB3= geometry.AABB3()
        self.obj: T|None= None

    def is_leaf(self) -> bool:
        return self.left == None


class Tree0:
    
    __slots__= (
        'root',
        'leaves'
    )

    def __init__(self) -> None:
        self.root: Node0= Node0()
        self.leaves: dict[T, Node0] = {}

    def _balance_leaf(self, leaf: Node0) -> Node0:
        a: Node0= leaf
        if a.is_leaf() or a.height < 2:
            return a

        b: Node0= a.left
        c: Node0= a.right

        balance: int= c.height - b.height

        if balance > 1:
            f: Node0= c.left
            g: Node0= c.right

            c.left= a
            c.parent= a.parent
            a.parent= c

            if c.parent != None:
                if c.parent.left is a:
                    c.parent.left= c
                else:
                    c.parent.right= c
            else:
                self.root= c

            if f.height > g.height:
                c.right= f
                a.right= g
                g.parent= a

                a.aabb= geometry.AABB3.create_combined_from(
                    b.aabb,
                    g.aabb
                )
                c.aabb= geometry.AABB3.create_combined_from(
                    a.aabb,
                    f.aabb
                )

                a.height= 1 + maths.maxi(b.height, g.height)
                c.height= 1 + maths.maxi(a.height, f.height)
            else:
                c.right= g
                a.right= f
                f.parent= a

                a.aabb= geometry.AABB3.create_combined_from(
                    b.aabb,
                    f.aabb
                )
                c.aabb= geometry.AABB3.create_combined_from(
                    a.aabb,
                    g.aabb
                )
                
                a.height= 1 + maths.maxi(b.height, f.height)
                c.height= 1 + maths.maxi(a.height, g.height)

            return c
        
        if balance < -1:
            d: Node0= b.left
            e: Node0= b.right

            b.left= a
            b.parent= a.parent
            a.parent= b

            if b.parent != None:
                if b.parent.left is a:
                    b.parent.left= b
                else:
                    b.parent.right= b
            else:
                self.root= b

            
            if d.height > e.height:
                b.right= d
                a.left= e
                e.parent= a

                a.aabb= geometry.AABB3.create_combined_from(
                    c.aabb, 
                    e.aabb
                )

                b.aabb= geometry.AABB3.create_combined_from(
                    a.aabb,
                    d.aabb
                )

                a.height= 1 + maths.maxi(c.height, e.height)
                b.height= 1 + maths.maxi(a.height, d.height)
            else:
                b.right= e
                a.left= d
                d.parent= a

                a.aabb= geometry.AABB3.create_combined_from(
                    c.aabb,
                    d.aabb
                )

                b.aabb= geometry.AABB3.create_combined_from(
                    a.aabb,
                    e.aabb
                )

                a.height= 1 + maths.maxi(c.height, d.height)
                b.height= 1 + maths.maxi(a.height, e.height)
            
            return b

        return a

    def _insert_leaf(self, leaf: Node0) -> None:
        if self.root == None:
            self.root= leaf
            self.root.parent= None
            return

        leaf_aabb= leaf.aabb
        current: Node0= self.root

        while not leaf.is_leaf():

            left= current.left
            right= current.right

            area= current.aabb.perimeter()

            union= geometry.AABB3.create_combined_from(
                current.aabb,
                leaf_aabb
            )
            union_perimeter= union.perimeter()

            cost: float= 2.0 * union_perimeter
            dcost: float= 2.0 * (union_perimeter - area)

            cost_left: float= 0.0
            if left.is_leaf():
                aabb= geometry.AABB3.create_combined_from(
                    leaf.aabb,
                    left.aabb
                )
                cost_left= aabb.perimeter() + dcost
            else:
                aabb= geometry.AABB3.create_combined_from(
                    leaf_aabb,
                    left.aabb
                )
                oldperimiter= left.aabb.perimeter()
                newperimiter= aabb.perimeter()
                cost_left= (newperimiter - oldperimiter) + dcost
            
            cost_right: float= 0.0
            if right.is_leaf():
                aabb= geometry.AABB3.create_combined_from(
                    leaf.aabb,
                    right.aabb
                )
                cost_right= aabb.perimeter() + dcost
            else:
                aabb= geometry.AABB3.create_combined_from(
                    right.aabb,
                    leaf_aabb
                )
                oldperimiter= right.aabb.perimeter()
                newperimiter= aabb.perimeter()
                cost_right= (newperimiter - oldperimiter) + dcost

            if cost < cost_left and cost < cost_right:
                break

            current= left if (cost_left < cost_right) else right

        sibling= current
        old_parent= sibling.parent
        new_parent= Node0()
        new_parent.parent= old_parent
        new_parent.aabb= geometry.AABB3.create_combined_from(
            leaf_aabb,
            sibling.aabb
        )
        new_parent.height= sibling.height + 1

        if old_parent != None:
            if old_parent.left is sibling:
                old_parent.left= new_parent
            else:
                old_parent.right= new_parent
            
            new_parent.left= sibling
            new_parent.right= leaf
            sibling.parent= new_parent
            leaf.parent= new_parent
        else:
            new_parent.left= sibling
            new_parent.right= leaf
            sibling.parent= new_parent
            leaf.parent= new_parent
            self.root= new_parent

        current= leaf.parent
        while current != None:
            current= self._balance_leaf(current)

            current.height= 1 + maths.maxi(
                current.left.height,
                current.right.height
            )

            current.aabb= geometry.AABB3.create_combined_from(
                current.left.aabb,
                current.right.aabb
            )

            current= current.parent

    def _remove_leaf(self, leaf: Node0) -> None:
        if self.root == None:
            return

        if leaf is self.root:
            self.root= None
            return

        parent= leaf.parent
        gparent= parent.parent
        sibling= parent.right if (parent.left is leaf) else parent.left

        if gparent != None:
            if gparent.left is parent:
                gparent.left= sibling
            else:
                gparent.right= sibling
            
            sibling.parent= gparent

            current: Node0= gparent
            while current != None:
                current= self._balance_leaf(current)

                left= current.left
                right= current.right
                current.height= 1 + maths.maxi(left.height, right.height)
                current.aabb= geometry.AABB3.create_combined_from(
                    left.aabb,
                    right.aabb
                )
                current= current.parent
        else:
            self.root= sibling
            sibling.parent= None

    def _add_node(self, obj: T) -> None:
        node= Node0()
        node.obj= obj
        node.aabb= obj.compute_aabb()
        node.aabb.expanded(0.1)
        self.leaves[obj]= node
        self._insert_leaf(node)

    def _update_node(self, obj: T, node: Node0) -> None:
        obj_bounds= obj.compute_aabb()
        obj_bounds.expanded(0.1)

        check= node.aabb.contains_aabb(obj_bounds)

        if check:
            p0= node.aabb.perimeter()
            p1= obj_bounds.perimeter()
            r= p0 / p1
            if r <= 2.0:
                return

        self._remove_leaf(node)
        node.aabb= obj_bounds
        self._insert_leaf(node)

    def _is_valid(self, node: Node0) -> bool:
        if node == None:
            return True

        if node is self.root:
            if node.parent != None:
                return False


        l= node.left
        r= node.right

        if node.is_leaf():
            if l != None:
                return False
            if r != None:
                return False
            if node.height != 0:
                return False
            return True

        h0= l.height
        h1= r.height
        h= 1 + maths.maxi(h0, h1)
        if node.height != h:
            return False
        
        aabb= geometry.AABB3.create_combined_from(
            l.aabb,
            r.aabb
        )
        if not aabb.get_min().is_equil(node.aabb.get_min()):
            return False

        if not aabb.get_max().is_equil(node.aabb.get_max()):
            return False
        
        if not (l.parent is node):
            return False
        if not (r.parent is node):
            return False

        cl= self._is_valid(l)
        cr= self._is_valid(r)

        return cl and cr

    def debug(self, s: shader.Shader, c: camera.Camera):
        for leaf in self.leaves.values():
            m= model.CubeModelAABB(leaf.aabb)
            m.draw(s, c, True)
            m.delete()

    def add(self, obj: T) -> None:
        node= self.leaves.get(obj, None)
        if node != None:
            self._update_node(obj, node)
        else:
            self._add_node(obj)

    def remove(self, obj: T) -> None:
        node= self.leaves.get(obj, None)
        if node != None:
            self._remove_leaf(node)
    
    def update(self, obj: T) -> None:
        node= self.leaves.get(obj, None)
        if node != None:
            self._update_node(obj, node)
        else:
            self._add_node(obj)

    def is_valid(self):
        return self._is_valid(self.root)

    def ray_cast(self, ray: geometry.Ray3) -> list[Node0|None]:
        direction= ray.direction
        if direction.is_zero():
            return False

        que= [self.root]
        while que:
            current= que.pop(0)

            if current == None:
                continue

            if not current.is_leaf():
                que.append(current.left)
                que.append(current.right)
            else:
                if ray.cast_aabb(current.aabb) > 0:
                    return True
        return False
 

# --- qtree


class Node1:

    def __init__(self) -> None:
        self.aabb: geometry.AABB3= geometry.AABB3()
        self.children: list['Node1'|None]= []
        self.objs: list[T|None]= []
        self.max_objs: int= 3
        self.margin: float= 0.0
        self.min_size: float= 0.0
        self.max_size: float= 0.0
        self.extent: float= 0.0
        self.center: maths.Vec3= maths.Vec3()

    @staticmethod
    def init(base_len: float, min_size: float, margin: float, center: maths.Vec3) -> 'Node1':
        node= Node1()
        node.set_up(base_len, min_size, margin, center)
        return node

    def set_up(self, base_len: float, min_size: float, margin: float, center: maths.Vec3):
        self.max_size= base_len
        self.min_size = min_size
        self.margin= margin 
        self.center= center
        self.aabb= geometry.AABB3(
            self.center,
            maths.Vec3.create_from_value(self.margin * self.max_size)
        )


    def _split(self):
        q: float= self.max_size * 0.25
        l: float= self.max_size * 0.5
        m: float= self.margin
        ms: float= self.min_size
        cen: maths.Vec3= self.center

        self.children= [None]*8

        self.children[0]= Node1.init(l, ms, m, cen + maths.Vec3(-q,  q, -q))
        self.children[1]= Node1.init(l, ms, m, cen + maths.Vec3( q,  q, -q))
        self.children[2]= Node1.init(l, ms, m, cen + maths.Vec3(-q,  q,  q))
        self.children[3]= Node1.init(l, ms, m, cen + maths.Vec3( q,  q,  q))
        self.children[4]= Node1.init(l, ms, m, cen + maths.Vec3(-q, -q, -q))
        self.children[5]= Node1.init(l, ms, m, cen + maths.Vec3( q, -q, -q))
        self.children[6]= Node1.init(l, ms, m, cen + maths.Vec3(-q, -q,  q))
        self.children[7]= Node1.init(l, ms, m, cen + maths.Vec3( q, -q,  q))

    def _best_fit(self, v3: maths.Vec3) -> int:
        result: int= 0
        if v3.x <= self.center.x:
            result += 1
        if v3.y >= self.center.y:
            result += 4
        if v3.z >= self.center.z:
            result += 2
        return result
    
    def _has_childern(self) -> bool:
        return len(self.children) > 0

    def _add(self, obj: T, obj_aabb: geometry.AABB3):
        if not self._has_childern():
            if len(self.objs) < self.max_objs or (self.max_size * 0.5) < self.min_size:
                self.objs.append(obj)
                return

            if len(self.children) == 0:
                self._split()

                n= len(self.objs)-1
                while n >= 0:
                    n -= 1
                    current= self.objs[n]
                    bounds= current.compute_aabb()
                    bf: int= self._best_fit(bounds.center)

                    if self.children[bf].aabb.contains_aabb(bounds):
                        self.children[bf]._add(current, bounds)
                        self.objs.remove(current)
                
        bf: int= self._best_fit(obj_aabb.center)
        if self.children[bf].aabb.contains_aabb(obj_aabb):
            self.children[bf]._add(obj, obj_aabb)
        else:
            self.objs.append(obj)

    def add(self, obj: T) -> bool:
        obj_aabb= obj.compute_aabb()
        if not self.aabb.contains_aabb(obj_aabb):
            return False
        self._add(obj, obj_aabb)
        return True

    def _should_merge(self) -> bool:
        count= len(self.objs)
        if self.children:
            for child in self.children:
                if child.children:
                    return False
                count += len(child.objs)
        return count <= self.max_objs

    def _merge(self) -> None:
        for child in self.children:
            for obj in child.objs[::-1]:
                self.objs.append(obj)
        self.children.clear()

    def _remove(self, obj: T, obj_aabb: geometry.AABB3) -> None:
        check: bool= False
        for o in self.objs:
            if o is obj:
                self.objs.remove(obj)
                check= True
                break

        if not check and self.children:
            bf= self._best_fit(obj_aabb.center)
            child= self.children[bf]
            child._remove(obj, obj_aabb)

        if check and self.children:
            if self._should_merge():
                self._merge()

    def remove(self, obj: T) -> bool:
        aabb= obj.compute_aabb()
        if not self.aabb.contains_aabb(aabb):
            return False
        self._remove(obj, aabb)
        return True

    def debug(self, s: shader.Shader, c: camera.Camera):
        que: list[Node1|None]= [self]
        while que:
            current=que.pop(0)

            if current.children:
                for child in current.children:
                    que.append(child)

            for obj in current.objs:
                obj_aabb= obj.compute_aabb()
                obj_aabb.expanded(0.1)
                m= model.CubeModelAABB(obj_aabb)
                m.draw(s, c, True)
                m.delete()
            m= model.CubeModelAABB(current.aabb)
            m.draw(s, c, True)
            m.delete()

    def raycast_check(self, ray: geometry.Ray3) -> bool:
        if ray.cast_aabb(self.aabb) < 0:
            return False

        que: list[Node1|None]= [self]
        while que:
            current= que.pop(0)
            if current.children:
                for child in current.children:
                    que.append(child)
            for obj in current.objs:
                if ray.cast_aabb(obj.compute_aabb()) >= 0:
                    return True
        return False

class Tree1:
    def __init__(self, max_size: float, min_size: float) -> None:
        self.root: Node1|None = Node1()
        self.root.set_up(max_size, min_size, 1.0, maths.Vec3())

    def add(self, obj: T) -> bool:
        return self.root.add(obj)

    def remove(self, obj: T) -> bool:
        return self.root.remove(obj)

    def raycast_check(self, ray: geometry.Ray3) -> bool:
        return self.root.raycast_check(ray)

    def debug(self, s: shader.Shader, c: camera.Camera):
        self.root.debug(s, c)

    
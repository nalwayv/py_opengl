"""
"""
from typing import TypeVar

from py_opengl import geometry
from py_opengl import maths
from py_opengl import model
from py_opengl import shader


# ---


T= TypeVar('T', bound= model.Model)


# ---


class Node:
    
    __slots__= (
        'left',
        'right',
        'parent',
        'height',
        'aabb',
        'obj'
    )

    def __init__(self) -> None:
        self.left: 'Node'|None= None
        self.right: 'Node'|None= None
        self.parent: 'Node'|None= None
        self.height: int= 0
        self.aabb: geometry.AABB3= geometry.AABB3()
        self.obj: T|None= None

    def is_leaf(self) -> bool:
        return self.left == None

    def has_obj(self) -> bool:
        return self.obj != None


# ---


class ABTree:
    
    __slots__= (
        'root',
        'leaves'
    )

    def __init__(self) -> None:
        self.root: Node= Node()
        self.leaves: dict[T, Node] = {}

    def _balance_leaf(self, leaf: Node) -> Node:
        a: Node= leaf
        if a.is_leaf() or a.height < 2:
            return a

        b: Node= a.left
        c: Node= a.right

        balance: int= c.height - b.height

        if balance > 1:
            f: Node= c.left
            g: Node= c.right

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
            d: Node= b.left
            e: Node= b.right

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

    def _insert_leaf(self, leaf: Node) -> None:
        if self.root == None:
            self.root= leaf
            self.root.parent= None
            return

        leaf_aabb= leaf.aabb
        current: Node= self.root

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
        new_parent= Node()
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

    def _remove_leaf(self, leaf: Node) -> None:
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

            current: Node= gparent
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
        node= Node()
        node.obj= obj
        node.aabb= obj.compute_aabb()
        node.aabb.expanded(0.1)
        self.leaves[obj]= node
        self._insert_leaf(node)

    def _update_node(self, obj: T, node: Node) -> None:
        obj_bounds= obj.compute_aabb()
        check= node.aabb.contains_aabb(obj_bounds)

        if check:
            p0= node.aabb.perimeter()
            p1= obj_bounds.perimeter()
            r= p0 / p1
            if r <= 2.0:
                return

        self._remove_leaf(node)
        node.aabb= obj_bounds.expand(0.1)
        self._insert_leaf(node)

    def _is_valid(self, node: Node) -> bool:
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

    def shift(self, shift_by: maths.Vec3) -> None:
        que: list[Node|None]= [self.root]

        while que:
            current: Node|None= que.pop(0)
            
            if current == None:
                continue

            if not current.is_leaf():
                que.append(current.left)
                que.append(current.right)
            
            m4= maths.Mat4.create_translation(shift_by)
            current.aabb.transform(m4)

    def debug(self, s: shader.Shader, view: maths.Mat4, projection: maths.Mat4) -> None:
        que: list[Node|None]= [self.root]

        while que:
            current: Node|None= que.pop(0)

            if current == None:
                return

            if not current.is_leaf():
                que.append(current.left)
                que.append(current.right)

            leaf_m= model.CubeModel(current.aabb.extents)
            leaf_m.translate(current.aabb.center)
            
            leaf_m.draw(s, view, projection, True)
            leaf_m.delete()

    def raycast(self, ray: geometry.Ray3) -> T|None:
        """
        """
        que: list[Node|None]= [self.root]

        closest_t: float= -1.0
        closest_o: T|None= None  

        while que:
            current: Node|None= que.pop(0)

            if current == None:
                continue

            if not current.is_leaf():
                que.append(current.left)
                que.append(current.right)

            t: float= ray.cast_aabb(current.aabb)
            if t > 0:
                if current.has_obj():
                    if closest_t < 0 or t < closest_t:
                        closest_t= t
                        closest_o= current.obj
        return closest_o

    def query(self, ab3: geometry.AABB3) -> list[T|None]:
        """
        """
        que: list[Node|None]= [self.root]
        result: list[T|None]= []

        while que:
            current: Node|None= que.pop(0)

            if current == None:
                continue

            if not current.is_leaf():
                que.append(current.left)
                que.append(current.right)

            if ab3.intersect_aabb(current.aabb):
                if current.has_obj():
                    result.append(current.obj)

        return result

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
            self.leaves.pop(node)
    
    def update(self, obj: T) -> None:
        node= self.leaves.get(obj, None)
        if node != None:
            self._update_node(obj, node)
        else:
            self._add_node(obj)

    def is_valid(self) -> bool:
        return self._is_valid(self.root)

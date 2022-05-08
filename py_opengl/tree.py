"""AABB3 Tree
"""
from dataclasses import dataclass, field
from typing import Final, Optional

from py_opengl import geometry
from py_opengl import maths


# ---


MARGIN: Final[float] = 4.0
AABB_REDUCTION_RATIO:  Final[float] = 2.0


# ---


@dataclass(eq= False, repr= False, slots= True)
class AABBNode:
    left: Optional['AABBNode']= None
    right: Optional['AABBNode']= None
    parent: Optional['AABBNode']= None
    height: int= 0
    aabb: geometry.AABB3= geometry.AABB3()

    item: Optional[geometry.Sphere3]= None

    def is_leaf(self) -> bool:
        return self.left is None


# TODO allow more shapes then just sphere3


@dataclass(eq= False, repr= False, slots= True)
class AABBTree:
    root: Optional[AABBNode]= None
    leaves: dict[geometry.Sphere3, AABBNode]= field(default_factory=dict)
    update_aabb: geometry.AABB3= geometry.AABB3()

    def add(self, obj: geometry.Sphere3):
        if node := self.leaves.get(obj):
            self._update_node(obj, node)
        else:
            self._add_node(obj)

    def remove(self, obj: geometry.Sphere3):
        if node := self.leaves.pop(obj, None):
            self._remove(node)

    def get_aabb(self, obj: geometry.Sphere3) -> geometry.AABB3:
        if node := self.leaves.get(node):
            return node.aabb
        
        obj_aabb: geometry.AABB3= obj.compute_aabb()
        if obj_aabb.is_degenerate():
            return obj_aabb

        obj_aabb.expanded(MARGIN)
        return obj_aabb

    def length(self) -> int:
        return len(self.leaves)

    def contains_obj(self, obj: geometry.Sphere3) -> bool:
        return obj in self.leaves

    def is_valid(self) -> bool:
        return self._is_valid(self.root)

    def _add_node(self, obj: geometry.Sphere3):
        self.update_aabb.set_from(obj.compute_aabb())
        self.update_aabb.expanded(MARGIN)

        node= AABBNode(
            aabb=self.update_aabb,
            item= obj
        )

        #insert
        self.leaves[obj] = node

        self._insert(node)
        # TODO update

    def _update_node(self, obj: geometry.Sphere3, node: AABBNode) -> None:
        self.update_aabb.set_from(obj.compute_aabb())
        check: bool= node.aabb.intersect_aabb(self.update_aabb)

        # TODO
        self.update_aabb.expanded(MARGIN)
        
        if check:
            p0= node.aabb.perimeter()
            p1= self.update_aabb.perimeter()
            r= p0/p1
            if r <= AABB_REDUCTION_RATIO:
                return

        self._remove(node)

        node.aabb.set_from(self.update_aabb)

        self._insert(node)
        #TODO update

    def _remove(self, node: AABBNode) -> None:
        if not self.root:
            return

        if node is self.root:
            self.root = None
            return

        parent: Optional[AABBNode]=  node.parent
        gparent: Optional[AABBNode]= parent.parent
        sibling: Optional[AABBNode]= None

        if parent.left is node:
            sibling= parent.right
        else:
            sibling= parent.left

        if gparent:
            if gparent.left is parent:
                gparent.left= sibling
            else:
                gparent.right= sibling

            sibling.parent= gparent

            next_node: Optional[AABBNode]= gparent
            while next_node is not None:
                next_node= self._balance(next_node)

                left: Optional[AABBNode]= next_node.left
                right: Optional[AABBNode]= next_node.right

                next_node.height= 1 + maths.maxi(left.height, right.height)
                next_node.aabb.combined_from(left.aabb, right.aabb)
                next_node= next_node.parent
        else:
            self.root= sibling
            sibling.parent= None

    def _balance(self, item: AABBNode) -> AABBNode:
        a: Optional[AABBNode]= item

        if a.is_leaf() or a.height < 2:
            return a

        b: Optional[AABBNode]= a.left
        c: Optional[AABBNode]= a.right

        balance: int= c.height - b.height

        # rotate c up
        if balance > 1:
            f: Optional[AABBNode]= c.left
            g: Optional[AABBNode]= c.right

            # swap
            c.left= a
            c.parent= a.parent
            a.parent= c

            if c.parent is not None:
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

                a.aabb.combined_from(b.aabb, g.aabb)
                c.aabb.combined_from(a.aabb, f.aabb)

                a.height= 1 + maths.maxi(b.height, g.height)
                c.height= 1 + maths.maxi(a.height, f.height)
            else:
                c.right= g
                a.right= f
                f.parent= a

                a.aabb.combined_from(b.aabb, f.aabb)
                c.aabb.combined_from(a.aabb, g.aabb)

                a.height= 1 + maths.maxi(b.height, f.height)
                c.height= 1 + maths.maxi(a.height, g.height)

            return c

        # rotate b up
        if balance < -1:
            d: Optional[AABBNode]= b.left
            e: Optional[AABBNode]= b.right

            # swap
            b.left= a
            b.parent= a.parent
            a.parent= b

            if b.parent is not None:
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

                a.aabb.combined_from(c.aabb, e.aabb)
                b.aabb.combined_from(a.aabb, d.aabb)

                a.height= 1 + maths.maxi(c.height, e.height)
                b.height= 1 + maths.maxi(a.height, d.height)
            else:
                b.right= e
                a.left= d
                d.parent= a

                a.aabb.combined_from(c.aabb, d.aabb)
                b.aabb.combined_from(a.aabb, e.aabb)

                a.height= 1 + maths.maxi(c.height, d.height)
                b.height= 1 + maths.maxi(a.height, e.height)

            return b

        return a

    def _insert(self, item: AABBNode) -> None:
        if self.root is None:
            self.root= item
            return

        tmp: geometry.AABB3= geometry.AABB3()
        item_aabb= item.aabb

        current= self.root
        while not current.is_leaf():
            left: Optional[AABBNode]= current.left
            right: Optional[AABBNode]= current.right

            tmp= current.aabb

            area: float= tmp.perimeter()
            combined_area= tmp.create_combined_with(item_aabb).perimeter()

            cost: float= 2.0 * combined_area
            d_cost: float= 2.0 * (combined_area - area)

            cost_left: float= 0.0
            if left.is_leaf():
                tmp.combined_from(left.aabb, item_aabb)
                cost_left= tmp.perimeter() + d_cost
            else:
                old_cost= left.aabb.perimeter()
                tmp.combined_from(left.aabb, item_aabb)
                new_cost= tmp.perimeter()
                cost_left = (new_cost - old_cost) + d_cost

            cost_right: float= 0.0
            if right.is_leaf():
                tmp.combined_from(right.aabb, item_aabb)
                cost_right= tmp.perimeter() + d_cost
            else:
                old_cost= right.aabb.perimeter()
                tmp.combined_from(right.aabb, item_aabb)
                new_cost= tmp.perimeter()
                cost_right = (new_cost - old_cost) + d_cost

            if cost < cost_left and cost < cost_right:
                break

            if cost_left < cost_right:
                current= left
            else:
                current= right

        parent= current.parent

        new_parent= AABBNode(
            parent= current.parent,
            aabb= current.aabb.create_combined_with(item_aabb),
            height= current.height + 1
        )

        if parent is not None:
            if parent.left is current:
                parent.left= new_parent
            else:
                parent.right= new_parent

            new_parent.left= current
            new_parent.right= item
            current.parent= new_parent
            item.parent= new_parent
        else:
            new_parent.left= current
            new_parent.right= item
            current.parent= new_parent
            item.parent= new_parent
            self.root= new_parent
        

        current= item.parent
        while current is not None:
            current= self._balance(current)

            left: Optional[AABBNode]= current.left
            right: Optional[AABBNode]= current.right

            current.height = 1 + maths.maxi(left.height, right.height)
            current.aabb.combined_from(left.aabb, right.aabb)
            current= current.parent

    def _is_valid(self, node: AABBNode) -> bool:
        if node is None:
            return True

        if node is self.root:
            if node.parent is not None:
                return False
        
        left: Optional[AABBNode]= node.left
        right: Optional[AABBNode]= node.right

        if node.is_leaf():
            if (
                node.left is not None or
                node.right is not None or
                node.height != 0 or
                node.item is None
            ):
                return False
            return True

        if not node.aabb.contains_aabb(left.aabb):
            return False

        if right is not None and not node.aabb.contains_aabb(right.aabb):
            return False
        
        if left.parent is not node:
            return False
        if right.parent is not node:
            return False

        check_left= self._is_valid(left)
        check_right= self._is_valid(right)

        return check_left and check_right
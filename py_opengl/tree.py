"""AABB3 Tree
"""
from dataclasses import dataclass, field
from typing import Optional, Any

from py_opengl import geometry
from py_opengl import maths


# TODO
# ---

# NULL_NODE: Final[int]= -1
# MARGIN: Final[float]= 4.0
# AABB_REDUCTION_RATIO: Final[float]= 2.0


# ---


# COMPUTE_FN: Callable[[Any, geometry.AABB3], None]
# EXPAND_FN: Callable[[Any, geometry.AABB3], None]


# ---


@dataclass(eq= False, repr= False, slots= True)
class Node:
    left: Optional['Node']= None
    right: Optional['Node']= None
    parent: Optional['Node']= None

    height: int= 0

    aabb: geometry.AABB3= geometry.AABB3()

    item: Optional[Any]= None

    def is_leaf(self) -> bool:
        return self.left is None


@dataclass(eq= False, repr= False, slots= True)
class AABBTree:
    root: Optional[Node]= None
    leaves: dict[Any, Node]= field(default_factory=dict)
    updateAABB: geometry.AABB3= geometry.AABB3()

    def _height(self, node: Node) -> int:
        if node is None:
            return 0
        
        a= self._height(node.left)
        b= self._height(node.right)
        return 1 + maths.maxi(a, b)

    def _insert(self, node: Node) -> None:
        if self.root is None:
            self.root= node
            return

        leaf_aabb: geometry.AABB3= node.aabb

        current: Optional[Node]= self.root
        while not current.is_leaf():
            area: float= current.aabb.perimeter()

            combined_AABB: geometry.AABB3= geometry.AABB3.create_combined_from(
                leaf_aabb,
                current.aabb
            )
            combined_area: float= combined_AABB.perimeter()

            cost: float= 2.0 * combined_area
            d_cost: float= 2.0 * (combined_area - area)

            left: Optional[Node]= current.left
            right: Optional[Node]= current.right

            # LEFT
            cost_left: float= 0.0
            if left.is_leaf():
                aabb_0= geometry.AABB3.create_combined_from(
                    leaf_aabb,
                    left.aabb
                )
                cost_left= aabb_0.perimeter() + d_cost
            else:
                aabb_0= geometry.AABB3().create_combined_from(
                    leaf_aabb,
                    left.aabb
                )
                old_area: float= left.aabb.perimeter()
                new_area: float= aabb_0.perimeter()
                cost_left= (new_area - old_area) + d_cost

            # RIGHT
            cost_right: float= 0.0
            if right.is_leaf():
                aabb_1= geometry.AABB3.create_combined_from(
                    leaf_aabb,
                    right.aabb
                )
                cost_right= aabb_1.perimeter() + d_cost
            else:
                aabb_1= geometry.AABB3.create_combined_from(
                    leaf_aabb,
                    right.aabb
                )
                old_area: float= right.aabb.perimeter()
                new_area: float= aabb_1.perimeter()
                cost_right= (new_area - old_area) + d_cost

            if cost < cost_left and cost < cost_right:
                break

            if cost_left < cost_right:
                current= left
            else:
                current= right

        sibling= current
        old_parent= sibling.parent
        new_parent= Node(
            parent= old_parent,
            aabb= geometry.AABB3.create_combined_from(
                leaf_aabb,
                sibling.aabb
            ),
            height= current.height + 1
        )

        if old_parent is not None:
            if old_parent.left is sibling:
                old_parent.left= new_parent
            else:
                old_parent.right= new_parent

            new_parent.left= sibling
            new_parent.right= node
            current.parent= new_parent
            node.parent= new_parent
        else:
            new_parent.left= sibling
            new_parent.right= node
            current.parent= new_parent
            node.parent= new_parent
            self.root= new_parent

        current: Optional[Node]= node.parent
        while current is not None:
            current= self._balance(current)

            left: Optional[Node]= current.left
            right: Optional[Node]= current.right

            current.height = 1 + maths.maxi(left.height, right.height)
            current.aabb.combined_from(
                left.aabb,
                right.aabb
            )
            current= current.parent

    def _remove(self, node: Node) -> None:
        if not self.root:
            return

        if node is self.root:
            self.root = None
            return

        parent: Optional[Node]=  node.parent
        gparent: Optional[Node]= parent.parent
        sibling: Optional[Node]= None

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

            current: Optional[Node]= gparent
            while current is not None:
                current= self._balance(current)

                left: Optional[Node]= current.left
                right: Optional[Node]= current.right

                current.height = 1 + maths.maxi(left.height, right.height)
                current.aabb.combined_from(left.aabb, right.aabb)
                current= current.parent

        else:
            self.root= sibling
            sibling.parent= None

    def _balance(self, node: Node) -> Optional[Node]:
        a: Optional[Node]= node

        if a.is_leaf() or a.height < 2:
            return a

        b: Optional[Node]= a.left
        c: Optional[Node]= a.right

        balance: int= c.height - b.height

        # rotate c up
        if balance > 1:
            f: Optional[Node]= c.left
            g: Optional[Node]= c.right

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
            d: Optional[Node]= b.left
            e: Optional[Node]= b.right

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

    def _is_valid(self, node: Node) -> bool:
        if node is None:
            return True

        if node is self.root:
            if node.parent is not None:
                return False
        
        left: Optional[Node]= node.left
        right: Optional[Node]= node.right

        if node.is_leaf():
            if node.left is not None:
                return False

            if node.right is not None:
                return False

            if node.height != 0:
                return False

            return True

        
        aabb= geometry.AABB3.create_combined_from(left.aabb, right.aabb)

        min_0= aabb.get_min()
        min_1= node.aabb.get_min()
        if not min_0.is_equil(min_1):
            return False

        max_0= aabb.get_max()
        max_1= node.aabb.get_max()
        if not max_0.is_equil(max_1):
            return False



        if left.parent is not node:
            return False

        if right.parent is not node:
            return False

        check_left= self._is_valid(left)
        check_right= self._is_valid(right)
        return check_left and check_right

    def _insert_node(self, t: Any):
        # TODO updateAABB
        # TODO t need to have a hash

        node= Node(item=t)
        node.aabb.set_from(self.updateAABB)

        self.leaves[t]= node
        self._insert(node)

    def clear(self):
        self.leaves.clear()
        self.root= None

    def insert(self, t: Any):
        node: Optional[Node]= self.leaves.get(t, None)
        if node is None:
            self._insert_node(t)

    def remove(self, t: Any):
        node: Optional[Node]= self.leaves.get(t, None)
        if node is not None:
            self._remove(node)

    def is_valid(self):
        return self._is_valid(self.root)

    def height(self):
        return self._height(self.root)

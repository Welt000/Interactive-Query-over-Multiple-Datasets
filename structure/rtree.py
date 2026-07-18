from functools import reduce
from structure.point import Point
class RTreeNode:
    def __init__(self, dim, M=36, parent=None):
        self.dim = dim  # dimension
        self.M = M  # max children
        self.entries = []  # used for storing data or the children
        self.parent = parent
        self.is_leaf = True
        self.min_point = [float('inf')] * dim  # min bounding point
        self.max_point = [float('-inf')] * dim  # max bounding point

    def insert_entry(self, pp):
        self.entries.append(pp)
        self.min_point = [min(self.min_point[d], pp.coord[d]) for d in range(self.dim)]
        self.max_point = [max(self.max_point[d], pp.coord[d]) for d in range(self.dim)]

    def __str__(self):
        return f"RTreeNode(Min: {self.min_point}, Max: {self.max_point}, Entries: {len(self.entries)})"

    def update_mbr(self):
        self.min_point = [float('inf')] * self.dim  # min bounding point
        self.max_point = [float('-inf')] * self.dim  # max bounding point
        if self.is_leaf:
            for i in range(self.dim):
                self.min_point[i] = min(self.min_point[i], min(child.coord[i] for child in self.entries))
                self.max_point[i] = max(self.max_point[i], max(child.coord[i] for child in self.entries))
        else:
            for i in range(self.dim):
                self.min_point[i] = min(self.min_point[i], min(child.min_point[i] for child in self.entries))
                self.max_point[i] = max(self.max_point[i], max(child.max_point[i] for child in self.entries))
        # update upward
        if self.parent:
            self.parent.update_mbr()


class RTree:
    def __init__(self, dim, M=36):
        self.M = M
        self.dim = dim
        self.root = RTreeNode(dim, M)

    def insert(self, pp=None):
        node = self._choose_leaf(self.root, pp)
        node.insert_entry(pp)
        node.update_mbr()
        if len(node.entries) > self.M:
            self._split_node(node)

    def _choose_leaf(self, node, pp):
        if node.is_leaf:
            return node
        # the node with the smallest area addition
        best_child = None
        min_increase = float('inf')
        for child in node.entries:
            area_before = self._calculate_area(child.min_point, child.max_point)
            area_after = self._calculate_area(
                [min(pp.coord[d], child.min_point[d]) for d in range(self.dim)],
                [max(pp.coord[d], child.max_point[d]) for d in range(self.dim)]
            )
            increase = area_after - area_before
            if increase < min_increase:
                min_increase = increase
                best_child = child
        return self._choose_leaf(best_child, pp)

    def _split_node(self, node):
        # split into two nodes
        new_sibling = RTreeNode(self.dim, self.M)
        half = len(node.entries) // 2
        if node.is_leaf:
            node.entries.sort(key=lambda x: x.coord[0])
        else:
            node.entries.sort(key=lambda x: x.min_point[0] + x.max_point[0])
            new_sibling.is_leaf = False
        new_sibling.entries = node.entries[half:]
        node.entries = node.entries[:half]

        # update parent
        if node.parent:
            node.parent.entries.append(new_sibling)
            new_sibling.parent = node.parent
            # after append, it may overflow
            if node.parent.M - len(node.parent.entries) < 0:
                self._split_node(node.parent)
        else:
            new_node = RTreeNode(self.dim, self.M)
            new_node.is_leaf = False
            new_node.entries.append(node)
            new_node.entries.append(new_sibling)
            node.parent = new_node
            new_sibling.parent = new_node
            self.root = new_node

        node.update_mbr()
        new_sibling.update_mbr()

    def _calculate_area(self, min_point, max_point):
        # 计算超矩形的“体积”
        return reduce(lambda x, y: x * y, [max_point[d] - min_point[d] for d in range(self.dim)])

'''
rt = RTree(2, 3)
for i in range(5):
    for j in range(5):
        pp = Point(2)
        pp.coord[0] = i
        pp.coord[1] = j
        rt.insert(pp)
print(rt)
'''

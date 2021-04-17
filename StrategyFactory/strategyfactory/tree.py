from typing import *
from anytree import NodeMixin, RenderTree


class Node(NodeMixin):
    def __init__(self, name, function_or_data, max_childs=2, parent=None):
        self._is_operation = True if isinstance(function_or_data, Callable) else False
        self._function_or_data = function_or_data
        self.name = name
        self.parent = parent
        self.max_childs = max_childs

    @property
    def is_operation(self):
        return self._is_operation

    @property
    def function_or_data(self):
        return self._function_or_data

    def evaluate(self):
        if self.is_operation:
            return self.function_or_data(*[c.evaluate() for c in self.children])
        return self.function_or_data

    @property
    def full(self):
        return len(self.children) >= self.max_childs

    def __repr__(self):
        return self.name


def generate_node_fn(name, function_or_data, max_childs=2):
    def wrapped(parent=None):
        return Node(name, function_or_data, max_childs, parent=parent)

    return wrapped


class Tree(object):
    def __init__(self):
        self.nodes = []
        self._root = None

    def insert(self, node_fn):
        if not self._root:
            self._root = node_fn(parent=None)
            self.nodes.append(self._root)
            self._currnode = self._root
            return

        if not self._currnode.full:
            node = node_fn(parent=self._currnode)
            self.nodes.append(node)

            if node.is_operation:
                self._currnode = node
        else:
            self._currnode = self._currnode.parent
            self.insert(node_fn)

    @property
    def iscompleted(self):
        if self._currnode.is_operation & (not self._currnode.full):
            return False
        temp = self._currnode
        while True:
            if temp.is_operation & (not temp.full):
                return False
            if temp.parent is None:
                return True
            temp = temp.parent

    def evaluate(self):
        return self.root.evaluate()

    def render(self, print_result=False):
        string = "\n".join([f"{pre}{node.name}" for pre, _, node in RenderTree(self.root)])
        if print_result:
            print(string)
        return string

    @property
    def root(self):
        return self._root

    @property
    def current(self):
        return self._currnode


def test_tree():
    import numpy as np

    tree = Tree()

    nodefn_div = generate_node_fn("{/}", lambda x, y: x / y)
    nodefn_mul = generate_node_fn("{*}", lambda x, y: x * y)
    nodefn_exp = generate_node_fn("{exp}", lambda x: np.exp(x), 1)
    nodefn_log = generate_node_fn("{log}", lambda x: np.log(x), 1)

    nodefn_2 = generate_node_fn("2", 2)
    nodefn_3 = generate_node_fn("3", 3)

    tree.insert(nodefn_div)
    tree.render(True)

    tree.insert(nodefn_mul)
    tree.render(True)

    tree.insert(nodefn_mul)
    tree.render(True)

    tree.insert(nodefn_exp)
    tree.render(True)

    tree.insert(nodefn_2)
    tree.render(True)

    tree.insert(nodefn_2)
    tree.render(True)

    tree.insert(nodefn_3)
    tree.render(True)

    tree.insert(nodefn_log)
    tree.render(True)

    tree.insert(nodefn_3)
    tree.render(True)

    return tree

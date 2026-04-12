"""
Binary Search Tree (BST) Implementation

This module provides a clean, well-structured implementation of a Binary Search Tree
with standard operations including insertion, search, and in-order traversal.

Author: AI Assistant
"""

from typing import Optional, List


class Node:
    """
    A class representing a node in the Binary Search Tree.

    Attributes:
        value: The value stored in the node.
        left: Reference to the left child node (values less than current).
        right: Reference to the right child node (values greater than current).
    """

    def __init__(self, value: int) -> None:
        """
        Initialize a new Node with the given value.

        Args:
            value: The integer value to store in this node.
        """
        self.value: int = value
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None


class BinarySearchTree:
    """
    A class representing a Binary Search Tree (BST).

    The BST maintains the property that for any given node:
    - All values in the left subtree are less than the node's value
    - All values in the right subtree are greater than the node's value

    Attributes:
        root: The root node of the tree. None if the tree is empty.
    """

    def __init__(self) -> None:
        """
        Initialize an empty Binary Search Tree.
        """
        self.root: Optional[Node] = None

    def insert(self, value: int) -> None:
        """
        Insert a new value into the Binary Search Tree.

        If the value already exists in the tree, it will not be inserted
        (no duplicates allowed).

        Args:
            value: The integer value to insert into the tree.

        Time Complexity: O(h) where h is the height of the tree
        Space Complexity: O(h) due to recursion stack
        """
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, current: Node, value: int) -> None:
        """
        Helper method to recursively insert a value into the tree.

        Args:
            current: The current node being examined.
            value: The value to insert.
        """
        if value < current.value:
            if current.left is None:
                current.left = Node(value)
            else:
                self._insert_recursive(current.left, value)
        elif value > current.value:
            if current.right is None:
                current.right = Node(value)
            else:
                self._insert_recursive(current.right, value)
        # If value == current.value, do nothing (no duplicates)

    def search(self, value: int) -> bool:
        """
        Search for a value in the Binary Search Tree.

        Args:
            value: The integer value to search for.

        Returns:
            True if the value exists in the tree, False otherwise.

        Time Complexity: O(h) where h is the height of the tree
        Space Complexity: O(1) for iterative approach
        """
        return self._search_recursive(self.root, value)

    def _search_recursive(self, current: Optional[Node], value: int) -> bool:
        """
        Helper method to recursively search for a value.

        Args:
            current: The current node being examined.
            value: The value to search for.

        Returns:
            True if found, False otherwise.
        """
        if current is None:
            return False

        if value == current.value:
            return True
        elif value < current.value:
            return self._search_recursive(current.left, value)
        else:
            return self._search_recursive(current.right, value)

    def in_order_traversal(self) -> List[int]:
        """
        Perform an in-order traversal of the Binary Search Tree.

        In-order traversal visits nodes in ascending order:
        left subtree -> current node -> right subtree

        Returns:
            A list of values in sorted (ascending) order.

        Time Complexity: O(n) where n is the number of nodes
        Space Complexity: O(n) for the result list and O(h) for recursion stack
        """
        result: List[int] = []
        self._in_order_recursive(self.root, result)
        return result

    def _in_order_recursive(self, current: Optional[Node], result: List[int]) -> None:
        """
        Helper method to perform recursive in-order traversal.

        Args:
            current: The current node being visited.
            result: The list to accumulate values in.
        """
        if current is not None:
            self._in_order_recursive(current.left, result)
            result.append(current.value)
            self._in_order_recursive(current.right, result)


def main() -> None:
    """
    Demonstration of Binary Search Tree usage.

    This function creates a BST, inserts values, performs searches,
    and demonstrates the in-order traversal.
    """
    # Create a new Binary Search Tree
    bst = BinarySearchTree()

    # Insert values into the tree
    values_to_insert = [50, 30, 70, 20, 40, 60, 80, 25, 35]
    print("Inserting values:", values_to_insert)

    for value in values_to_insert:
        bst.insert(value)

    # Perform searches
    print("\n--- Search Operations ---")
    search_values = [25, 100, 50, 90]
    for value in search_values:
        result = bst.search(value)
        print(f"Search for {value}: {'Found' if result else 'Not Found'}")

    # In-order traversal (should return sorted values)
    print("\n--- In-Order Traversal ---")
    sorted_values = bst.in_order_traversal()
    print(f"Sorted values: {sorted_values}")

    # Demonstrate that duplicates are not inserted
    print("\n--- Duplicate Insertion Test ---")
    print("Inserting 50 again (duplicate)...")
    bst.insert(50)
    print(f"Tree after duplicate insertion: {bst.in_order_traversal()}")

    # Edge case: empty tree
    print("\n--- Empty Tree Test ---")
    empty_bst = BinarySearchTree()
    print(f"Empty tree in-order: {empty_bst.in_order_traversal()}")
    print(f"Search in empty tree: {empty_bst.search(10)}")


if __name__ == "__main__":
    main()

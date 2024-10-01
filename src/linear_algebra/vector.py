"""Vector module."""

from dataclasses import dataclass
from functools import cached_property
from math import sqrt
from typing import Callable

MISMATCHED_SIZE = "Vectors must be of same size. Self: %s, Vector: %s"


@dataclass(frozen=True)
class Vector:
    """Vector representation.

    Attributes:
        values: Values in the vector
    """

    values: tuple[float, ...]

    @staticmethod
    def uniform(size: int, value: float = 0.0) -> "Vector":
        """Generate a Vector with uniform values.

        Args:
            size: number of values in the vector
            value: value for each dimension in the vector

        Returns:
           Vector containing a (size) number of (value)
        """
        return Vector((value,) * size)

    @staticmethod
    def random(
        size: int,
        random_function: Callable[[], float],
    ) -> "Vector":
        """Generate a vector with random values based on a provided function.

        Args:
            size: number of values to generate
            random_function: random value generator to use

        Returns:
            A Vector of size (size), using values from (random_function)
        """
        return Vector(tuple(random_function() for _ in range(size)))

    @cached_property
    def magnitude(self) -> float:
        """Return the magnitude of the vector.

        Returns:
            The magnitude of the vector
        """
        return sqrt(sum(value**2 for value in self))

    def __len__(self):
        """Return the length of the vector.

        Returns:
            The length of the vector.
        """
        return len(self.values)

    def __getitem__(self, index: int):
        """Return the value at a given index in the vector.

        Args:
            index: the index of the value to get

        Returns:
           The value of the vector at (index)
        """
        return self.values[index]

    def __iter__(self):
        """Return an iterator of the vector's values.

        Returns:
            An iterator of the vector's values
        """
        return iter(self.values)

    def __neg__(self) -> "Vector":
        """Return the negative of the current vector.

        Returns:
            The negative of the current vector.
        """
        return Vector(tuple(-value for value in self))

    def __add__(self, vector: "Vector") -> "Vector":
        """Add two vectors.

        Args:
            vector: The vector to add to the current vector

        Raises:
            ValueError: if the vectors are of two different sizes

        Returns:
           The sum of the provided vector and this vector
        """
        if len(self) != len(vector):
            raise ValueError(MISMATCHED_SIZE % (self, vector))
        return Vector(
            tuple(
                self_value + vector_value
                for self_value, vector_value in zip(self, vector)
            )
        )

    def __sub__(self, vector: "Vector") -> "Vector":
        """Subtract a given vector from the provided vector.

        Args:
            vector: the vector to subtract from the current vector.

        Raises:
            ValueError: if the vectors are of two different sizes

        Returns:
            The difference between the current vector and (vector)
        """
        return self + (-vector)

    def __mul__(self, scalar: int | float) -> "Vector":
        """Multiply the current vector by a scalar value.

        Args:
            scalar: the value to multiply by

        Returns:
           the current vector scaled by (scalar)
        """
        return Vector(tuple(value * scalar for value in self))

    def __rmul__(self, scalar: int | float) -> "Vector":
        """Multiply the current vector by a scalar value.

        Args:
            scalar: the value to multiply by

        Returns:
           the current vector scaled by (scalar)
        """
        return self * scalar

    def __truediv__(self, divisor: int | float) -> "Vector":
        """Divide the current vector by a divisor.

        Args:
            divisor: the value by which to divide the vector

        Raises:
            ZeroDivisionError: if the divisor is 0

        Returns:
           the current vector divided by (divisor)
        """
        if divisor == 0.0:
            raise ZeroDivisionError()
        return self * (1.0 / divisor)

    def __matmul__(self, vector: "Vector") -> float:
        """Return the dot product of two vectors.

        Args:
            vector: the vector to multiply by

        Raises:
            ValueError: if the two vectors are of different size

        Returns:
           the dot product of the current vector and (vector)
        """
        return self.dot(vector)

    def dot(self, vector: "Vector") -> float:
        """Return the dot product of two vectors.

        Args:
            vector: the vector to multiply by

        Raises:
            ValueError: if the two vectors are of different size

        Returns:
           the dot product of the current vector and (vector)
        """
        if len(self) != len(vector):
            raise ValueError(MISMATCHED_SIZE % (self, vector))
        return sum(
            self_value * vector_value
            for self_value, vector_value in zip(self, vector)
        )

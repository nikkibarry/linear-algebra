"""Tests for the `vector` module."""

from random import Random

from linear_algebra.vector import Vector
from tests.unit.assertions import assert_raises


class TestVector:
    def test_uniform_default(self):
        zeroes = Vector.uniform(2)
        assert zeroes == Vector((0.0, 0.0))

    def test_uniform_with_value(self):
        ones = Vector.uniform(3, 1.0)
        assert ones == Vector((1.0, 1.0, 1.0))

    def test_add(self):
        vector_1 = Vector((1.0, 2.0, 3.0))
        vector_2 = Vector((4.0, 5.0, 6.0))
        expected_result = Vector((5.0, 7.0, 9.0))
        assert vector_1 + vector_2 == expected_result
        assert vector_2 + vector_1 == expected_result

    def test_add_different_sized_raises_ValueError(self):
        vector_1 = Vector((1.0, 2.0))
        vector_2 = Vector((4.0, 5.0, 6.0))

        def add():
            return vector_1 + vector_2

        assert_raises(
            add,
            ValueError,
            "Addition with different sized vectors should raise a ValueError",
        )

    def test_sub(self):
        vector_1 = Vector((1.0, 2.0, 3.0))
        vector_2 = Vector((4.0, 5.0, 6.0))

        expected_result = Vector((-3.0, -3.0, -3.0))
        actual_result = vector_1 - vector_2
        assert expected_result == actual_result

        expected_result = Vector((3.0, 3.0, 3.0))
        assert vector_2 - vector_1 == expected_result

    def test_sub_different_sized_raises_ValueError(self):
        vector_1 = Vector((1.0, 2.0))
        vector_2 = Vector((4.0, 5.0, 6.0))

        def sub():
            return vector_1 - vector_2

        assert_raises(
            sub,
            ValueError,
            "Subtraction with different sized vectors should raise ValueError",
        )

    def test_magnitude(self):
        vector = Vector((4.0, 3.0))
        expected_magnitude = 5.0
        assert vector.magnitude == expected_magnitude

    def test_truediv(self):
        vector = Vector((4.0, 3.0))
        divisor = 2.0
        expected_vector = Vector((4.0 / divisor, 3.0 / divisor))
        assert vector / divisor == expected_vector

    def test_truediv_by_zero_raises_ZeroDivisionError(self):
        def divide():
            return Vector((4.0, 3.0)) / 0

        assert_raises(
            divide,
            ZeroDivisionError,
            "Expect division by 0 to result in ZeroDivisionError",
        )

    def test_mul(self):
        values = (4.0, 3.0)
        vector = Vector(values)
        scalar = 2.0
        expected_result = Vector(tuple(scalar * value for value in values))
        assert vector * scalar == expected_result
        assert scalar * vector == expected_result

    def test_getitem(self):
        values = (4.0, 3.0)
        vector = Vector(values)

        assert len(values) == len(vector)
        for i in range(len(values)):
            assert values[i] == vector[i]

    def test_iter(self):
        values = (4.0, 3.0)
        vector = Vector(values)

        for value, vector_value in zip(values, vector):
            assert value == vector_value

    def test_neg(self):
        vector = Vector((-4.0, 3.0))
        neg_vector = Vector((4.0, -3.0))
        assert -vector == neg_vector

    def test_dot(self):
        vector_1 = Vector((3.0, 4.0))
        vector_2 = Vector((5.0, 6.0))
        expected_result = 15.0 + 24.0
        assert vector_1.dot(vector_2) == expected_result
        assert vector_2.dot(vector_1) == expected_result
        assert vector_1 @ vector_2 == expected_result
        assert vector_2 @ vector_1 == expected_result

    def test_dot_differentSize_raises_ValueError(self):
        def dot():
            return Vector((1.0,)).dot(Vector((2.0, 3.0)))

        assert_raises(
            dot,
            ValueError,
            "Expected dot product of vectors of different size to raise ValueError",
        )

    def test_random(self):
        random_gen_1 = Random(0)
        random_gen_2 = Random(0)

        vector = Vector.random(3, random_gen_1.random)
        expected_vector = Vector(
            (
                random_gen_2.random(),
                random_gen_2.random(),
                random_gen_2.random(),
            )
        )

        assert vector == expected_vector

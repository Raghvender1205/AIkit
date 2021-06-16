import random
import sys
import unittest

import aikit.utils


def foo():
    return 42


class Object:
    def foo(self):
        return 42


class Hook(aikit.utils.Hook):
    def __init__(self, module):
        super(Hook, self).__init__(module, 'foo')

    def __hook__(self):
        return 217


class TestHook(unittest.TestCase):
    def test_module_scope(self):
        assert foo() == 42

        with Hook(sys.modules[__name__]):
            assert foo() == 217

        assert foo() == 42

    def test_module_manual(self):
        hook = Hook(sys.modules[__name__])
        assert foo() == 42

        hook.enable()
        assert foo() == 217

        hook.disable()
        assert foo() == 42

    def test_object(self):
        o = Object()

        assert o.foo() == 42

        with Hook(o):
            assert o.foo() == 217

        assert o.foo() == 42

    def test_object_manual(self):
        o = Object()
        hook = Hook(o)
        assert o.foo() == 42

        hook.enable()
        assert o.foo() == 217

        hook.disable()
        assert o.foo() == 42

    def test_recursion(self):
        class Module:
            @staticmethod
            def foo(x):
                return x + 10

        def hook(x):
            return Module.foo(x) + 1

        for recursion in [True, False]:
            with aikit.utils.Hook(Module, 'foo', hook, recursion=recursion):
                if recursion:
                    with self.assertRaises(RecursionError):
                        Module.foo(0)
                else:
                    assert Module.foo(0) == 11


class TestRandom(unittest.TestCase):
    def test_string(self):
        for _ in range(50):
            length = aikit.utils.random.number(5, 10)
            a = aikit.utils.random.string(length=length)
            b = aikit.utils.random.string(length=length)

            assert len(a) == length
            assert len(b) == length

            assert a != b

    def test_strings(self):
        count = aikit.utils.random.number(5, 10)

        assert len(aikit.utils.random.strings(count=count)) == count

    def test_number(self):
        DEFAULT_MIN = 10
        DEFAULT_MAX = 100
        for i in range(10000):
            number = aikit.utils.random.number()  # no arguments
            self.assertGreaterEqual(number, DEFAULT_MIN)
            self.assertLessEqual(number, DEFAULT_MAX)

        MIN = 2
        MAX = 217
        for i in range(10000):
            number = aikit.utils.random.number(MIN, MAX)
            self.assertGreaterEqual(number, MIN)
            self.assertLessEqual(number, MAX)

    def test_shape(self):
        for i in range(10000):
            shape = aikit.utils.random.shape()
            self.assertIsInstance(shape, tuple)
            self.assertGreaterEqual(len(shape), 2)
            self.assertLessEqual(len(shape), 4)

            for size in shape:
                self.assertGreaterEqual(len(shape), 2)
                self.assertLessEqual(len(shape), 5)


class TestAttribute(unittest.TestCase):
    def test_scope_single(self):
        obj = Object()

        name = aikit.utils.random.string()
        value = aikit.utils.random.string()

        assert not hasattr(obj, name)

        with aikit.utils.Attribute(obj, name, value):
            assert getattr(obj, name) == value

        assert not hasattr(obj, name)

    def test_scope_multiple(self):
        obj = Object()

        count = aikit.utils.random.number(2, 10)

        names = aikit.utils.random.strings(count)
        values = aikit.utils.random.strings(count)

        for name in names:
            assert not hasattr(obj, name)

        with aikit.utils.Attribute(obj, names, values):
            for name, value in zip(names, values):
                assert getattr(obj, name) == value

        for name in names:
            assert not hasattr(obj, name)

    def test_rename(self):
        obj = Object()

        old = aikit.utils.random.string()
        new = aikit.utils.random.string()
        value = aikit.utils.random.string()

        setattr(obj, old, value)

        assert hasattr(obj, old)
        assert not hasattr(obj, new)

        aikit.utils.attribute.rename(obj, old, new)

        assert not hasattr(obj, old)
        assert getattr(obj, new) == value


class TestGPUs(unittest.TestCase):
    def test_not_available(self):
        assert not aikit.utils.gpus.available()

    def test_count_is_0(self):
        assert aikit.utils.gpus.count() == 0


if __name__ == '__main__':
    unittest.main()

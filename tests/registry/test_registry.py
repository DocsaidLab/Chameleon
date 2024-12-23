import pytest

from chameleon.registry import Registry

TEST = Registry(name='test')


def test_Registry():
    class Test1:
        def __init__(self):
            self.test = 1

    TEST.register_module('test1', module=Test1)

    test_obj = TEST.build({'name': 'test1'})
    assert test_obj.test == 1


def test_Registry_repr():
    class Test2:
        def __init__(self):
            self.test = 2
    TEST.register_module('test2', module=Test2)
    assert 'test2' in repr(TEST)


def test_Registry_list_module():
    modules = TEST.list_module("*2")
    assert 'test2' in modules

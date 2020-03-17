"""Various useful functions used throughout the package"""

import math
from copy import copy


def is_list_or_tuple(var):
    """Checks if an object is a list or a tuple"""
    return isinstance(var, list) or isinstance(var, tuple)


def assert_list_or_tuple(var):
    """Asserts that an object is a list or a tuple"""
    if not is_list_or_tuple(var):
        raise TypeError(f"variable should be a list or a tuple, not a {type(var)}")


def assert_str(var):
    """Asserts that an object is a str"""
    if not isinstance(var, str):
        raise TypeError(f"variable should be a str, not a {type(var)}")


def simplify(obj):
    """Tries to simplify an object"""
    if hasattr(obj, "simplify"):
        return obj.simplify()
    return copy(obj)


def expand(obj):
    """Tries to expand an object"""
    if hasattr(obj, "expand"):
        return obj.expand()
    return copy(obj)


def is_zero(obj):
    """Tries to check if an object is considered zero"""
    if isinstance(obj, int) or isinstance(obj, float):
        return math.isclose(obj, 0, abs_tol=1e-16)
    if hasattr(obj, "is_zero"):
        return obj.is_zero()
    return obj == 0


def is_one(obj):
    """Tries to check if an object is considered one"""
    if isinstance(obj, int) or isinstance(obj, float):
        return math.isclose(obj, 1, abs_tol=1e-16)
    if hasattr(obj, "is_one"):
        return obj.is_one()
    return obj == 1


def replace_var(obj, old_variable=None, new_variable=None):
    """Tries to replace a variable in an object."""
    if hasattr(obj, "replace_var"):
        if old_variable is None:
            new_obj = copy(obj)
            for old_variable in get_variables(obj):
                new_obj = replace_var(new_obj, old_variable=old_variable)
            return new_obj
        if new_variable is None:
            new_variable = old_variable + "'"
        return obj.replace_var(old_variable, new_variable)
    return copy(obj)


def get_variables(obj):
    """Tries to get variables in an object."""
    if hasattr(obj, "get_variables"):
        return obj.get_variables()
    return set([])


def has_variable(obj, variable):
    """Tries to check if an object has a variable."""
    if hasattr(obj, "has_variable"):
        return obj.has_variable(variable)
    return False

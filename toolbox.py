from copy import copy


def is_list_or_tuple(var):
    return isinstance(var, list) or isinstance(var, tuple)


def assert_list_or_tuple(var):
    if not is_list_or_tuple(var):
        raise TypeError(f"variable should be a list or a tuple, not a {type(var)}")


def assert_str(var):
    if not isinstance(var, str):
        raise TypeError(f"variable should be a str, not a {type(var)}")


def simplify(obj):
    if hasattr(obj, "simplify"):
        return obj.simplify()
    return copy(obj)


def expand(obj):
    if hasattr(obj, "expand"):
        return obj.expand()
    return copy(obj)


def is_zero(obj):
    if hasattr(obj, "is_zero"):
        return obj.is_zero()
    return obj == 0


def is_one(obj):
    if hasattr(obj, "is_one"):
        return obj.is_one()
    return obj == 1


def replace_var(obj, old_variable=None, new_variable=None):
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
    if hasattr(obj, "get_variables"):
        return obj.get_variables()
    return set([])

def is_list_or_tuple(var):
    return isinstance(var, list) or isinstance(var, tuple)

def assert_list_or_tuple(var):
    if not is_list_or_tuple(var):
        raise TypeError(f"variable should be a list or a tuple, not a {type(var)}")


def assert_str(var):
    if not isinstance(var, str):
        raise TypeError(f"variable should be a str, not a {type(var)}")

from states import BaseState

from toolbox import assert_str, assert_list_or_tuple


class FockOp:
    def __init__(self, variable, creation=True):
        assert_str(variable)
        self._variable = variable
        self._creation = creation

    def dagger(self):
        return FockOp(self._variable, not self._creation)


class BaseFockState(BaseState):
    def __init__(self, fock_ops):
        assert_list_or_tuple(fock_ops)
        self._fock_ops = []
        for fock_op in fock_ops:
            assert_fock_op(fock_op)
            self._fock_ops.append(fock_op)

    def inner_product(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError()
        # TODO Check overlap
        # TODO return DeltaFunction expr


def assert_fock_op(op):
    if not isinstance(op, FockOp):
        raise TypeError(f"operator should be of type FockOp, not {type(op)}")

from collections import defaultdict

from scalars import ComplexScalar, DeltaFunction, ProductOfScalars
from states import BaseState
from toolbox import assert_str, assert_list_or_tuple


class FockOp:
    def __init__(self, mode, variable, creation=True):
        assert_str(mode)
        assert_str(variable)
        self._mode = mode
        self._variable = variable
        self._creation = creation

    def _key(self):
        return self._mode, self._variable, self._creation

    def __eq__(self, other):
        if not isinstance(other, FockOp):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        dag = "+" if self._creation else ""
        return f"{self._mode}{dag}({self._variable})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self._mode}, {self._variable}, {self._creation})"

    def dagger(self):
        return FockOp(self._mode, self._variable, not self._creation)


class FockOpProduct:
    def __init__(self, fock_ops=None):
        self._fock_ops = defaultdict(int)
        if fock_ops is not None:
            assert_list_or_tuple(fock_ops)
            for fock_op in fock_ops:
                assert_fock_op(fock_op)
                # TODO we only allow these to be creation ops for now
                if not fock_op._creation:
                    raise NotImplementedError
                self._fock_ops[fock_op] += 1

    def __mul__(self, other):
        if isinstance(other, FockOp):
            new_op = FockOpProduct()
            for fock_op, count in self._fock_ops.items():
                new_op._fock_ops[fock_op] += count
            new_op._fock_ops[other] += 1
        elif isinstance(other, FockOpProduct):
            new_op = FockOpProduct()
            for fock_op, count in self._fock_ops.items():
                new_op._fock_ops[fock_op] += count
            for fock_op, count in other._fock_ops.items():
                new_op._fock_ops[fock_op] += count
            return new_op

    def __str__(self):
        to_print = ""
        for fock_op, count in self._fock_ops.items():
            to_print += f"{fock_op}^{count} * "
        return to_print[:-3]

    def _key(self):
        return tuple(sorted(self._fock_ops.items(), key=lambda x: x[0]._key()))

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._key() == other._key()

    def dagger(self):
        new_op = FockOpProduct()
        for fock_op, count in self._fock_ops.items():
            new_op[fock_op.dagger()] = count

        return new_op

    def variables_in_mode(self, mode):
        variables = []
        for fock_op, count in self._fock_ops.items():
            if fock_op._mode == mode:
                for _ in range(count):
                    variables.append(fock_op._variable)

        return variables

    def variables_by_modes(self):
        variables = defaultdict(list)
        for fock_op, count in self._fock_ops.items():
            mode = fock_op._mode
            for _ in range(count):
                variables[mode].append(fock_op._variable)

        return variables


class BaseFockState(BaseState):
    def __init__(self, fock_ops=None):
        if isinstance(fock_ops, FockOpProduct):
            self._fock_op_product = fock_ops
        else:
            self._fock_op_product = FockOpProduct(fock_ops)

    def _key(self):
        return self._fock_op_product._key()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        to_print = ""
        for fock_op, count in self._fock_op_product._fock_ops.items():
            to_print += f"{fock_op}^{count}"
        return to_print + "|0>"

    def inner_product(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError()
        l_vars_by_mode = self._fock_op_product.variables_by_modes()
        r_vars_by_mode = other._fock_op_product.variables_by_modes()
        scalar = ProductOfScalars()
        for mode, l_vars in l_vars_by_mode.items():
            r_vars = r_vars_by_mode.get(mode, [])
            if len(l_vars) != len(r_vars):
                return ComplexScalar(0)
            for l_var, r_var in zip(l_vars, r_vars):
                scalar *= DeltaFunction(l_var, r_var)

        return scalar

    def _compatible(self, other):
        if not isinstance(other, BaseFockState):
            return False
        return True

    def _bra_str(self):
        to_print = ""
        for fock_op, count in self._fock_op_product._fock_ops.items():
            to_print += f"{fock_op.dagger()}^{count}"
        return "<0|" + to_print


def assert_fock_op(op):
    if not isinstance(op, FockOp):
        raise TypeError(f"operator should be of type FockOp, not {type(op)}")


def test_fock_op_product():
    aw = FockOp('a', 'w')
    av = FockOp('a', 'v')

    print(FockOpProduct([aw, av]))
    print(FockOpProduct([aw, av, aw]))


def test_inner_product():
    aw = FockOp('a', 'w')
    av = FockOp('a', 'v')
    s1 = BaseFockState([aw])
    s2 = BaseFockState([av])
    print(s1.inner_product(s2))


if __name__ == '__main__':
    # test_fock_op_product()
    test_inner_product()

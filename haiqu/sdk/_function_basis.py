"""Evaluation of the basis functions a function-tomography fit is expressed in.

Each axis' basis is a family name or a list of expressions in ``x``. Basis functions are evaluated
at the given ``x`` as is, without mapping the interval onto the family's own domain.
"""

import itertools
from typing import Callable, List, Sequence, Union

import numpy as np
import sympy

from .constants import TOMOGRAPHY_FAMILIES

_X = sympy.Symbol("x")


def axis_basis(entry: Union[str, Sequence[str]], m: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable evaluating one axis' leading ``m`` basis functions.

    Args:
        entry: A family name (``"legendre"``, ``"chebyshev_t"`` or ``"fourier"``), or the axis' own
            basis functions as expressions in ``x``.
        m: Number of leading basis functions to evaluate.

    Returns:
        A callable taking a 1D array ``x`` and returning an array of shape ``(len(x), m)`` whose
        column ``j`` holds the ``j``-th basis function evaluated at ``x``.

    Raises:
        ValueError: If ``m`` is below 1, the family name is unknown, fewer than ``m`` expressions
            are given, or an expression cannot be parsed or uses a variable other than ``x``.
    """
    expressions = _leading_entries(entry, m)
    if entry == "legendre":
        return lambda x: np.polynomial.legendre.legvander(x, m - 1)
    if entry == "chebyshev_t":
        return lambda x: np.polynomial.chebyshev.chebvander(x, m - 1)
    if entry == "fourier":
        return lambda x: _fourier_rows(x, m)
    functions = [_compile(expression) for expression in expressions]
    return lambda x: np.stack([_column(function, x) for function in functions], axis=-1)


def axis_basis_expressions(entry: Union[str, Sequence[str]], m: int, symbol: sympy.Symbol) -> List[sympy.Expr]:
    """Return one axis' leading ``m`` basis functions as SymPy expressions in ``symbol``.

    Polynomial families come out as explicit polynomials, and Fourier terms as ``cos(pi*k*symbol)``
    and ``sin(pi*k*symbol)``.

    Args:
        entry: A family name (``"legendre"``, ``"chebyshev_t"`` or ``"fourier"``), or the axis' own
            basis functions as expressions in ``x``.
        m: Number of leading basis functions to return.
        symbol: The variable the expressions are written in, in place of ``x``.

    Returns:
        The ``m`` basis functions, in basis order.

    Raises:
        ValueError: As :func:`axis_basis`.
    """
    expressions = _leading_entries(entry, m)
    if entry == "legendre":
        return [sympy.legendre(j, symbol) for j in range(m)]
    if entry == "chebyshev_t":
        return [sympy.chebyshevt(j, symbol) for j in range(m)]
    if entry == "fourier":
        terms = [sympy.Integer(1)]
        for k in itertools.count(1):
            terms += [sympy.cos(sympy.pi * k * symbol), sympy.sin(sympy.pi * k * symbol)]
            if len(terms) >= m:
                return terms[:m]
    return [parse_expression(expression, "basis function").subs(_X, symbol) for expression in expressions]


def _leading_entries(entry: Union[str, Sequence[str]], m: int) -> List[str]:
    """Check an axis' basis against ``m``, returning its leading ``m`` expressions, or none for a family."""
    if m < 1:
        raise ValueError(f"the number of basis functions must be at least 1, got {m}")
    if isinstance(entry, str):
        check_family(entry)
        return []
    expressions = list(entry)
    if m > len(expressions):
        raise ValueError(
            f"The tomography result is inconsistent: {m} basis functions are needed, but only "
            f"{len(expressions)} are listed. Please contact support."
        )
    return expressions[:m]


def check_family(name: str) -> str:
    """Return ``name`` if it is a known basis family, raising ``ValueError`` otherwise."""
    if name not in TOMOGRAPHY_FAMILIES:
        raise ValueError(f"unknown basis family {name!r}; expected one of {TOMOGRAPHY_FAMILIES}")
    return name


def parse_expression(expression, what: str = "expression") -> sympy.Expr:
    """Parse a string or SymPy expression that may depend on the single variable ``x`` only.

    Args:
        expression: The expression, as a string or a SymPy object.
        what: How the expression is named in error messages.

    Raises:
        ValueError: If the expression cannot be parsed or uses a variable other than ``x``.
    """
    try:
        parsed = sympy.sympify(expression)
    except (sympy.SympifyError, SyntaxError, TypeError) as exc:
        raise ValueError(f"Could not parse `{what}` as a SymPy expression: {expression!r}") from exc
    extra_symbols = {symbol for symbol in parsed.free_symbols if symbol.name != "x"}
    if extra_symbols:
        raise ValueError(f"`{what}` must be an expression in the single variable `x`, but also got symbols {extra_symbols}")
    return parsed


def contract(coefficients: np.ndarray, rows: Sequence[np.ndarray]) -> np.ndarray:
    """Sum a coefficient tensor against per-axis basis values, one point at a time.

    Args:
        coefficients: Tensor of shape ``(m_1, ..., m_d)``.
        rows: One array per axis, axis ``a`` of shape ``(n, m_a)``, holding that axis' basis
            functions at each of the ``n`` points.

    Returns:
        Array of shape ``(n,)`` with ``sum c[j_1, ..., j_d] * prod_a rows[a][k, j_a]`` at each point ``k``.
    """
    n = rows[0].shape[0]
    values = rows[0] @ coefficients.reshape(coefficients.shape[0], -1)
    for axis in range(1, coefficients.ndim):
        # Spelled out rather than -1, which numpy cannot resolve once there are no points.
        rest = values.shape[1] // coefficients.shape[axis]
        values = np.sum(values.reshape(n, coefficients.shape[axis], rest) * rows[axis][:, :, None], axis=1)
    return values.reshape(n)


def contract_expressions(coefficients: np.ndarray, bases: Sequence[Sequence[sympy.Expr]]) -> sympy.Expr:
    """Sum a coefficient tensor against per-axis basis expressions.

    Args:
        coefficients: Tensor of shape ``(m_1, ..., m_d)``.
        bases: One sequence per axis, axis ``a`` holding that axis' ``m_a`` basis functions.

    Returns:
        The expression ``sum c[j_1, ..., j_d] * prod_a bases[a][j_a]``.
    """
    return sympy.Add(
        *(
            sympy.Float(float(value)) * sympy.Mul(*(basis[j] for basis, j in zip(bases, index)))
            for index, value in np.ndenumerate(coefficients)
        )
    )


def _fourier_rows(x: np.ndarray, m: int) -> np.ndarray:
    """Evaluate ``1, cos(pi x), sin(pi x), cos(2 pi x), sin(2 pi x), ...``, truncated at ``m`` terms."""
    columns = [np.ones_like(x, dtype=float)]
    k = 1
    while len(columns) < m:
        columns.append(np.cos(np.pi * k * x))
        if len(columns) < m:
            columns.append(np.sin(np.pi * k * x))
        k += 1
    return np.stack(columns, axis=-1)


def _compile(expression: str) -> Callable:
    """Parse an expression in ``x`` and turn it into a numpy-vectorised function."""
    return sympy.lambdify(_X, parse_expression(expression, "basis function"), modules=["numpy", "scipy"])


def _column(function: Callable, x: np.ndarray) -> np.ndarray:
    """Evaluate one basis function at ``x`` as a real column of ``x``'s length."""
    # A constant expression evaluates to a scalar, so it is broadcast to one value per point.
    values = np.broadcast_to(np.asarray(function(x)), x.shape)
    if np.iscomplexobj(values):
        if np.any(np.imag(values) != 0):
            raise ValueError("basis functions must be real-valued, but one evaluated to a complex number")
        values = np.real(values)
    return values.astype(float)

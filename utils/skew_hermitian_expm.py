import theano
import numpy as np

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    # some ops (e.g. Cholesky, Solve, A_Xinv_b) won't work
    imported_scipy = False

from theano import Op, Apply
from theano.tensor import as_tensor_variable


# hermitian conjugate of a matrix
def hconj(x):
    return np.conj(x).T


# extended variant of skew exp
def skew_expm(A):
    # compute iA = V diag(w) V^H, with diagonal of real eigenvalues w
    w, V = scipy.linalg.eigh(1j * A)
    # then A = V diah(-iw) V^H, with diagonal of imaginary eigenvalues -iw
    wimag = -w
    # exp(A) = V exp(diag(-iw)) V^H = V diag(cos(-iw) + i sin(-iw)) V^H
    return (V * (np.cos(wimag) + 1j * np.sin(wimag))).dot(hconj(V))


class SkewHermitianExpm(Op):
    """
    Compute the matrix exponential of a skew hermitian matrix.
    """

    __props__ = ()

    def make_node(self, A):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Expm op")

        A = as_tensor_variable(A)
        assert A.ndim == 2
        expm = theano.tensor.tensor3(name='keks', dtype=A.dtype)
        return Apply(self, [A, ], [expm, ])

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (expm,) = outputs
        # expm[0] = 10 * A

        mat = np.tril(A, -1) + 1j * np.triu(A, 0).T - hconj(np.tril(A, -1) + 1j * np.triu(A, 1).T)
        # w, V = scipy.linalg.eigh(1j * mat)
        # wimag = -w
        # temp = (V * (np.cos(wimag) + 1j * np.sin(wimag))).dot(hconj(V))
        temp = skew_expm(mat)
        expm[0] = np.stack([temp.real, temp.imag])

    def grad(self, inputs, outputs):
        (A,) = inputs
        (g_out,) = outputs
        return [SkewHermitianExpmGrad()(A, g_out)]

    def infer_shape(self, node, shapes):
        return [(2,) + shapes[0]]


class SkewHermitianExpmGrad(Op):
    """
    Gradient of the matrix exponential of a square array.
    """

    __props__ = ()

    def make_node(self, A, gw):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Expm op")
        A = as_tensor_variable(A)
        gw = as_tensor_variable(gw)
        assert A.ndim == 2
        out = theano.tensor.matrix(name='peks', dtype=A.dtype)
        return Apply(self, [A, gw], [out, ])

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        # Kalbfleisch and Lawless, J. Am. Stat. Assoc. 80 (1985) Equation 3.4
        # Kind of... You need to do some algebra from there to arrive at
        # this expression.

        (A, gA) = inputs
        (out,) = outputs

        mat = np.tril(A, -1) + 1j * np.triu(A, 0).T - hconj(np.tril(A, -1) + 1j * np.triu(A, 1).T)
        gmat = gA[0, ...] + 1j * gA[1, ...]
        # gmat = np.tril(gA, -1) + 1j * np.triu(gA, 0).T - hconj(np.tril(gA, -1) + 1j * np.triu(gA, 1).T)
        w, V = scipy.linalg.eigh(1j * mat)
        # print('k: {}'.format(np.allclose(V @ np.diag(w) @ hconj(V), mat)))
        w = -1j * w
        # U = scipy.linalg.inv(V)
        U = hconj(V)

        exp_w = np.exp(w)
        X = np.subtract.outer(exp_w, exp_w) / np.subtract.outer(w, w)
        np.fill_diagonal(X, exp_w)
        Y = hconj(V.dot(U.dot(hconj(gmat)).dot(V) * X).dot(U))
        assert Y.shape == A.shape
        dA = np.zeros_like(A, dtype=A.dtype)
        dA += np.tri(*dA.shape, -1) * Y.real  # lower triag real part derivative
        dA -= (np.tri(*dA.shape, -1).T * Y.real).T  # upper triag real part derivative (with minus sign)
        dA += (np.tri(*dA.shape, 0) * Y.imag).T  # lower triag imag part derivative (with diagonal elements)
        dA += np.tri(*dA.shape, -1).T * Y.imag  # upper triag imag part derivatie
        out[0] = dA.astype(A.dtype)


skew_hermitian_expm = SkewHermitianExpm()
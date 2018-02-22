import theano
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from theano import tensor

from manifolds.manifold import Manifold
from utils.complex_expm import complex_expm
from utils.skew_hermitian_expm import skew_hermitian_expm
from utils.theano_complex_extension import frac, identity, zeros, complex_dot, hconj, skew_frac, skew_hermitian_parametrized

from theano.tensor.shared_randomstreams import RandomStreams

srnd = RandomStreams(rnd.randint(0, 1000))


class ImplicitUnitary(Manifold):
    """
    Returns a manifold struct to optimize over the set of subspaces in C^n.

    function M = grassmanncomplexfactory(n, p)
    function M = grassmanncomplexfactory(n, p, k)

    Complex Grassmann manifold: each point on this manifold is a collection
    of k vector subspaces of dimension p embedded in C^n.

    The metric is obtained by making the Grassmannian a Riemannian quotient
    manifold of the complex Stiefel manifold, i.e., the manifold of
    orthonormal matrices, itself endowed with a metric by making it a
    Riemannian submanifold of the Euclidean space, endowed with the usual
    real-trace inner product, that is, it is the usual metric for the complex
    plane identified with R^2.

    This structure deals with complex matrices X of size n x p x k
    (or n x p if k = 1, which is the default) such that each n x p matrix is
    orthonormal, i.e., X'*X = eye(p) if k = 1, or X(:, :, i)' * X(:, :, i) =
    eye(p) for i = 1 : k if k > 1. Each n x p matrix is a numerical
    representation of the vector subspace its columns span.

    By default, k = 1.

    See also: grassmannfactory, stiefelcomplexfactory, grassmanngeneralizedfactory

    This file is part of Manopt: www.manopt.org.
    Original author: Hiroyuki Sato, May 21, 2015.
    Contributors:
    Change log:
    """
    def __init__(self, n, retr_mode="exp"):
        if n <= 0:
            raise ValueError('n must be at least 1')
        if retr_mode not in ["exp"]:
            raise ValueError('retr_type mist be "exp", but is "{}"'.format(retr_mode))
        self.retr_mode = retr_mode
        self._n = n
        # I didn't implement it for k > 1
        self._name = 'Unitary manifold U({}) = St({}, {})'.format(n, n, n)

    @property
    def name(self):
        return self._name

    @property
    def short_name(self):
        return "Unitary({})".format(self._n)

    @property
    def dim(self):
        return self._k * (2 * self._n * self._p - self._p**2)

    @property
    def typicaldist(self):
        return np.sqrt(self._p, self._k)

    def inner(self, X, G, H):
        GR, GI = skew_frac(G)
        HR, HI = skew_frac(H)
        # (AR + iAI)(BR + iBI) = ARBR - AIBI + i(ARBI + AIBR)
        # we return only real part of sum(hadamard(G, H))
        # old # return T.real(T.sum((GR + 1j * GI) *(HR + 1j * HI)))
        return tensor.sum(GR * HR - GI * HI)

    def norm(self, X, G):
        GR, GI = skew_frac(G)
        return (GR + 1j * GI).norm()

    def dist(self, X, Y):
        raise NotImplementedError

    def herm(self, X):
        Xman = skew_hermitian_expm(X)
        XH = hconj(Xman)
        return 0.5 * (Xman + XH)

    def proj(self, X, U):
        # return skew-hermitian parametrized matrix
        Xman = skew_hermitian_expm(X)
        projection = complex_dot(U, Xman) - complex_dot(Xman, U)
        return skew_hermitian_parametrized(projection)

    def tangent(self, X, U):
        return self.proj(X, U)

    def egrad2rgrad(self, X, U):
        return self.proj(X, U)

    def ehess2rhess(self, X, egrad, ehess, H):
        # TODO implement this for future
        XHG = complex_dot(hconj(X), egrad)
        #XHG = X.conj().dot(egrad)
        herXHG = self.herm(XHG)
        HherXHG = complex_dot(H, herXHG)
        rhess = self.proj(X, ehess - HherXHG)
        return rhess

    def retr(self, X, U, mode="default"):
        if mode == "exp":
            return self.exp(X, U)
        elif mode == "default":
            return self.retr(X, U, self.retr_mode)
        else:
            raise ValueError('mode must equal to "exp" or "default", but "{}" is given'.format(mode))

    def concat(self, arrays, axis):
        return tensor.concatenate(arrays, axis=axis+1)

    def exp(self, X, U):
        # The exponential (in the sense of Lie group theory) of a tangent
        # vector U at X.
        # in implicit form we have unchanged skew-hermitian matrix returned.
        # for explicit unitary matrix use self.explicit
        #return skew_hermitian_expm(U)
        return U

    def log(self, X, Y):
        # The logarithm (in the sense of Lie group theory) of Y. This is the
        # inverse of exp.
        raise NotImplementedError

    def explicit(self, U):
        return skew_hermitian_expm(U)

    def rand_np(self):
        return rnd.normal(size=(self._n, self._n))

    def zeros(self):
        raise NotImplementedError('Unitary manifold have no zero elements')

    def identity(self):
        return tensor.zeros((self._n, self._n))

    def identity_np(self):
        return np.zeros((self._n, self._n))

    def rand(self):
        return srnd.normal(size=(self._n, self._n))

    def randvec(self, X):
        randvec_embedding = tensor.stack([rnd.normal(size=(self._n, self._n)),
                                          1j * rnd.normal(size=(self._n, self._n))])
        U = self.proj(X, randvec_embedding)
        # should we scale that?
        # U = U / np.log(self.norm(X, U))
        return U

    def zerovec(self, X):
        return tensor.zeros_like(X)

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        if u2 is None and a2 is None:
            return a1 * u1
        elif None not in [a1, u1, a2, u2]:
            return a1 * u1 + a2 * u2
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')
from __future__ import division, absolute_import, print_function

import re
import six
import numpy as np
from scipy import special

try:
    from scipy.special import cython_special
except ImportError:
    pass

from .common import Benchmark, with_attributes


FUNC_ARGS = {
    'airy_d': (1,),
    'airy_D': (1,),
    'beta_dd': (0.25, 0.75),
    'erf_d': (1,),
    'erf_D': (1+1j,),
    'exprel_d': (1e-6,),
    'gamma_d': (100,),
    'gamma_D': (100+100j,),
    'jv_dd': (1, 1),
    'jv_dD': (1, (1+1j)),
    'loggamma_D': (20,),
    'logit_d': (0.5,),
    'psi_d': (1,),
    'psi_D': (1,),
}


class _CythonSpecialMeta(type):
    """
    Add time_* benchmarks corresponding to cython_special._bench_*_cy
    """

    def __new__(cls, cls_name, bases, dct):
        params = [(10, 100, 1000), ('python', 'numpy', 'cython')]
        param_names = ['N', 'api']

        def get_time_func(name, args):
            py_func = getattr(cython_special, '_bench_{}_py'.format(name))
            cy_func = getattr(cython_special, '_bench_{}_cy'.format(name))

            m = re.match('^(.*)_[dDl]+$', name)
            np_func = getattr(special, m.group(1))

            @with_attributes(params=[(args,)] + params, param_names=['argument'] + param_names)
            def func(self, args, N, api):
                if api == 'python':
                    py_func(N, *args)
                elif api == 'numpy':
                    np_func(*self.obj)
                else:
                    cy_func(N, *args)

            func.__name__ = 'time_' + name
            return func

        for name in dir(cython_special):
            m = re.match('^_bench_(.*)_cy$', name)
            if m:
                name = m.group(1)
                func = get_time_func(name, FUNC_ARGS[name])
                dct[func.__name__] = func

        return type.__new__(cls, cls_name, bases, dct)


class CythonSpecial(six.with_metaclass(_CythonSpecialMeta)):
    def setup(self, args, N, api):
        self.obj = []
        for arg in args:
            self.obj.append(arg*np.ones(N))
        self.obj = tuple(self.obj)

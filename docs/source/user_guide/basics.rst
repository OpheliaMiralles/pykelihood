Basics
==========

The most basic thing you can use ``pykelihood`` for is creating and manipulating distributions as objects.
The distribution parameters can be accessed like standard Python attributes. Sampling from the distribution or computing
the quantiles can be done using the same semantics as with the Python package ``scipy.stats``.

>>> from pykelihood.distributions import Normal
>>> n = Normal(1, 2)
>>> n
Normal(loc=1.0, scale=2.0)

``n`` is an *object* of type ``Normal``. It has 2 parameters, ``loc`` and ``scale``. They can be accessed like standard
Python attributes:

>>> n.loc
1.0

Using the ``Normal`` object, you can calculate standard values using the same semantics as **scipy.stats**:

>>> n.pdf([0, 1, 2])
array([0.17603266, 0.19947114, 0.17603266])
>>> n.cdf([0, 1, 2])
array([0.30853754, 0.5       , 0.69146246])

Or you can also generate random values according to this distribution:

>>> n.rvs(10)
array([ 3.31370986,  5.02699468, -0.3573229 ,  1.00460378, -3.26044871,
        1.86362711, -0.84192901,  0.81132182, -2.03266978,  1.48079944])


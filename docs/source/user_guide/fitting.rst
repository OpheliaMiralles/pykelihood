Fitting
==========

To fit the distribution to data, the syntax is simply ``distribution.fit(data)``, where fixed parameters can be added as
additional arguments to the fit, as in the package ``scipy.stats``.

Let's generate a larger sample from our previous object:

>>> data = Normal(1, 2).rvs(1000)
>>> data.mean()
1.025039359276458
>>> data.std()
1.9376460645596842

We can fit a ``Normal`` distribution to this data, which will return another ``Normal`` object:

>>> Normal.fit(data)
Normal(loc=1.0250822420920338, scale=1.9376400770300832)

As you can see, the values are slightly different from the moments in the data.
This is due to the fact that the ``fit`` method returns the Maximum Likelihood Estimator (MLE)
for the data, and is thus the result of an optimisation (using **scipy.optimize**). Custom optimizer and arguments passed
to ``scipy.optimize.minimize`` can be passed as ``kwargs`` to the ``fit`` method of any distribution.

The syntax ``distribution.fit(data, loc=0)`` can be used to fit the distribution to the data while keeping the ``loc``
parameter null:

>>> Normal.fit(data, loc=1)
Normal(loc=1.0, scale=1.9377929687500024)

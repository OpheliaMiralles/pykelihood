Some notes/remarks/ideas/general thoughts.


# Named parameters

For a while I was thinking it could be a good idea to put the names of the parameters inside the parameter.
Then a distribution would hold a **list** of parameters, and would iterate through that list to find a parameter
that matches a given name. That's what SciPy does.
It turns it's not a good idea for pykelihood. The reason is that a "parameter" to a distribution isn't necessarily
a Parameter, it can be a kernel, or any container of parameters. So we would have to attach a name to all objects,
which doesn't really make sense since they can be created standalone.
So the current approach where parameter-containers hold a **map** from name to _some object with parameters_ is the
correct paradigm.

# Parameterized protocol

Speaking of these objects, the current infrastructure is way more complex than necessary. We have params, params_names,
optimisation_params(_dict), etc. What we really need is an **ordered mapping**. Mapping to provide the names,
ordered to ensure the _flattened_ representation is deterministic. Then we can have some utilities that turn that mapping
into whatever shape is useful.

# Parameter protocol

A parameter should be a simple protocol. It needs a value, a way to set that value from a flattened representation, and its size.
The size is useful for the sparse case where we need to know how many values belong to a given parameter.

# Parameter domains

We should ideally set bounds on the parameters, that makes the optimization when fitting more robust and avoids
going out of the disribution's support. The domain should be some metadata on the distribution, SciPy stores it so we
can probably extract that.
The issue is that for complex parameters such as kernels, the relation between the distribution's parameters' domain and the
kernel's parameters' domains can be complex and not 1-dimensional.
To still incorporate the domain, we can rely on constraint-aware algorithms, e.g., COBYQA, or manually add a penalty for out-of-domain
values.
Not sure what is best at the moment.

# Sparse parameters

Ideally we should be able to handle sparse parameters, i.e., whose value is a sparse array.
A related idea is to be able to optimize only some values in an n-dimensional parameter, n>0.
For example, estimating the correlation matrix only needs 1 parameter, the diagonal is fixed to 1.
For this, a sparse array is not sufficient, we need to sum 2 arrays together with one fixed and one
variable. Or perhaps this is a specific requirement that can be handled case-by-case by implementing
the Parameter-protocol?

# Constraints

When fitting, it can be interesting to add constraints. The most obvious one is setting the value for a parameter to some
fixed value, and optimizing over the rest.
We currently have the constant parameter that acts as a "global" constraint (it applies to all fits). Is it good?
How can we handle other types of constraints? For example, loc == shape in a Normal distribution.
This means we can reduce the number of parameters in the optimization. We can build a DAG of parameters,
and on each step map that DAG back to the parameter values.
We can't (or don't want to) support a full constraint algebra, only cases where the value of a parameter can be
expressed as a function of the values of the other parameters. So not 2*loc == 4*shape, but loc == 2*shape.

# n-dimensional kernels

When everything is 0- and 1-dimensional, everything works nicely because broadcasting is intuitive.
If we reach 2+-d, we need to start thinking about axes. For example, what happens if we apply the linear
kernel to a matrix? We currently apply the linear op to all the matrix, but maybe it should only apply
to the last axis?
We need to think in terms of the usual dimension, i.e. (batch, time, x, y, variable) for spatio-temporal data.
The example use-case for linear applies to time.
We also have to check how SciPy handles n-d parameters, we know it can handle 1-d but not sure what happens afterwards.

# Multiple parameterizations

The SciPy v2 distribution infrastructure support having multiple parameterizations for a given distribution.
We currently handle that by creating new distributions, which means giving it a new name and overriding all
the required methods.
Can we use a model closer to SciPy? Should we?

# Bayesian

We could put a distribution as a parameter in another distribution.
MLE doesn't make sense in that case (does it?), but it would be nice to be able to do something.
An option is to automatically build a PyMC model, then optimize that with MCMC.
That sounds very heavy and complicated, but it could be a nice extension.

# Inplace fitting

We currently create a new instance on each function evaluation. That's awful for performance.
We could add a parameter `inplace` to the fit method. When true, we don't create a new instance
and just set the value on the current parameters.
When false, we copy the instance once at the beginning, then run with inplace=True on the copy.
The problem with this is that it can introduce side-effects, for instance if a kernel instance
is used in multiple distributions. Maybe we shouldn't allow setting inplace to true, and always
do one copy?

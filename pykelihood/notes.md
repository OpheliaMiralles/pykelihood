import inspect
from scipy.stats import rv_continuous, rv_discrete
from pykelihood.utils import ifnone
from pykelihood.distributions import ScipyDistribution

def create_scipy_distributions():
    """
    Dynamically create pykelihood-compatible classes for all scipy distributions.
    """
    from scipy.stats import distributions as scipy_distributions
    from pykelihood.distributions import __all__ as pykelihood_all

    # Iterate over all attributes in scipy.stats.distributions
    for name, obj in inspect.getmembers(scipy_distributions):
        # Check if the object is a continuous or discrete distribution
        if isinstance(obj, type) and issubclass(obj, (rv_continuous, rv_discrete)):
            # Dynamically create a class
            class_name = name.capitalize()
            params = obj.shapes.split(", ") if obj.shapes else []
            params += ["loc", "scale"]

            # Define the new class
            class_dict = {
                "params_names": tuple(params),
                "base_module": obj,
                "__init__": lambda self, **kwargs: super(self.__class__, self).__init__(**kwargs),
                "_to_scipy_args": lambda self, **kwargs: {
                    param: kwargs.get(param, getattr(self, param)())
                    for param in self.params_names
                },
            }

            # Create the class and add it to the module
            new_class = type(class_name, (ScipyDistribution,), class_dict)
            globals()[class_name] = new_class
            pykelihood_all.append(class_name)

# Call the function to create and register all distributions
create_scipy_distributions()

class FitResult:
    """
    Encapsulates the results of fitting a distribution to data.

    Attributes
    ----------
    fitted : Distribution
        The fitted distribution.
    data : Obs
        The data used for fitting.
    score_fn : Callable[[Distribution, Obs], float]
        The scoring function used for fitting.
    x0 : Sequence[float]
        Initial parameters for fitting.
    optimize_result : OptimizeResult
        The result of the optimization process.
    """

    def __init__(
        self,
        fitted: Distribution,
        data: Obs,
        score_fn: Callable[[Distribution, Obs], float],
        x0: Sequence[float],
        optimize_result: OptimizeResult,
    ):
        self.fitted = fitted
        self.data = data
        self.score_fn = score_fn
        self.x0 = x0
        self.optimize_result = optimize_result

    def confidence_interval(self, param: str, alpha: float = 0.05, precision: float = 1e-5):
        """
        Calculate the confidence interval for a parameter.

        Parameters
        ----------
        param : str
            Name of the parameter.
        alpha : float, optional
            Significance level, by default 0.05.
        precision : float, optional
            Precision for the confidence interval calculation, by default 1e-5.

        Returns
        -------
        tuple
            Lower and upper bounds of the confidence interval.
        """
        from pykelihood.profiler import Profiler

        profiler = Profiler(
            self.fitted,
            self.data,
            self.score_fn,
            single_profiling_param=param,
            inference_confidence=1 - alpha,
        )
        return profiler.confidence_interval(param, precision=precision)

    def __getattr__(self, name):
        """
        Delegate attribute access to the fitted distribution.

        Parameters
        ----------
        name : str
            Name of the attribute.

        Returns
        -------
        Any
            The attribute from the fitted distribution.
        """
        return getattr(self.fitted, name)
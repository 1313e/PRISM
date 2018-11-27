.. _parameters:

Parameters
----------
Below are descriptions of all the parameters that can be provided to *PRISM* in a text-file when initializing the :class:`~prism.Pipeline` class (using the :attr:`~prism.Pipeline.prism_file` argument):

:attr:`~prism.Pipeline.n_sam_init` (Default: 500)
	Number of model evaluation samples that is used to construct the first iteration of the emulator.
	This value must be a positive integer.

:attr:`~prism._projection.Projection.proj_res` (Default: 25)
	Number of emulator evaluation samples that is used to generate the grid for the projection figures (it defines the resolution of the projection).
	This value must be a positive integer.

:attr:`~prism._projection.Projection.proj_depth` (Default: 250)
	Number of emulator evaluation samples that is used to generate the samples in every projected grid point (it defines the accuracy/depth of the projection).
	This value must be a positive integer.

:attr:`~prism.Pipeline.base_eval_sam` (Default: 800)
	Base number of emulator evaluation samples that is used to analyze an iteration of the emulator.
	It is multiplied by the iteration number and the number of model parameters to generate the true number of emulator evaluations, in order to ensure an increase in emulator accuracy.
	This value must be a positive integer.

:attr:`~prism.Emulator.sigma` (Default: 0.8)
	The Gaussian sigma/standard deviation that is used when determining the Gaussian contribution to the overall emulator variance.
	This value is only required when :attr:`~prism.Emulator.method` == ``'gaussian'``, as the Gaussian sigma is obtained from the residual variance left after the regression optimization if regression is included.
	This value must be non-zero.

:attr:`~prism.Emulator.l_corr` (Default: 0.3)
	The amplitude(s) of the Gaussian correlation length.
	This number is multiplied by the difference between the upper and lower value boundaries of the model parameters to obtain the Gaussian correlation length for every model parameter.
	This value must be positive and either a scalar or a list of :attr:`~prism.modellink.ModelLink.n_par` scalars (where the values corresponds to the sorted list of model parameters).

:attr:`~prism.Pipeline.impl_cut` (Default: [0.0, 4.0, 3.8, 3.5])
	A list of implausibility cut-off values that specifies the maximum implausibility values a parameter set is allowed to have to be considered 'plausible'.
	A zero can be used as a filler value, either taking on the preceding value or acting as a wildcard if the preceding value is a wildcard or non-existent.
	Zeros are appended at the end of the list if the length is less than the number of comparison data points, while extra values are ignored if the length is more.
	This must be a sorted list of positive values (excluding zeros).

:attr:`~prism.Pipeline.criterion` (Default: 'multi')
	The criterion to use for determining the quality of the LHDs that are used, represented by an integer, float, string or *None*.
	This parameter is the only non-*PRISM* parameter. Instead, it is used in the lhd()-function of the `e13Tools`_ package.
	By default, ``'multi'`` is used to give equal priority to maximizing minimum distances and minimizing the maximum correlation between pair-wise samples.

:attr:`~prism.Emulator.method` (Default: 'full')
	The method to use for constructing the emulator.
	``'gaussian'`` will only include Gaussian processes (no regression), which is much faster, but also less accurate.
	``'regression'`` will only include regression processes (no Gaussian), which is more accurate than Gaussian only, but underestimates the emulator variance by multiple orders of magnitude.
	``'full'`` includes both Gaussian and regression processes, which is slower than Gaussian only, but by far the most accurate both in terms of expectation and variance values.

	``'gaussian'`` can be used for faster exploration especially for simple models.
	``'regression'`` should only be used when the polynomial representation of a model is important and enough model realizations are available.
	``'full'`` should be used by default.
	
	.. warning::
	   When using *PRISM* on a model that can be described completely by the regression function (anything that has an analytical, polynomial form up to order :attr:`~prism.Emulator.poly_order` like a straight line or a quadratic function), use the ``'gaussian'`` method unless unavoidable (in which case :attr:`~prism.Pipeline.n_sam_init` and :attr:`~prism.Pipeline.base_eval_sam` must be set to very low values).

	   When using the regression method on such a model, *PRISM* will be able to capture the behavior of the model perfectly given enough samples, in which case the residual (unexplained) variance will be approximately zero and therefore :attr:`~prism.Emulator.sigma` as well.
	   This can occassionally cause floating point errors when calculating emulator variances, which in turn can give unexplainable artifacts in the adjustment terms, therefore causing answers to be incorrect.

	   Since *PRISM*'s purpose is to identify the characteristics of a model and therefore it does not know anything about its workings, it is not possible to automatically detect such problems.

:attr:`~prism.Emulator.use_regr_cov` (Default: False)
	Whether or not the regression variance should be taken into account for the variance calculations.
	The regression variance is the variance on the regression process itself and is only significant if a low number of model realizations (:attr:`~prism.Pipeline.n_sam_init` and :attr:`~prism.Pipeline.base_eval_sam`) is used to construct the emulator systems.
	Including it usually only has a very small effect on the overall variance value, while it can slow down the emulator evaluation rate by as much as a factor of 3.
	This value is not required if :attr:`~prism.Emulator.method` == ``'gaussian'`` and is automatically set to *True* if :attr:`~prism.Emulator.method` == ``'regression'``.
	This value must be a bool.

:attr:`~prism.Emulator.poly_order` (Default: 3)
	Up to which order all polynomial terms of all model parameters should be included in the active parameters and regression processes.
	This value is not required if :attr:`~prism.Emulator.method` == ``'gaussian'`` and :attr:`~prism.Pipeline.do_active_anal` == *False*.
	This value must be a positive integer.

:attr:`~prism.Pipeline.do_active_anal` (Default: True)
	Whether or not an active parameters analysis must be carried out for every iteration of every emulator system.
	If *False*, all potentially active parameters listed in :attr:`~prism.Pipeline.pot_active_par` will be active.
	This value must be a bool.

:attr:`~prism.Pipeline.freeze_active_par` (Default: True)
	Whether or not active parameters should be frozen in their active state.
	If *True*, parameters that have been considered active in a previous iteration of an emulator system, will automatically be active again (and skip any active parameters analysis).
	This value must be a bool.

:attr:`~prism.Pipeline.pot_active_par` (Default: None)
	A list of parameter names that indicate which parameters are potentially active.
	Potentially active parameters are the only parameters that will enter the active parameters analysis (or will all be active if :attr:`~prism.Pipeline.do_active_anal` == *False*).
	Therefore, all parameters not listed will never be considered active.
	If all parameters should be potentially active, then a *None* can be given.
	This must either be a list of parameter names or *None*.

:attr:`~prism.Emulator.use_mock` (Default: False)
	Whether or not mock data must be used as comparison data when constructing a new emulator.
	Mock data is calculated by evaluating the model for a randomly chosen set of parameter values, and adding the model discrepancy variances as noise to the returned data values.
	When using mock data for an emulator, it is not possible to change the comparison data in later emulator iterations.
	This value must be a bool.

.. _e13Tools: https://github.com/1313e/e13Tools

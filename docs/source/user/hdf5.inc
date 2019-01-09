.. _hdf5:

HDF5
++++
Whenever *PRISM* constructs an emulator, it automatically stores all the calculated data for it in an `HDF5-file`_ named ``'prism.hdf5'`` (default) in the designated working directory.
This file contains all the data that is required in order to recreate all emulator systems that have been constructed for the emulator belonging to this run.
If the :class:`~prism.Pipeline` class is initialized by using an HDF5-file made by *PRISM*, it will load in this data and return a :obj:`~prism.Pipeline` object in the same state as described in the file.

Below is a short overview of all the data that can be found inside a *PRISM* master HDF5-file.
HDF5-files can be viewed freely by the user using the `HDFView`_ application made available by `The HDFGroup`_.

.. _HDF5-file: https://portal.hdfgroup.org/display/HDF5/HDF5
.. _HDFView: https://portal.hdfgroup.org/display/HDFVIEW/HDFView
.. _The HDFGroup: https://portal.hdfgroup.org

----

The general file contains:
    - Attributes (9/10): Describe the general non-changeable properties of the emulator, which include:

        - Emulator type and method;
        - Gaussian parameters;
        - Name of used :class:`~prism.modellink.ModelLink` subclass;
        - Used *PRISM* version;
        - Polynomial order;
        - Bools for using mock data or regression covariance;
        - Mock data parameters if mock data was used.

    - Every emulator iteration has its own data group with the iteration number as its name.
      This data group stores all data/information specific to that iteration.

----

An iteration data group (``'i'``) contains:
    - Attributes (10): Describe the general properties and results of this iteration, including:

        - Active parameters for this emulator iteration;
        - Implausibility cut-off parameters;
        - Number of emulated data points, emulator systems, emulator evaluation samples, plausible samples and model realization samples;
        - Bools stating whether this emulator iteration contains plausible regions or used an external model realization set.

    - ``'emul_n'``: The data group that contains all data for a specific emulator system in this iteration.
      The value of ``'n'`` indicates which emulator system it is, not the data point.
      See below for its contents;
    - ``'impl_sam'``: The set of emulator evaluation samples that survived the implausibility checks and will be used to construct the next iteration;
    - ``'proj_hcube'``: The data group that contains all data for the (created) projections for this iteration, if at least one has been made. See below for its contents;
    - ``'sam_set'``: The set of model realization samples that were used to construct this iteration.
      In every iteration after the first, this is the ``'impl_sam'`` of the previous iteration;
    - ``'statistics'``: An empty data set that stores several different types of statistics as its attributes, including:

        - Size of the MPI communicator during various construction steps;
        - Average evaluation rate/time of the emulator and model;
        - Total time cost of most construction steps;
        - Percentage of parameter space that is still plausible within the iteration.

----

An emulator system data group (``'i/emul_n'``) contains:
    - Attributes (5+): List the details about the model comparison data point used in this emulator system, including:

        - Active parameters for this emulator system;
        - Data errors, identifiers, value space and value;
        - Regression score and residual variance if regression was used.

    - ``'cov_mat'``: The pre-calculated covariance matrix of all model evaluation samples in this emulator system.
      This data set is never used in *PRISM* and stored solely for user-convenience;
    - ``'cov_mat_inv'``: The pre-calculated inverse of ``'cov_mat'``;
    - ``'exp_dot_term'``: The pre-calculated second expectation adjustment dot-term (:math:`\mathrm{Var}\left(D\right)^{-1}\cdot\left(D-\mathrm{E}(D)\right)`) of all model evaluation samples in this emulator system.
    - ``'mod_set'``: The model outputs for the data point in this emulator system corresponding to the ``'sam_set'`` used in this iteration;
    - ``'poly_coef'`` (if regression is used): The non-zero coefficients for the polynomial terms in the regression function in this emulator system;
    - ``'poly_coef_cov'`` (if regression and regr_cov are used): The covariances for all polynomial coefficients ``'poly_coef'``;
    - ``'poly_idx'`` (if regression is used): The indices of the polynomial terms with non-zero coefficients if all active parameters are converted to polynomial terms;
    - ``'poly_powers'`` (if regression is used): The powers of the polynomial terms corresponding to ``'poly_idx'``.
      Both ``'poly_idx'`` and ``'poly_powers'`` are required since different methods of calculating the polynomial terms are used depending on the number of required terms and samples;
    - ``'prior_exp_sam_set'``: The pre-calculated prior expectation values of all model evaluation samples in this emulator system.
      This data set is also never used in *PRISM*.

----

A projections data group (``'i/proj_hcube'``) contains individual projection data groups (``'i/proj_hcube/<name>'``), which contain:
    - Attributes (4): List the general properties with which this projection was made, including:

        - Implausibility cut-off parameters (they can differ from the iteration itself);
        - Projection depth and resolution.

    - ``'impl_los'``: The calculated line-of-sight depth for all grid points in this projection;
    - ``'impl_min'``: The calculated minimum implausibility values for all grid points in this projection.

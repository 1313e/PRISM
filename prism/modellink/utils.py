# -*- coding: utf-8 -*-

"""
Utilities
=========
Provides several utility functions concerning the
:class:`~prism.modellink.ModelLink` class.

"""


# %% IMPORTS
# Built-in imports
from inspect import _VAR_KEYWORD, _VAR_POSITIONAL, _empty, isclass, signature
from os import path
import warnings

# Package imports
from e13tools import InputError
from e13tools.utils import check_instance, convert_str_seq
from mpi4pyd.MPI import get_HybridComm_obj
import numpy as np
from numpy.random import rand
from sortedcontainers import SortedDict as sdict

# PRISM imports
from prism._internal import RequestWarning, check_vals, np_array

# All declaration
__all__ = ['convert_data', 'convert_parameters', 'test_subclass']


# %% UTILITY FUNCTIONS
# This function converts provided model data into format used by PRISM
def convert_data(model_data):
    """
    Converts the provided `model_data` into a full data dict, taking into
    account all formatting options, and returns it.

    This function can be used externally to check how the provided `model_data`
    would be interpreted when provided to the
    :class:`~prism.modellink.ModelLink` subclass. Its output can be used for
    the 'model_data' input argument.

    Parameters
    ----------
    model_data : array_like, dict or str
        Anything that can be converted to a dict that provides model data
        information.

    Returns
    -------
    data_dict : dict
        Dict with the provided `model_data` converted to its full format.

    """

    # If a data file is given
    if isinstance(model_data, str):
        # Obtain absolute path to given file
        data_file = path.abspath(model_data)

        # Read the data file in as a string
        data_points = np.genfromtxt(data_file, dtype=(str), delimiter=':',
                                    autostrip=True)

        # Make sure that data_points is 2D
        data_points = np_array(data_points, ndmin=2)

        # Convert read-in data to dict
        model_data = dict(data_points)

    # If a data dict is given
    elif isinstance(model_data, dict):
        model_data = dict(model_data)

    # If anything else is given
    else:
        # Check if it can be converted to a dict
        try:
            model_data = dict(model_data)
        except Exception:
            raise TypeError("Input model data cannot be converted to type "
                            "'dict'!")

    # Make empty data_dict
    data_dict = dict()

    # Loop over all items in model_data
    for key, value in model_data.items():
        # Convert key to an actual data_idx
        idx = convert_str_seq(key)

        # Check if tmp_idx is not empty
        if not len(idx):
            raise InputError("Model data contains a data point with no "
                             "identifier!")

        # Convert value to an actual data point
        data = convert_str_seq(value)

        # Check if provided data value is valid
        val = check_vals(data[0], 'data_val%s' % (idx), 'float')

        # Extract data error and space
        # If length is two, centered error and no data space were given
        if(len(data) == 2):
            err = [check_vals(data[1], 'data_err%s' % (idx), 'float', 'pos')]*2
            spc = 'lin'

        # If length is three, there are two possibilities
        elif(len(data) == 3):
            # If the third column contains a string, it is the data space
            if isinstance(data[2], str):
                err = [check_vals(data[1], 'data_err%s' % (idx),
                                  'float', 'pos')]*2
                spc = data[2]

            # If the third column contains no string, it is error interval
            else:
                err = check_vals(data[1:3], 'data_err%s' % (idx),
                                 'float', 'pos')
                spc = 'lin'

        # If length is four+, error interval and data space were given
        else:
            err = check_vals(data[1:3], 'data_err%s' % (idx), 'float', 'pos')
            spc = data[3]

        # Check if valid data space has been provided
        spc = str(spc).replace("'", '').replace('"', '')
        if spc.lower() in ('lin', 'linear'):
            spc = 'lin'
        elif spc.lower() in ('log', 'log10', 'log_10'):
            spc = 'log10'
        elif spc.lower() in ('ln', 'loge', 'log_e'):
            spc = 'ln'
        else:
            raise ValueError("Input argument 'data_spc%s' is invalid (%r)!"
                             % (idx, spc))

        # Save data identifier as tuple or single element
        if(len(idx) == 1):
            idx = idx[0]
        else:
            idx = tuple(idx)

        # Add entire data point to data_dict
        data_dict[idx] = [val, *err, spc]

    # Return data_dict
    return(data_dict)


# This function converts provided model parameters into format used by PRISM
def convert_parameters(model_parameters):
    """
    Converts the provided `model_parameters` into a full parameters dict,
    taking into account all formatting options, and returns it.

    This function can be used externally to check how the provided
    `model_parameters` would be interpreted when provided to the
    :class:`~prism.modellink.ModelLink` subclass. Its output can be used for
    the 'model_parameters' input argument.

    Parameters
    ----------
    model_parameters : array_like, dict or str
        Anything that can be converted to a dict that provides model parameters
        information.

    Returns
    -------
    par_dict : dict
        Dict with the provided `model_parameters` converted to its full format.

    """

    # If a parameter file is given
    if isinstance(model_parameters, str):
        # Obtain absolute path to given file
        par_file = path.abspath(model_parameters)

        # Read the parameter file in as a string
        pars = np.genfromtxt(par_file, dtype=(str), delimiter=':',
                             autostrip=True)

        # Make sure that pars is 2D
        pars = np_array(pars, ndmin=2)

        # Convert read-in parameters to dict
        model_parameters = sdict(pars)

    # If a parameter dict is given
    elif isinstance(model_parameters, dict):
        model_parameters = sdict(model_parameters)

    # If anything else is given
    else:
        # Check if it can be converted to a dict
        try:
            model_parameters = sdict(model_parameters)
        except Exception:
            raise TypeError("Input model parameters cannot be converted to"
                            " type 'dict'!")

    # Initialize empty par_dict
    par_dict = sdict()

    # Loop over all items in model_parameters
    for i, (name, values_str) in enumerate(model_parameters.items()):
        # Convert values_str to values
        values = convert_str_seq(values_str)

        # Check if provided name is a string
        name = check_vals(name, 'par_name[%s]' % (name), 'str')

        # Check if provided range consists of two floats
        par_rng = check_vals(values[:2], 'par_rng[%s]' % (name), 'float')

        # Check if provided lower bound is lower than the upper bound
        if(par_rng[0] >= par_rng[1]):
            raise ValueError("Input argument 'par_rng[%s]' does not define a "
                             "valid parameter range (%f !< %f)!"
                             % (name, par_rng[0], par_rng[1]))

        # Check if a float parameter estimate was provided
        try:
            est = check_vals(values[2], 'par_est[%s]' % (name), 'float')
        # If no estimate was provided, save it as None
        except IndexError:
            est = None
        # If no float was provided, check if it was None
        except TypeError as error:
            # If it is None, save it as such
            if(values[2].lower() == 'none'):
                est = None
            # If it is not None, reraise the previous error
            else:
                raise error
        # If a float was provided, check if it is within parameter range
        else:
            if not(values[0] <= est <= values[1]):
                raise ValueError("Input argument 'par_est[%s]' is outside "
                                 "of defined parameter range!" % (name))

        # Add parameter to par_dict
        par_dict[name] = [*par_rng, est]

    # Return par_dict
    return(par_dict)


# This function tests a given ModelLink subclass
# TODO: Are there any more tests that can be done here?
def test_subclass(subclass, *args, **kwargs):
    """
    Tests a provided :class:`~prism.modellink.ModelLink` `subclass` by
    initializing it with the given `args` and `kwargs` and checking if all
    required methods can be properly called.

    This function needs to be called by all MPI ranks.

    Parameters
    ----------
    subclass : :class:`~prism.modellink.ModelLink` subclass
        The :class:`~prism.modellink.ModelLink` subclass that requires testing.
    args : tuple
        Positional arguments that need to be provided to the constructor of the
        `subclass`.
    kwargs : dict
        Keyword arguments that need to be provided to the constructor of the
        `subclass`.

    Returns
    -------
    modellink_obj : :obj:`~prism.modellink.ModelLink` object
        Instance of the provided `subclass` if all tests pass successfully.
        Specific exceptions are raised if a test fails.

    Note
    ----
    Depending on the complexity of the model wrapped in the given `subclass`,
    this function may take a while to execute.

    """

    # Import ModelLink class
    from prism.modellink import ModelLink

    # Check if provided subclass is a class
    if not isclass(subclass):
        raise InputError("Input argument 'subclass' must be a class!")

    # Check if provided subclass is a subclass of ModelLink
    if not issubclass(subclass, ModelLink):
        raise TypeError("Input argument 'subclass' must be a subclass of the "
                        "ModelLink class!")

    # Try to initialize provided subclass
    try:
        modellink_obj = subclass(*args, **kwargs)
    except Exception as error:
        raise InputError("Input argument 'subclass' cannot be initialized! "
                         "(%s)" % (error))

    # Check if modellink_obj was initialized properly
    if not check_instance(modellink_obj, ModelLink):
        obj_name = modellink_obj.__class__.__name__
        raise InputError("Provided ModelLink subclass %r was not "
                         "initialized properly! Make sure that %r calls "
                         "the super constructor during initialization!"
                         % (obj_name, obj_name))

    # Obtain list of arguments call_model should take
    call_model_args = list(signature(ModelLink.call_model).parameters)
    call_model_args.remove('self')

    # Check if call_model takes the correct arguments
    obj_call_model_args = dict(signature(modellink_obj.call_model).parameters)
    for arg in call_model_args:
        if arg not in obj_call_model_args.keys():
            raise InputError("The 'call_model()'-method in provided ModelLink "
                             "subclass %r does not take required input "
                             "argument %r!" % (modellink_obj._name, arg))
        else:
            obj_call_model_args.pop(arg)

    # Check if call_model takes any other arguments
    for arg, par in obj_call_model_args.items():
        # If this parameter has no default value and is not *args or **kwargs
        if(par.default == _empty and par.kind != _VAR_POSITIONAL and
           par.kind != _VAR_KEYWORD):
            # Raise error
            raise InputError("The 'call_model()'-method in provided ModelLink "
                             "subclass %r takes an unknown non-optional input "
                             "argument %r!" % (modellink_obj._name, arg))

    # Obtain list of arguments get_md_var should take
    get_md_var_args = list(signature(ModelLink.get_md_var).parameters)
    get_md_var_args.remove('self')

    # Check if get_md_var takes the correct arguments
    obj_get_md_var_args = dict(signature(modellink_obj.get_md_var).parameters)
    for arg in get_md_var_args:
        if arg not in obj_get_md_var_args.keys():
            raise InputError("The 'get_md_var()'-method in provided ModelLink "
                             "subclass %r does not take required input "
                             "argument %r!" % (modellink_obj._name, arg))
        else:
            obj_get_md_var_args.pop(arg)

    # Check if get_md_var takes any other arguments
    for arg, par in obj_get_md_var_args.items():
        # If this parameter has no default value and is not *args or **kwargs
        if(par.default == _empty and par.kind != _VAR_POSITIONAL and
           par.kind != _VAR_KEYWORD):
            # Raise an error
            raise InputError("The 'get_md_var()'-method in provided ModelLink "
                             "subclass %r takes an unknown non-optional input "
                             "argument %r!" % (modellink_obj._name, arg))

    # Set MPI intra-communicator
    comm = get_HybridComm_obj()

    # Obtain random sam_set on controller
    if not comm._rank:
        sam_set = modellink_obj._to_par_space(rand(1, modellink_obj._n_par))
    # Workers get dummy sam_set
    else:
        sam_set = []

    # Broadcast random sam_set to workers
    sam_set = comm.bcast(sam_set, 0)

    # Try to evaluate sam_set in the model
    try:
        # Check who needs to call the model
        if not comm._rank or modellink_obj._MPI_call:
            # Do multi-call
            if modellink_obj._multi_call:
                mod_set = modellink_obj.call_model(
                    emul_i=0,
                    par_set=sdict(zip(modellink_obj._par_name, sam_set.T)),
                    data_idx=modellink_obj._data_idx)

            # Single-call
            else:
                # Initialize mod_set
                mod_set = np.zeros([sam_set.shape[0], modellink_obj._n_data])

                # Loop over all samples in sam_set
                for i, par_set in enumerate(sam_set):
                    mod_set[i] = modellink_obj.call_model(
                        emul_i=0,
                        par_set=sdict(zip(modellink_obj._par_name, par_set)),
                        data_idx=modellink_obj._data_idx)

    # If call_model was not overridden, catch NotImplementedError
    except NotImplementedError:
        raise NotImplementedError("Provided ModelLink subclass %r has no "
                                  "user-written 'call_model()'-method!"
                                  % (modellink_obj._name))

    # If successful, check if obtained mod_set has correct shape
    if not comm._rank:
        mod_set = modellink_obj._check_mod_set(mod_set, 'mod_set')

    # Check if the model discrepancy variance can be obtained
    try:
        md_var = modellink_obj.get_md_var(
            emul_i=0,
            par_set=sdict(zip(modellink_obj._par_name, sam_set[0])),
            data_idx=modellink_obj._data_idx)

    # If get_md_var was not overridden, catch NotImplementedError
    except NotImplementedError:
        warn_msg = ("Provided ModelLink subclass %r has no user-written "
                    "'get_md_var()'-method! Default model discrepancy variance"
                    " description would be used instead!"
                    % (modellink_obj._name))
        warnings.warn(warn_msg, RequestWarning, stacklevel=2)

    # If successful, check if obtained md_var has correct shape
    else:
        md_var = modellink_obj._check_md_var(md_var, 'md_var')

    # Return modellink_obj
    return(modellink_obj)

# %% REMOVED IN v0.5.0
# Function for checking if a bool has been provided
@docstring_append(check_bool_doc)
def check_bool(value, name):
    # Check if bool is provided and return if so
    if(str(value).lower() in ('false', '0')):
        return(0)
    elif(str(value).lower() in ('true', '1')):
        return(1)
    else:
        logger.error("Input argument '%s' is not of type 'bool'!" % (name))
        raise TypeError("Input argument '%s' is not of type 'bool'!" % (name))


# Function for checking if a finite value has been provided
@docstring_append(check_fin_doc)
def check_finite(value, name):
    # Check if finite value is provided and return if so
    if np.isfinite(value):
        return(value)
    else:
        logger.error("Input argument '%s' is not finite!" % (name))
        raise ValueError("Input argument '%s' is not finite!" % (name))


# Function for checking if a float has been provided
@docstring_append(check_type_doc % ("a float"))
def check_float(value, name):
    # Check if finite value is provided
    value = check_finite(value, name)

    # Check if float is provided and return if so
    if isinstance(value, (int, float, np.integer, np.floating)):
        return(value)
    else:
        logger.error("Input argument '%s' is not of type 'float'!" % (name))
        raise TypeError("Input argument '%s' is not of type 'float'!" % (name))


# Function for checking if an int has been provided
@docstring_append(check_type_doc % ("an integer"))
def check_int(value, name):
    # Check if finite value is provided
    value = check_finite(value, name)

    # Check if int is provided and return if so
    if isinstance(value, (int, np.integer)):
        return(value)
    else:
        logger.error("Input argument '%s' is not of type 'int'!" % (name))
        raise TypeError("Input argument '%s' is not of type 'int'!" % (name))


# Function for checking if a provided float is negative
@docstring_append(check_val_doc % ("a negative float"))
def check_neg_float(value, name):
    # Check if float is provided
    value = check_float(value, name)

    # Check if float is negative
    if(value < 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not negative!" % (name))
        raise ValueError("Input argument '%s' is not negative!" % (name))


# Function for checking if a provided int is negative
@docstring_append(check_val_doc % ("a negative integer"))
def check_neg_int(value, name):
    # Check if int is provided
    value = check_int(value, name)

    # Check if int is negative
    if(value < 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not negative!" % (name))
        raise ValueError("Input argument '%s' is not negative!" % (name))


# Function for checking if a provided float is non-negative
@docstring_append(check_val_doc % ("a non-negative float"))
def check_nneg_float(value, name):
    # Check if float is provided
    value = check_float(value, name)

    # Check if float is non-negative
    if not(value < 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-negative!" % (name))
        raise ValueError("Input argument '%s' is not non-negative!" % (name))


# Function for checking if a provided int is non-negative
@docstring_append(check_val_doc % ("a non-negative integer"))
def check_nneg_int(value, name):
    # Check if int is provided
    value = check_int(value, name)

    # Check if int is non-negative
    if not(value < 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-negative!" % (name))
        raise ValueError("Input argument '%s' is not non-negative!" % (name))


# Function for checking if a provided float is non-positive
@docstring_append(check_val_doc % ("a non-positive float"))
def check_npos_float(value, name):
    # Check if float is provided
    value = check_float(value, name)

    # Check if float is non-positive
    if not(value > 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-positive!" % (name))
        raise ValueError("Input argument '%s' is not non-positive!" % (name))


# Function for checking if a provided int is non-positive
@docstring_append(check_val_doc % ("a non-positive integer"))
def check_npos_int(value, name):
    # Check if int is provided
    value = check_int(value, name)

    # Check if int is non-positive
    if not(value > 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-positive!" % (name))
        raise ValueError("Input argument '%s' is not non-positive!" % (name))


# Function for checking if a provided float is non-zero
@docstring_append(check_val_doc % ("a non-zero float"))
def check_nzero_float(value, name):
    # Check if float is provided
    value = check_float(value, name)

    # Check if float is non-zero
    if not(value == 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-zero!" % (name))
        raise ValueError("Input argument '%s' is not non-zero!" % (name))


# Function for checking if a provided int is non-zero
@docstring_append(check_val_doc % ("a non-zero integer"))
def check_nzero_int(value, name):
    # Check if int is provided
    value = check_int(value, name)

    # Check if int is non-zero
    if not(value == 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not non-zero!" % (name))
        raise ValueError("Input argument '%s' is not non-zero!" % (name))


# Function for checking if a provided float is positive
@docstring_append(check_val_doc % ("a positive float"))
def check_pos_float(value, name):
    # Check if float is provided
    value = check_float(value, name)

    # Check if float is positive
    if(value > 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not positive!" % (name))
        raise ValueError("Input argument '%s' is not positive!" % (name))


# Function for checking if a provided int is positive
@docstring_append(check_val_doc % ("a positive integer"))
def check_pos_int(value, name):
    # Check if int is provided
    value = check_int(value, name)

    # Check if int is positive
    if(value > 0):
        return(value)
    else:
        logger.error("Input argument '%s' is not positive!" % (name))
        raise ValueError("Input argument '%s' is not positive!" % (name))


# Function for checking if a str has been provided
@docstring_append(check_type_doc % ("a string"))
def check_str(value, name):
    # Check if str is provided and return if so
    if isinstance(value, (str, np.string_, unicode)):
        return(value)
    else:
        logger.error("Input argument '%s' is not of type 'str'!" % (name))
        raise TypeError("Input argument '%s' is not of type 'str'!" % (name))


# %% REMOVED in v0.5.7
# This function checks if the input values meet all given criteria
def check_vals(values, name, *args):
    """
    Checks if all values in provided input argument `values` with `name` meet
    all criteria given in `args`. If no criteria are given, it is checked if
    all values are finite.
    Returns `values` (0 or 1 in case of bool) if *True* and raises a
    :class:`~ValueError` or :class:`~TypeError` if *False*.

    Parameters
    ----------
    values : array_like of {int, float, str, bool}
        The values to be checked against all given criteria in `args`.
    name : str
        The name of the input argument, which is used in the error message if
        a criterion is not met.
    args : tuple of {'bool', 'float', 'int', 'neg', 'nneg', 'normal', 'npos', \
        'nzero', 'pos', 'str'}
        Sequence of strings determining the criteria that `values` must meet.
        If `args` is empty, it is checked if `values` are finite.

    Returns
    -------
    return_values : array_like of {int, float, str}
        If `args` contained 'bool', returns 0 or 1. Else, returns `values`.

    Notes
    -----
    If `values` is array_like, every element is replaced by its checked values
    (0s or 1s in case of bools, or ints converted to floats in case of floats).
    Because of this, a copy will be made of `values`. If this is not possible,
    `values` is adjusted in place.

    """

    # Define logger
    logger = getRLogger('CHECK')

    # Convert args to a list
    args = list(args)

    # Check ndim of values and iterate over values if ndim > 0
    if np.ndim(values):
        # If values is a NumPy array, make empty copy and upcast if necessary
        if isinstance(values, np.ndarray):
            if 'bool' in args or 'int' in args:
                values_copy = np.empty_like(values, dtype=int)
            elif 'float' in args:
                values_copy = np.empty_like(values, dtype=float)
            elif 'str' in args:
                values_copy = np.empty_like(values, dtype=str)
            else:
                values_copy = np.empty_like(values)

        # If not a NumPy array, make a normal copy
        else:
            # Check if values has the copy()-method and use it if so
            try:
                values_copy = values.copy()
            # Else, use the built-in copy() function
            except AttributeError:
                values_copy = copy(values)

        # Iterate over first dimension of values
        for idx, value in enumerate(values):
            # Check value
            values_copy[idx] = check_vals(value, '%s[%i]' % (name, idx), *args)

        # Return values
        return(values_copy)

    # If ndim == 0, set value to values
    else:
        value = values

    # Check for bool
    if 'bool' in args:
        # Check if bool is provided and return if so
        if(str(value).lower() in ('false', '0')):
            return(0)
        elif(str(value).lower() in ('true', '1')):
            return(1)
        else:
            err_msg = "Input argument %r is not of type 'bool'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for string
    elif 'str' in args:
        # Check if str is provided and return if so
        if isinstance(value, (str, np.string_, unicode)):
            return(value)
        else:
            err_msg = "Input argument %r is not of type 'str'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for float
    elif 'float' in args:
        # Check if float is provided and return if so
        if isinstance(value, (int, float, np.integer, np.floating)):
            # Remove 'float' from args and check it again
            args.remove('float')
            value = check_vals(value, name, *args)
            return(float(value))
        else:
            err_msg = "Input argument %r is not of type 'float'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for integer
    elif 'int' in args:
        # Check if int is provided and return if so
        if isinstance(value, (int, np.integer)):
            # Remove 'int' from args and check it again
            args.remove('int')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not of type 'int'!" % (name)
            raise_error(TypeError, err_msg, logger)

    # Check for negative value
    elif 'neg' in args:
        # Check if value is negative and return if so
        if(value < 0):
            # Remove 'neg' from args and check it again
            args.remove('neg')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not negative!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for non-negative value
    elif 'nneg' in args:
        # Check if value is non-negative and return if so
        if not(value < 0):
            # Remove 'nneg' from args and check it again
            args.remove('nneg')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not non-negative!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for normalized value [0, 1]
    elif 'normal' in args:
        # Check if value is normalized and return if so
        if(0 <= value <= 1):
            # Remove 'normal' from args and check it again
            args.remove('normal')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not normalized!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for non-positive value
    elif 'npos' in args:
        # Check if value is non-positive and return if so
        if not(value > 0):
            # Remove 'npos' from args and check it again
            args.remove('npos')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not non-positive!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for non-zero value
    elif 'nzero' in args:
        # Check if value is non-zero and return if so
        if not(value == 0):
            # Remove 'nzero' from args and check it again
            args.remove('nzero')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not non-zero!" % (name)
            raise_error(ValueError, err_msg, logger)

    # Check for positive value
    elif 'pos' in args:
        # Check if value is positive and return if so
        if(value > 0):
            # Remove 'pos' from args and check it again
            args.remove('pos')
            value = check_vals(value, name, *args)
            return(value)
        else:
            err_msg = "Input argument %r is not positive!" % (name)
            raise_error(ValueError, err_msg, logger)

    # If no criteria are given, it must be a finite value
    elif not len(args):
        # Check if finite value is provided and return if so
        try:
            if np.isfinite(value):
                return(value)
        except Exception:
            pass
        err_msg = "Input argument %r is not finite!" % (name)
        raise_error(ValueError, err_msg, logger)

    # If none of the criteria is found, the criteria are invalid
    else:
        err_msg = "Input argument 'args' is invalid!"
        raise_error(InputError, err_msg, logger)

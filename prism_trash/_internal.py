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
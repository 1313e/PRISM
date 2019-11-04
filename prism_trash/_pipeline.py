# %% REMOVED IN v0.4.12
# This function extracts the set of active parameters
# TODO: Perform exhaustive backward stepwise regression on order > 1
@docstring_substitute(emul_i=std_emul_i_doc)
def _get_active_par(self, emul_i):
    """
    Determines the active parameters to be used for every individual data
    point in the provided emulator iteration `emul_i`. Uses backwards
    stepwise elimination to determine the set of active parameters.

    Parameters
    ----------
    %(emul_i)s

    Generates
    ---------
    active_par : 1D :obj:`~numpy.ndarray` object
        Array containing the indices of all the parameters that are active
        in the emulator iteration `emul_i`.
    active_par_data : List of 1D :obj:`~numpy.ndarray` objects
        List containing the indices of all the parameters that are active
        in the emulator iteration `emul_i` for every individual data point.

    """

    # Log that active parameters are being determined
    logger = logging.getLogger('ACTIVE_PAR')
    logger.info("Determining active parameters.")

    # Check if active parameters analysis has been requested
    if not self._do_active_anal:
        # If not requested, then save all potentially active parameters
        active_par = self._pot_active_par
        active_par_data = [active_par]*self._emulator._n_data[emul_i]

    else:
        # If requested, perform an exhaustive backward stepwise regression
        active_par = SortedSet()
        active_par_data = []
        pot_n_par = len(self._pot_active_par)
        for i in range(self._emulator._n_data[emul_i]):
            # Create ExhaustiveFeatureSelector object
            efs_obj = EFS(LR(), min_features=1, max_features=pot_n_par,
                          print_progress=False, scoring='r2')

            # Fit the data set
            efs_obj.fit(self._emulator._sam_set[emul_i][
                            :, self._pot_active_par],
                        self._emulator._mod_set[emul_i][i])

            # Extract the active parameters for this data set
            active_par_data.append(
                self._pot_active_par[np.sort(efs_obj.best_idx_)])

            # And extract the unique active parameters for this iteration
            active_par.update(active_par_data[i])

            # Log the resulting active parameters
            logger.info("Active parameters for data set %s: %s"
                        % (i, [self._modellink._par_name[par]
                               for par in active_par_data[i]]))

        # Convert active_par to a NumPy array
        active_par = np.array(list(active_par))

    # Save the active parameters
    self._emulator._save_data(emul_i, 'active_par',
                              [active_par, active_par_data])

    # Log that active parameter determination is finished
    logger.info("Finished determining active parameters.")


# Read in the pipeline attributes
# HINT: This method is obsolete and even incompatible with the code
def _retrieve_parameters(self):
    """
    Reads in the pipeline parameters from the provided HDF5-file and saves
    them in the current :obj:`~Pipeline` instance.

    """

    # Log that parameters are being read
    logger = logging.getLogger('INIT')
    logger.info("Retrieving pipeline parameters from provided HDF5-file.")

    # Open hdf5-file
    file = self._open_hdf5('r')

    # Read in all the pipeline attributes
    self._n_sam_init = file.attrs['n_sam_init']
    self._do_active_anal = file.attrs['do_active_anal']
    self._pot_active_par = file.attrs['pot_active_par']
    self._criterion = file.attrs['criterion'].decode('utf-8')

    # Close hdf5-file
    self._close_hdf5(file)

    # Log that reading is finished
    logger.info("Finished retrieving parameters.")


# %% REMOVED IN v0.4.22
# Open hdf5-file
def _open_hdf5(self, mode, filename=None, **kwargs):
    """
    Opens the HDF5-file `filename` according to some set of default
    parameters and returns the opened HDF5-object.

    Parameters
    ----------
    mode : {'r', 'r+', 'w', 'w-'/'x', 'a'}
        String indicating how the HDF5-file needs to be opened.

    Optional
    --------
    filename : str. Default: None
        The name/path of the HDF5-file that needs to be opened in
        `working_dir`. Default is to open the HDF5-file that was provided
        during class initialization.
    **kwargs : dict. Default: {'driver': None, 'libver': 'earliest'}
        Other keyword arguments that need to be given to the
        :func:`~h5py.File` function.

    Returns
    -------
    file : :obj:`~h5py._hl.files.File` object
        Opened HDF5-file object.

    """

    # Log that an HDF5-file is being opened
    logger = logging.getLogger('HDF5-FILE')

    # Set default settings
    hdf5_kwargs = {'driver': None,
                   'libver': 'earliest'}

    # Check filename
    if filename is None:
        filename = self._hdf5_file
    else:
        pass

    # Update hdf5_kwargs with provided ones
    hdf5_kwargs.update(kwargs)

    # Open hdf5-file
    logger.info("Opening HDF5-file '%s' (mode: '%s')." % (filename, mode))
    file = h5py.File(filename, mode, **hdf5_kwargs)

    # Return the opened hdf5-file
    return(file)


# Close hdf5-file
def _close_hdf5(self, file):
    """
    Closes the opened HDF5-file object `file`. This method exists only
    for logging purposes.

    Parameters
    ----------
    file : :obj:`~h5py._hl.files.File` object
        Opened HDF5-file object requiring closing.

    """

    # Log that an HDF5-file will be closed
    logger = logging.getLogger('HDF5-FILE')

    # Close hdf5-file
    file.close()

    # Log about closing the file
    logger.info("Closed HDF5-file.")


# %% REMOVED IN v1.0.0rc5
@property
def worker_mode(self):
    """
    bool: Whether or not all MPI ranks are in worker mode, in which all
    worker ranks are listening for calls from the controller rank. If
    *True*, all workers are continuously listening for calls made with
    :meth:`~_make_call` until set to *False*.
    By default, this is *False*.

    Setting this value to *True* allows for easier use of *PRISM* in
    combination with serial/OpenMP codes (like MCMC methods).

    """

    return(bool(self._worker_mode))


@worker_mode.setter
def worker_mode(self, flag):
    # Make logger
    logger = getCLogger('WORKER_M')

    # Check if provided value is a bool
    flag = check_vals(flag, 'worker_mode', 'bool')

    # If flag and worker_mode are the same, skip
    if flag is self._worker_mode:
        pass
    # If worker mode is turned off, turn it on
    elif flag:
        # Set worker_mode to 1
        self._worker_mode = 1

        # Workers start listening for calls
        self._listen_for_calls()

        # Log that workers are now listening
        logger.info("Workers are now listening for calls.")
    # If worker mode is turned on, turn it off
    else:
        # Make workers stop listening for calls
        self._make_call(None)

        # Log that workers are no longer listening for calls
        logger.info("Workers are no longer listening for calls.")

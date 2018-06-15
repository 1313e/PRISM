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

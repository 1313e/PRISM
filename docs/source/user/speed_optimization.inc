.. _speed_optimization:

Speed optimization
++++++++++++++++++
As with all algorithms, there are several different aspects (mostly numbers) in the *PRISM* code that influence its performance.
Below is an overview of the most important aspects.

----

Number of samples
    The number of samples (:attr:`~prism.Emulator.n_sam`) that makes up an emulator system heavily influences both the construction speed and the evaluation rate of that system, which is (in)directly controlled by :attr:`~prism.Pipeline.n_sam_init` and :attr:`~prism.Pipeline.base_eval_sam`.
    Decreasing their values will have a positive effect on the construction speed and the evaluation rate of an emulator system and thus the emulator iteration it belongs to as a whole, but will naturally decrease its accuracy.
    Additionally, it is worth noting that earlier emulator iterations are evaluated much more than later ones, given that a parameter set is only evaluated in an iteration if it was found plausible in all previous ones.

    .. admonition:: Performance impact

        Increasing the number of samples in an emulator system by a factor of two decreases its evaluation rate by up to a factor of three.


Number of MPI processes
    Trivially, the number of MPI processes that are used for *PRISM* influences its performance.
    Currently, *PRISM* uses a high-level MPI implementation, which means that the evaluation rate of the emulator scales with the highest number of emulator systems (emulated data points) that are assigned to a single MPI process.
    For example, having `16` emulator systems will roughly yield the same evaluation rate on `8` processes and `15` processes (and everything in between).
    Low-level MPI is planned to be implemented in the future, removing this limitation.
	
    .. admonition:: Optimal performance
	
        Make sure that the number of emulator systems (preferably for all iterations) can be divided by or is lower than the number of MPI processes.


Number of OpenMP threads
    This is only important when one uses multiple MPI processes.
    Many of the calculations in *PRISM* require NumPy's :mod:`~numpy.linalg` functions, which use OpenMP.
    On many architectures, these functions (as any other OpenMP operation) will by default spawn as many OpenMP threads as there are cores available.
    However, given that such operations do not know when they are called by an MPI process, every MPI process will attempt to use all cores (e.g., `16` MPI processes will each spawn `16` OpenMP threads, resulting in `16*16=256` OpenMP threads in total).
    This will result in OpenMP operations 'fighting' for computation time, which reduces the overall computation speed dramatically.

    .. admonition:: Optimal performance

        Setting the number of OpenMP threads to `1` (``export OMP_NUM_THREADS=1`` on UNIX or ``set OMP_NUM_THREADS=1`` on Windows) will ensure that this effect does not occur.
        Optionally, if one has more cores available than MPI processes, performance can be improved by setting ``OMP_NUM_THREADS`` to ``int(n_cores/n_MPI_processes)``.

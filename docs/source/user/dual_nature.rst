.. _dual_nature:

Dual nature
+++++++++++
*PRISM* features a high-level MPI implementation, as described in :ref:`mpi`: all user-methods and most major methods are to be executed by all MPI ranks at the same time, and *PRISM* will automatically distribute the work among the available ranks within this function/method.
This allows for *PRISM* to be used with both serial and parallel models, by setting the :attr:`~prism.modellink.ModelLink.MPI_call` flag accordingly, while also allowing for the same code to be used in serial and parallel.
However, given that the emulator of *PRISM* can be very useful for usage in other routines, like :ref:`hybrid_sampling`, an external code will call *PRISM*'s methods.
In order to use *PRISM* in parallel with a parallelized model, this code would have to call *PRISM* with all MPI ranks simultaneously at all times, which may not always be possible.

Therefore, *PRISM* has a `dual execution/call nature`, where it can be switched between two different modes.
In the default mode, *PRISM* works as described before, where all MPI ranks call the same user-code.
However, by setting the :attr:`~prism.Pipeline.worker_mode` to *True* (on all MPI ranks), 


PEP 377 

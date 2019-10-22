.. _dual_nature:

Dual nature (normal/worker mode)
++++++++++++++++++++++++++++++++
*PRISM* features a high-level MPI implementation, as described in :ref:`mpi`: all user-methods and most major methods are to be executed by all MPI ranks at the same time, and *PRISM* will automatically distribute the work among the available ranks within this function/method.
This allows for *PRISM* to be used with both serial and parallel models, by setting the :attr:`~prism.modellink.ModelLink.MPI_call` flag accordingly, while also allowing for the same code to be used in serial and parallel.
However, given that the emulator of *PRISM* can be very useful for usage in other routines, like :ref:`hybrid_sampling`, an external code will call *PRISM*'s methods.
In order to use *PRISM* in parallel with a parallelized model, this code would have to call *PRISM* with all MPI ranks simultaneously at all times, which may not always be possible (e.g., when using MCMC methods).

Therefore, *PRISM* has a `dual execution/call nature`, where it can be switched between two different modes.
In the default mode, *PRISM* works as described before, where all MPI ranks call the same user-code.
However, by using the :class:`~prism._pipeline.WorkerMode` context manager, accessed through :meth:`~prism.Pipeline.worker_mode`, all code within will be executed in worker mode.
When in worker mode, all worker ranks are continously listening for calls from the controller rank, made with the :meth:`~prism.Pipeline._make_call` and :meth:`~prism.Pipeline._make_call_workers` methods.
They will continue to do so until the controller exits :class:`~prism._pipeline.WorkerMode` with :meth:`~prism._pipeline.WorkerMode.__exit__`.
Manually exiting should solely be done in advanced use-cases.

In :attr:`~prism.Pipeline.worker_mode`, one uses the following structure (assuming that the :obj:`~prism.Pipeline` instance is called :pycode:`pipe`)::

    # Code to be executed in default mode

    with pipe.worker_mode:
        if pipe.is_controller:
            # Code to be executed in worker mode

    # More code to be executed in default mode

.. note::
    All code that is inside the :attr:`~prism.Pipeline.worker_mode` context manager should solely be executed by the controller rank.
    If not, all worker ranks will execute this code after the controller ranks exits the context manager.
    Currently, it is not possible to make a context manager handle this automatically (the rejected :PEP:`377` describes this perfectly).

The :meth:`~prism.Pipeline._make_call` method accepts almost anything that can be called.
It can also be used when not in :attr:`~prism.Pipeline.worker_mode`, in which case it works the exact same way for all MPI ranks.
Its sole limitation is that all supplied arguments must be pickleable (e.g., compiled code objects are NOT pickleable due to safety reasons), both when used in :attr:`~prism.Pipeline.worker_mode` and outside of it.
The :mod:`copyreg` module can be used to register specific objects to become pickleable (including compiled code objects).

The :attr:`~prism.Pipeline.worker_mode` can be used in a variety of ways, as described below.
It can be used to access any attribute of the :obj:`~prism.Pipeline` instance::

    with pipe.worker_mode:
        if pipe.is_controller:
            # Construct first emulator iteration
            pipe._make_call('construct', 1)

            # Print latest constructed emulator iteration
            print(pipe._make_call('emulator._get_emul_i', 1, 0))

            # Make a specific projection with the 'row' alignment
            pipe._make_call('project', 1, (0, 1), align='row')

which is equivalent to::

    # Construct first emulator iteration
    pipe.construct(1)

    # Print latest constructed emulator iteration
    print(pipe.emulator._get_emul_i(1, 0))

    # Make a specific projection with the 'row' alignment
    pipe.project(1, (0, 1), align='row')

The above two code snippets are equal to each other, and the :attr:`~prism.Pipeline.worker_mode` will most likely be used very rarely in this fashion.
However, by supplying the :meth:`~prism.Pipeline._make_call` method with a callable function (that can be pickled), externally defined functions can be executed::

    # Enable worker mode
    with pipe.worker_mode:
        if pipe.is_controller:
            # Import print function that prepends MPI rank to message
            from prism._internal import rprint

            # Make call to use this function
            # Equivalent to 'rprint("Reporting in.")'
            pipe._make_call(rprint, "Reporting in.")

This is especially useful when one combines a serial code with *PRISM*, but wants *PRISM* to execute in MPI.
An application example of this is :ref:`hybrid_sampling`.

.. versionchanged:: 1.2.0
    It is also possible to make a call that is solely executed by the workers, by using the :meth:`~prism.Pipeline._make_call_workers` method.

.. versionchanged:: 1.2.0
    If any positional or keyword argument is a string written as :pycode:`'pipe.XXX'`, it is assumed that :pycode:`'XXX'` refers to a :class:`~prism.Pipeline` attribute of the MPI rank receiving the call.
    It will be replaced with the corresponding attribute before `exec_fn` is called.

.. versionchanged:: 1.2.0
    Initializing a worker mode within an already existing worker mode is possible and will function properly.
    An example of this is using the :meth:`~prism.Pipeline.construct` or :meth:`~prism.Pipeline.crystal` method within worker mode, as both use one themselves as well.


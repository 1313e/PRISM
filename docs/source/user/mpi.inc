.. _mpi:

MPI implementation
++++++++++++++++++
Given that most scientific models are either already parallelized or could benefit from parallelization, we had to make sure that *PRISM* allows for both MPI and OpenMP coded models to be connected.
Additionally, since individual emulator systems in an emulator iteration are independent of each other, the extra CPUs required for the model should also be usable by the emulator.
For that reason, *PRISM* features a high-level MPI implementation for using MPI-coded models, while the Python package *NumPy* handles the OpenMP side.
A mixture of both is also possible (due to the :attr:`~prism.Pipeline.worker_mode` context manager).

Here, we discuss the MPI scaling tests that were performed on *PRISM*.
For the tests, the same :class:`~prism.modellink.GaussianLink` class was used as in :ref:`minimal_example`, but this time with :math:`32` emulator systems (comparison data points) instead of :math:`3`.
In *PRISM*, all emulator systems are spread out over the available number of MPI processes as much as possible while also trying to balance the number of calculations performed per MPI process.
Since all emulator systems are stored in different HDF5-files, it is possible to reinitialize the :class:`~prism.Pipeline` using the same :class:`~prism.emulator.Emulator` class and :class:`~prism.modellink.ModelLink` subclass on a different number of MPI processes.
To make sure that the results are not influenced by the variation in evaluation rates, we constructed an emulator of the Gaussian model and used the exact same emulator in every test.

The tests were carried out using any number of MPI processes between :math:`1` and :math:`32`, and using a single OpenMP thread each time for consistency.
We generated a Latin-Hypercube design of :math:`3\cdot10^6` samples and measured the average evaluation rate of the emulator using the same Latin-Hypercube design each time.
To take into account any variations in the evaluation rate caused by initializations, this test was performed :math:`20` times.
As a result, this Latin-Hypercube design was evaluated in the emulator a total of :math:`640` times, giving an absolute total of :math:`1.92\cdot10^9` emulator evaluations.

.. figure:: images/MPI_np32nt20ns3000000.png
    :alt: MPI scaling test results.
    :width: 100%
    :align: center
    :name: MPI_scaling

    Figure showing the MPI scaling of *PRISM* using the emulator of a simple Gaussian model with :math:`32` emulator systems.
    The tests involved analyzing a Latin-Hypercube design of :math:`3\cdot10^6` samples in the emulator, determining the average evaluation rate and executing this a total of :math:`20` times using the same sample set every time.
    The emulator used for this was identical in every instance.
    **Left axis:** The average evaluation rate of the emulator vs. the number of MPI processes it is running on.
    **Right axis:** The relative speed-up factor vs. the number of MPI processes, which is defined as :math:`\frac{f(x)}{f(1)\cdot x}` with :math:`f(x)` the average evaluation rate and :math:`x` the number of MPI processes.
    **Dotted line:** The minimum acceptable relative speed-up factor, which is always :math:`1/x`.
    **Dashed line:** A straight line with a slope of :math:`{\sim}0.645`, connecting the lowest and highest evaluation rates.
    The tests were performed using the *OzSTAR computing facility* at the Swinburne University of Technology, Melbourne, Australia.

In :numref:`MPI_scaling`, we show the results of the performed MPI scaling tests.
On the left y-axis, the average evaluation rate vs. the number of MPI processes the test ran on is plotted, while the relative speed-up factor vs. the number of MPI processes is plotted on the right y-axis.
The relative speed-up factor is defined as :math:`f(x)/(f(1)\cdot x)` with :math:`f(x)` the average evaluation rate and :math:`x` the number of MPI processes.
The ideal MPI scaling would correspond to a relative speed-up factor of unity for all :math:`x`.

In this figure, we can see the effect of the high-level MPI implementation.
Because the emulator systems are spread out over the available MPI processes, the evaluation rate is mostly determined by the runtime of the MPI process with the highest number of systems assigned.
Therefore, if the number of emulator systems (:math:`32` in this case) cannot be divided by the number of available MPI processes, the speed gain is reduced, leading to the plateaus like the one between :math:`x=16` and :math:`x=31`.
Due to the emulator systems not being the same, their individual evaluation rates are different such that a different evaluation rate has a bigger effect on the average evaluation rate of the emulator the more MPI processes there are.
This is shown by the straight dashed line drawn between :math:`f(1)` and :math:`f(32)`, which has a slope of :math:`{\sim}0.645`.

The relative speed-up factor shows the efficiency of every individual MPI process in a specific run, compared to using a single MPI process.
This also shows the effect of the high-level MPI implementation, giving peaks when the maximum number of emulator systems per MPI process has decreased.
The dotted line shows the minimum acceptable relative speed-up factor, which is always defined as :math:`1/x`.
On this line, the average evaluation rate :math:`f(x)` for any given number of MPI processes is always equal to :math:`f(1)`.


.. _data_idx:

Data identifiers (`data_idx`)
+++++++++++++++++++++++++++++
The comparison data points that are given to the :class:`~prism.modellink.ModelLink` class each require a unique data point identifier, allowing *PRISM* to distinguish between them.
This data identifier (called `data_idx` in the code) can be used by the model wrapped in the :meth:`~prism.modellink.ModelLink.call_model` method as a description of how to calculate/extract the data point.
It can be provided as a non-mutable sequence of a combination of integers, floats and strings, each element describing a part of the operations required.
The data identifier sequence can be of any length, and the length can differ between data points.

In its simplest form, the data identifier is a single value that is given to a function :math:`f(x)`, which is a function that is defined for a given model parameter set and returns the function value belonging to the input :math:`x`.
This is the way the data identifier works for the two standard :class:`~prism.modellink.ModelLink` subclasses, the :class:`~prism.modellink.SineWaveLink` and :class:`~prism.modellink.GaussianLink` classes.

For more sophisticated models, a single value/element is not enough to uniquely identify a data point.
For example, a model might generate several different named data sets that combined describe the model realization.
In this case, a data identifier can be made up of a string describing which data set the required data point is in and two integers acting as the indices in that data set (e.g., ``('property1', 346, 41)``).

An even more complex example is when a data point needs to be retrieved from a specific named data set at a certain point in a model simulation, after which an operation needs to be carried out (like, making a histogram of the results) and the resulting data point is then found at a specific value in that histogram.
The histogram here might only be necessary to make for specific data sets, while different operations are required for others.
*PRISM* allows for such complex data identifiers to be given, as it treats every sequence of data identifier elements as separated.
Two different data identifiers working as described above can for example be written as ``[(14, 'property1', 'histogram', 7.5), (17, 'property2', 'average')]``, where the first data point requires an extra (float) value for the histogram and the second does not.
In order to do this, one would of course be required to make sure that the :meth:`~prism.modellink.ModelLink.call_model` method can perform these operations when provided with the proper data identifier.

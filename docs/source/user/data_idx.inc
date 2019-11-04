.. _data_idx:

Data identifiers (`data_idx`)
+++++++++++++++++++++++++++++
The comparison data points that are given to the :class:`~prism.modellink.ModelLink` class each require a unique data point identifier, allowing *PRISM* to distinguish between them.
This data identifier (called :attr:`~prism.modellink.ModelLink.data_idx`) can be used by the model wrapped in the :meth:`~prism.modellink.ModelLink.call_model` method as a description of how to calculate/extract the data point.
It can be provided as a non-mutable sequence (a Python tuple) of a combination of booleans; integers; floats; and strings, each element describing a part of the operations required.
The data identifier sequence can be of any length, and the length can differ between data points.

.. note::
    If a data identifier is given as a single element, then the identifier is saved as that single element instead of a tuple.
    For example, :pycode:`data_idx = [(1), (2), (3, 4), ...]` would be saved as :pycode:`data_idx = [1, 2, (3, 4), ...]`.


In its simplest form, the data identifier is a single value that is given to a function :math:`f(x)`, which is a function that is defined for a given model parameter set and returns the function value belonging to the input :math:`x`.
This is the way the data identifier works for the three standard :class:`~prism.modellink.ModelLink` subclasses; :class:`~prism.modellink.SineWaveLink`; :class:`~prism.modellink.GaussianLink`; and :class:`~prism.modellink.PolyLink`.
It is also used in the :class:`~LineLink` class described in the `line_link.py`_ file above.

For more sophisticated models, a single value/element is not enough to uniquely identify a data point.
A simple example of this would be if the model generates a two-dimensional array of values, where one specific value needs to be returned.
Then, the data identifier can be given as a tuple of two integers, like :pycode:`data_idx = [(1, 1), (4, 8), ...]`.
In the case that the model also generates several two-dimensional arrays which are named, an extra string could be used to identify this array first: :pycode:`data_idx = [('array1', 1, 1), ('array4', 4, 8), ...]`.

An even more complex example is when a data point needs to be retrieved from a specific named data set at a certain point in a model simulation, after which an operation needs to be carried out (like, making a histogram of the results) and the resulting data point is then found at a specific value in that histogram.
The histogram here might only be necessary to make for specific data sets, while different operations are required for others.
*PRISM* allows for such complex data identifiers to be given, as it treats every sequence of data identifier elements as separated.
Two different data identifiers working as described above can for example be written as :pycode:`data_idx = [(14, 'array1', 'histogram', 7.5), (17, 'array7', 'average'), ...]`, where the first data point requires an extra (float) value for the histogram and the second does not.
In order to do this, one would of course be required to make sure that the :meth:`~prism.modellink.ModelLink.call_model` method can perform these operations when provided with the proper data identifier.


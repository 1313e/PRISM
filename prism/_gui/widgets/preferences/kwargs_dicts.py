# -*- coding: utf-8 -*-

"""
GUI Kwargs Dicts Options
========================
Provides a custom :class:`~PyQt5.QtWidgets.QDialog` subclass that allows for
the projection keyword argument dicts to be modified properly in the Projection
GUI preferences.

"""


# %% IMPORTS
# Built-in imports
from itertools import chain

# Package imports
from e13tools.utils import docstring_append, docstring_substitute
from matplotlib import cm
from matplotlib import rcParamsDefault as rcParams
from matplotlib.lines import lineMarkers, lineStyles
import numpy as np
from PyQt5 import QtCore as QC, QtGui as QG, QtWidgets as QW
from sortedcontainers import SortedDict as sdict, SortedSet as sset

# PRISM imports
from prism._docstrings import create_type_doc, kwargs_doc, qt_slot_doc
from prism._gui.widgets import (
    BaseBox, QW_QComboBox, QW_QDoubleSpinBox, QW_QEditableComboBox, QW_QLabel,
    QW_QSpinBox, get_box_value, set_box_value)
from prism._gui.widgets.preferences.custom_boxes import (
    ColorBox, DefaultBox, FigSizeBox)

# All declaration
__all__ = ['KwargsDictBoxLayout', 'KwargsDictDialog', 'KwargsDictDialogPage']


# %% CLASS DEFINITIONS
# Make a subclass that allows for kwargs dicts to be saved and edited
class KwargsDictBoxLayout(QW.QHBoxLayout):
    """
    Defines the :class:`~KwargsDictBoxLayout` class for the preferences window.

    This class provides the options entry box that gives the user access to a
    separate window, where the various different keyword dicts can be modified.

    """

    # This function creates an editable list of input boxes for the kwargs
    @docstring_substitute(optional=kwargs_doc.format(
        'PyQt5.QtWidgets.QHBoxLayout'))
    def __init__(self, options_dialog_obj, *args, **kwargs):
        """
        Initialize an instance of the :class:`~KwargsDictBoxLayout` class.

        Parameters
        ----------
        options_dialog_obj : \
            :obj:`~prism._gui.widgets.preferences.OptionsDialog` object
            Instance of the
            :class:`~prism._gui.widgets.preferences.OptionsDialog` class that
            acts as the parent of the :class:`~KwargsDictDialog` this layout
            creates.

        %(optional)s

        """

        # Save provided options_dialog_obj
        self.options = options_dialog_obj

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Create the kwargs dict window
        self.init()

    # This function creates the kwargs dict window
    def init(self):
        """
        Sets up the box layout after it has been initialized.

        This function is mainly responsible for initializing the
        :class:`~KwargsDictDialog` class and binding it.

        """

        # Initialize the window for the kwargs dict
        self.dict_dialog = KwargsDictDialog(self.options)

        # Add a view button
        view_but = QW.QPushButton('View')
        view_but.setToolTip("View/edit the projection keyword dicts")
        view_but.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
        view_but.clicked.connect(self.dict_dialog)
        self.view_but = view_but
        self.addWidget(view_but)

    # This function calls the add_page()-method of dict_dialog
    def add_dict(self, *args, **kwargs):
        """
        Adds a new kwargs dict to the box layout, by calling the
        :meth:`~KwargsDictDialog.add_page` method using the provided `args` and
        `kwargs`.

        """

        self.dict_dialog.add_page(*args, **kwargs)


# Make a subclass that shows the kwargs dict entries window
class KwargsDictDialog(QW.QDialog):
    """
    Defines the :class:`~KwargsDictDialog` class for the preferences window.

    This class provides the 'Projection keyword argument dicts' dialog, which
    allows for the various different kwargs dicts to be modified by the user.

    """

    @docstring_substitute(optional=kwargs_doc.format(
        'PyQt5.QtWidgets.QDialog'))
    def __init__(self, options_dialog_obj, *args, **kwargs):
        """
        Initialize an instance of the :class:`~KwargsDictDialog` class.

        Parameters
        ----------
        options_dialog_obj : \
            :obj:`~prism._gui.widgets.preferences.OptionsDialog` object
            Instance of the
            :class:`~prism._gui.widgets.preferences.OptionsDialog` class that
            acts as the parent of this dialog.

        %(optional)s

        """

        # Save provided options_dialog_obj
        self.options = options_dialog_obj
        self.options.dict_dialog = self

        # Call super constructor
        super().__init__(self.options, *args, **kwargs)

        # Create the kwargs dict window
        self.init()

    # This function shows an editable window with the entries in the dict
    @QC.pyqtSlot()
    def __call__(self):
        """
        Qt slot that shows the kwargs dict dialog in the center of the
        preferences window.

        """

        # Show it
        self.show()

        # Move the kwargs_dicts window to the center of the main window
        self.move(self.options.geometry().center()-self.rect().center())

    # This function creates the kwargs dict window
    def init(self):
        """
        Sets up the kwargs dict dialog after it has been initialized.

        This function is mainly responsible for setting up the layout of the
        dialog, and making sure that new kwargs dict pages can be added.

        """

        # Create a window layout
        window_layout = QW.QVBoxLayout(self)

        # Create a splitter widget for this window
        splitter_widget = QW.QSplitter()
        splitter_widget.setChildrenCollapsible(False)
        window_layout.addWidget(splitter_widget)

        # Create a contents widget
        self.contents = QW.QListWidget()
        self.contents.setMovement(QW.QListView.Static)
        self.contents.setSpacing(1)
        splitter_widget.addWidget(self.contents)

        # Create pages widget
        self.pages = QW.QStackedWidget()
        splitter_widget.addWidget(self.pages)

        # Set signal handling
        self.contents.currentRowChanged.connect(self.pages.setCurrentIndex)

        # Add a close button
        button_box = QW.QDialogButtonBox()
        window_layout.addWidget(button_box)
        close_but = button_box.addButton(button_box.Close)
        close_but.clicked.connect(self.close)
        self.close_but = close_but

        # Set some properties for this window
        self.setWindowTitle("Viewing projection keyword dicts")     # Title

        # Set the height of an editable entry
        self.entry_height = 24

    # This function creates a new page
    @docstring_substitute(optional=kwargs_doc.format(
        'prism._gui.widgets.preferences.KwargsDictDialogPage'))
    def add_page(self, name, option_key, *args, **kwargs):
        """
        Initializes a new :obj:`~KwargsDictDialogPage` object with name `name`
        and adds it to this dialog.

        Parameters
        ----------
        name : str
            The name that this kwargs dict page will have.
        option_key : str
            The name of the options entry that this page will create.
            The value of `option_key` must correspond to the name the
            associated dict has in the :meth:`~prism.Pipeline.project` method.

        %(optional)s

        """

        # Create a page
        kwargs_page = KwargsDictDialogPage(self, name, *args, **kwargs)

        # Add this new page to the option_entries
        self.options.create_entry(option_key, kwargs_page,
                                  self.options.proj_defaults[option_key])

        # Create a scrollarea for the page
        scrollarea = QW.QScrollArea(self)
        scrollarea.setWidgetResizable(True)
        scrollarea.setWidget(kwargs_page)

        # Add it to the contents and pages widgets
        self.contents.addItem(name)
        self.contents.setFixedWidth(1.1*self.contents.sizeHintForColumn(0))
        self.pages.addWidget(scrollarea)


# Make a class for describing a kwargs dict page
class KwargsDictDialogPage(BaseBox):
    """
    Defines the :class:`~KwargsDictDialogPage` class for the kwargs dict
    dialog.

    This class provides the tab/page in the kwargs dict dialog where the items
    of the associated kwargs dict can be viewed and modified by the user.

    """

    @docstring_substitute(optional=kwargs_doc.format(
        'prism._gui.widgets.core.BaseBox'))
    def __init__(self, kwargs_dict_dialog_obj, name, std_entries,
                 banned_entries, *args, **kwargs):
        """
        Initialize an instance of the :class:`~KwargsDictDialogPage` class.

        Parameters
        ----------
        kwargs_dict_dialog_obj : :obj:`~KwargsDictDialog` object
            Instance of the :class:`~KwargsDictDialog` class that initialized
            this kwargs dict page.
        name : str
            The name of this kwargs dict page.
        std_entries : list of str
            A list of all standard entry types that this kwargs dict should
            accept.
        banned_entries : list of str
            A list of all entry types that this kwargs dict should not accept.
            Usually, these entry types are used by *PRISM* and therefore should
            not be modified by the user.

        %(optional)s

        """

        # Save provided kwargs_dict_dialog_obj
        self.entry_height = kwargs_dict_dialog_obj.entry_height
        self.name = name
        self.std_entries = sset(std_entries)
        self.banned_entries = sset(banned_entries)

        # Call super constructor
        super().__init__(*args, **kwargs)

        # Create the kwargs dict window
        self.init()

    # This function creates the kwargs dict page
    def init(self):
        """
        Sets up the kwargs dict page after it has been initialized.

        This function is mainly responsibe for creating the layout of the page;
        determining what entry types are available; and preparing for the user
        to add entries.

        """

        # Create page layout
        page_layout = QW.QVBoxLayout(self)

        # Create a grid for this layout
        self.kwargs_grid = QW.QGridLayout()
        self.kwargs_grid.setColumnStretch(1, 1)
        self.kwargs_grid.setColumnStretch(2, 2)
        page_layout.addLayout(self.kwargs_grid)

        # Add a header
        self.kwargs_grid.addWidget(QW_QLabel(""), 0, 0)
        self.kwargs_grid.addWidget(QW_QLabel("Entry type"), 0, 1)
        self.kwargs_grid.addWidget(QW_QLabel("Field value"), 0, 2)

        # Make sure that '' is not in std_entries or banned_entries
        self.std_entries.discard('')
        self.banned_entries.discard('')

        # Create list of available entry types
        self.avail_entries = sset([attr[12:] for attr in dir(self)
                                   if attr.startswith('create_type_')])

        # Convert std_entries to solely contain valid available entry types
        self.std_entries.intersection_update(self.avail_entries)
        self.std_entries.difference_update(self.banned_entries)

        # Add an 'add' button at the bottom of this layout
        add_but = QW.QToolButton()
        add_but.setFixedSize(self.entry_height, self.entry_height)
        add_but.setToolTip("Add a new entry")
        add_but.clicked.connect(self.add_editable_entry)
        self.add_but = add_but

        # If this theme has an 'add' icon, use it
        if QG.QIcon.hasThemeIcon('add'):    # pragma: no cover
            add_but.setIcon(QG.QIcon.fromTheme('add'))
        # Else, use a simple plus
        else:
            add_but.setText('+')

        # Add button to layout
        page_layout.addWidget(add_but)
        page_layout.addStretch()

        # Set a minimum width for the first grid column
        self.kwargs_grid.setColumnMinimumWidth(0, self.entry_height)

    # This function gets the dict value of a page
    def get_box_value(self):
        """
        Returns the current value of the kwargs dict page.

        Returns
        -------
        page_dict : dict
            A dict containing all valid entries that are currently on this
            kwargs dict page. Any invalid entries (banned or empty ones) are
            ignored.

        """

        # Create an empty dict to hold the values in
        page_dict = sdict()

        # Loop over all items in grid and save them to page_dict
        for row in range(1, self.kwargs_grid.rowCount()):
            # Obtain item that should contain the entry_type in this row
            entry_type_item = self.kwargs_grid.itemAtPosition(row, 1)

            # If entry_type_item is None, this row contains no items
            if entry_type_item is None:
                continue

            # Obtain the entry_type
            entry_type = get_box_value(entry_type_item.widget())

            # If the entry_type is empty or banned, skip this row
            if(entry_type == '' or entry_type in self.banned_entries):
                continue

            # Obtain the value of the corresponding field box
            field_value = get_box_value(
                self.kwargs_grid.itemAtPosition(row, 2).widget())

            # Add this to the dict
            page_dict[entry_type] = field_value

        # Return page_dict
        return(page_dict)

    # This function sets the dict value of a page
    def set_box_value(self, page_dict):
        """
        Sets the current value of the kwargs dict page to `page_dict`.

        Parameters
        ----------
        page_dict : dict
            A dict containing all entries that this kwargs dict page must have.
            Current entries that are also in `page_dict` will be reused,
            otherwise they are deleted.

        """

        # Make empty dict containing all current valid entries
        cur_entry_dict = sdict()

        # Remove all entries from kwargs_grid
        for row in range(1, self.kwargs_grid.rowCount()):
            # Obtain item that should contain the entry_type in this row
            entry_type_item = self.kwargs_grid.itemAtPosition(row, 1)

            # If entry_type_item is None, this row contains no items
            if entry_type_item is None:
                continue

            # Obtain the entry_type
            entry_type = get_box_value(entry_type_item.widget())

            # Delete this entry if not in page_dict or if it is not allowed
            if(entry_type not in page_dict or (entry_type == '') or
               entry_type in self.banned_entries):
                self.remove_editable_entry(entry_type_item.widget())
                continue

            # If this entry appears multiple times, delete its previous entry
            if entry_type in cur_entry_dict:
                for item in cur_entry_dict[entry_type]:
                    item.widget().close()
                    del item
                cur_entry_dict.pop(entry_type)

            # Add this entry to cur_entry_dict
            cur_entry_dict[entry_type] =\
                [self.kwargs_grid.takeAt(3) for _ in range(3)]

        # Loop over all items in page_dict and act accordingly
        for row, (entry_type, field_value) in enumerate(page_dict.items(), 1):
            # Check if this entry_type already existed
            if entry_type in cur_entry_dict:
                # If so, put it back into kwargs_grid
                for col, item in enumerate(cur_entry_dict[entry_type]):
                    self.kwargs_grid.addItem(item, row, col)
            else:
                # If not, add it to kwargs_grid
                self.add_editable_entry()

                # Set this new entry to the proper type
                set_box_value(self.kwargs_grid.itemAtPosition(row, 1).widget(),
                              entry_type)

            # Set the value of the corresponding field
            set_box_value(self.kwargs_grid.itemAtPosition(row, 2).widget(),
                          field_value)

    # This function creates an editable entry
    @QC.pyqtSlot()
    @docstring_substitute(qt_slot=qt_slot_doc)
    def add_editable_entry(self):
        """
        Adds a new editable entry to the kwargs dict page, which allows for the
        user to edit the contents of the kwargs dict.

        %(qt_slot)s

        """

        # Create a combobox with different standard kwargs
        kwargs_box = QW_QEditableComboBox()
        kwargs_box.setFixedHeight(self.entry_height)
        kwargs_box.addItem('')
        kwargs_box.addItems(self.std_entries)
        kwargs_box.setToolTip("Select a standard type for this entry or add "
                              "it manually")
        kwargs_box.currentTextChanged.connect(
            lambda x: self.entry_type_selected(x, kwargs_box))

        # Create a delete button
        delete_but = QW.QToolButton()
        delete_but.setFixedSize(self.entry_height, self.entry_height)
        delete_but.setToolTip("Delete this entry")
        delete_but.clicked.connect(
            lambda: self.remove_editable_entry(kwargs_box))

        # If this theme has a 'remove' icon, use it
        if QG.QIcon.hasThemeIcon('remove'):     # pragma: no cover
            delete_but.setIcon(QG.QIcon.fromTheme('remove'))
        # Else, use a simple cross
        else:
            delete_but.setText('X')

        # Determine the number of entries currently in kwargs_grid
        n_rows = self.kwargs_grid.count()//3

        # Make a new editable entry
        self.kwargs_grid.addWidget(delete_but, n_rows, 0)
        self.kwargs_grid.addWidget(kwargs_box, n_rows, 1)
        self.kwargs_grid.addWidget(QW.QLabel(''), n_rows, 2)

    # This function deletes an editable entry
    @QC.pyqtSlot(QW.QComboBox)
    @docstring_substitute(qt_slot=qt_slot_doc)
    def remove_editable_entry(self, kwargs_box):
        """
        Removes the editable entry associated with the provided `kwargs_box`.

        %(qt_slot)s

        Parameters
        ----------
        kwargs_box : \
            :obj:`~prism._gui.widgets.base_widgets.QW_QEditableComboBox` object
            The combobox that is used for setting the entry type of this entry.

        """

        # Determine at what index the provided kwargs_box currently is
        index = self.kwargs_grid.indexOf(kwargs_box)

        # As every row contains 3 items, remove item 3 times at this index-1
        for _ in range(3):
            # Take the current layoutitem at this index-1
            item = self.kwargs_grid.takeAt(index-1)

            # Close the widget in this item and delete the item
            item.widget().close()
            del item

    # This function is called when an item in the combobox is selected
    # TODO: Make sure that two fields cannot have the same name
    @QC.pyqtSlot(str, QW.QComboBox)
    def entry_type_selected(self, entry_type, kwargs_box):
        """
        Qt slot that modifies the field box associated with the provided
        `kwargs_box` to given `entry_type`.

        Parameters
        ----------
        entry_type : str
            The entry type that is requested for the field box.
        kwargs_box : \
            :obj:`~prism._gui.widgets.base_widgets.QW_QEditableComboBox` object
            The combobox that is used for setting the entry type of this entry.

        """

        # Determine at what index the provided kwargs_box currently is
        index = self.kwargs_grid.indexOf(kwargs_box)

        # Retrieve what the current field_box is
        cur_box = self.kwargs_grid.itemAt(index+1).widget()

        # Check what entry_type is given and act accordingly
        if(entry_type == ''):
            # If '' is selected, use an empty label widget
            field_box = QW.QLabel('')
        elif entry_type in self.banned_entries:
            # If one of the banned types is selected, show a warning message
            warn_msg = (r"<b><i>%s</i></b> is a reserved or banned entry type!"
                        % (entry_type))
            field_box = QW.QLabel(warn_msg)
        elif entry_type in self.std_entries:
            # If one of the standard types is selected, create its box
            field_box = getattr(self, 'create_type_%s' % (entry_type))()
        else:
            # If an unknown type is given, add default box if not used already
            if isinstance(cur_box, DefaultBox):
                return
            else:
                field_box = DefaultBox()

        # Set the height of this box
        field_box.setFixedHeight(self.entry_height)

        # Replace current field_box with new field_box
        cur_item = self.kwargs_grid.replaceWidget(cur_box, field_box)
        cur_item.widget().close()
        del cur_item

    # This function creates an alpha box
    @docstring_append(create_type_doc.format('alpha'))
    def create_type_alpha(self):
        # Make double spinbox for alpha
        alpha_box = QW_QDoubleSpinBox()
        alpha_box.setRange(0, 1)
        set_box_value(alpha_box, 1)
        alpha_box.setToolTip("Alpha value to use for the plotted data")
        return(alpha_box)

    # This function creates a cmap box
    @docstring_append(create_type_doc.format('cmap'))
    def create_type_cmap(self):
        # Obtain a list with default colormaps that should be at the top
        std_cmaps = sset(['cividis', 'dusk', 'freeze', 'heat', 'inferno',
                          'magma', 'plasma', 'rainforest', 'viridis'])
        std_cmaps_r = sset([cmap+'_r' for cmap in std_cmaps])

        # Obtain a list with all colormaps and their reverses
        all_cmaps = sset([cmap for cmap in cm.cmap_d
                          if not cmap.endswith('_r')])
        all_cmaps_r = sset([cmap for cmap in cm.cmap_d if cmap.endswith('_r')])

        # Gather all sets together
        cmaps = (std_cmaps, std_cmaps_r, all_cmaps, all_cmaps_r)

        # Determine the cumulative lengths of all four sets
        cum_len = np.cumsum(list(map(len, cmaps)))

        # Set the size for the colormap previews
        cmap_size = (100, 15)

        # If the colormap icons have not been created yet, do that now
        if not hasattr(self, 'cmap_icons'):
            cmap_icons = sdict()
            for cmap in chain(all_cmaps, all_cmaps_r):
                cmap_icons[cmap] = self.create_cmap_icon(cmap, cmap_size)
            self.cmap_icons = cmap_icons

        # Create a combobox for cmaps
        cmaps_box = QW_QComboBox()
        for cmap in chain(*cmaps):
            cmap_icon = self.cmap_icons[cmap]
            cmaps_box.addItem(cmap_icon, cmap)

        # Add some separators
        for i in reversed(cum_len[:-1]):
            cmaps_box.insertSeparator(i)
        cmaps_box.insertSeparator(cum_len[1]+1)

        # Set remaining properties
        set_box_value(cmaps_box, rcParams['image.cmap'])
        cmaps_box.setIconSize(QC.QSize(*cmap_size))
        cmaps_box.setToolTip("Colormap to be used for the corresponding plot "
                             "type")
        cmaps_box.currentTextChanged.connect(self.cmap_selected)
        return(cmaps_box)

    # This function creates an icon of a colormap
    @staticmethod
    def create_cmap_icon(cmap, size):
        """
        Creates a :obj:`~PyQt5.QtGui.QIcon` object of the given `cmap` with the
        provided `size`.

        Parameters
        ----------
        cmap : :obj:`~matplotlib.colors.Colormap` object or str
            The colormap for which an icon needs to be created.
        size : tuple
            A tuple containing the width and height dimension values of the
            icon to be created.

        Returns
        -------
        icon : :obj:`~PyQt5.QtGui.QIcon` object
            The instance of the :class:`~PyQt5.QtGui.QIcon` class that was
            created from the provided `cmap` and `size`.

        """

        # Obtain the cmap
        cmap = cm.get_cmap(cmap)

        # Obtain the RGBA values of the colormap
        x = np.linspace(0, 1, 256)
        rgba = cmap(x)

        # Convert to Qt RGBA values
        rgba = [QG.QColor(
            int(r*255),
            int(g*255),
            int(b*255),
            int(a*255)).rgba() for r, g, b, a in rgba]

        # Create an image object
        image = QG.QImage(256, 1, QG.QImage.Format_Indexed8)

        # Set the value of every pixel in this image
        image.setColorTable(rgba)
        for i in range(256):
            image.setPixel(i, 0, i)

        # Scale the image to its proper size
        image = image.scaled(*size)

        # Convert the image to a pixmap
        pixmap = QG.QPixmap.fromImage(image)

        # Convert the pixmap to an icon
        icon = QG.QIcon(pixmap)

        # Return the icon
        return(icon)

    # This function checks a selected cmap
    @QC.pyqtSlot(str)
    def cmap_selected(self, cmap):
        """
        Qt slot that checks a provided `cmap` and shows an error message if
        `cmap` is a terrible colormap.

        """

        # Make a tuple with terrible colormaps
        bad_cmaps = ('gist_ncar', 'gist_rainbow', 'gist_stern', 'jet',
                     'nipy_spectral')

        # If a terrible colormap is selected, show error message
        if cmap.startswith(bad_cmaps):
            # Create error message
            err_msg = ("The selected <b><i>%s</i></b> cmap is terrible for "
                       "drawing PRISM's projection figures. To avoid "
                       "introducing fake perceptual features, it is "
                       "recommended to pick a <i>perceptually uniform "
                       "sequential</i> colormap, like the ones at the top of "
                       "this list.<br><br>"
                       "See <a href=\"%s\">here</a> for more information on "
                       "this subject."
                       % (cmap, ("https://e13tools.readthedocs.io/en/latest/"
                                 "user/colormaps.html#background")))

            # Show error window
            QW.QMessageBox.warning(
                self, "%s WARNING" % (cmap.upper()), err_msg)

    # This function creates a color picker box
    @docstring_append(create_type_doc.format('color'))
    def create_type_color(self):
        return(ColorBox())

    # This function creates a dpi box
    @docstring_append(create_type_doc.format('dpi'))
    def create_type_dpi(self):
        # Make spinbox for dpi
        dpi_box = QW_QSpinBox()
        dpi_box.setRange(1, 9999999)
        set_box_value(dpi_box, rcParams['figure.dpi'])
        dpi_box.setToolTip("DPI (dots per inch) to use for the projection "
                           "figure")
        return(dpi_box)

    # This function creates a figsize box
    @docstring_append(create_type_doc.format('figsize'))
    def create_type_figsize(self):
        return(FigSizeBox())

    # This function creates a linestyle box
    @docstring_append(create_type_doc.format('linestyle'))
    def create_type_linestyle(self):
        # Obtain list with all supported linestyles
        linestyles_lst = [(key, value[6:]) for key, value in lineStyles.items()
                          if value != '_draw_nothing']
        linestyles_lst.sort(key=lambda x: x[0])

        # Make combobox for linestyles
        linestyle_box = QW_QComboBox()
        for i, (linestyle, tooltip) in enumerate(linestyles_lst):
            linestyle_box.addItem(linestyle)
            linestyle_box.setItemData(i, tooltip, QC.Qt.ToolTipRole)
        set_box_value(linestyle_box, rcParams['lines.linestyle'])
        linestyle_box.setToolTip("Linestyle to be used for the corresponding "
                                 "plot type")
        return(linestyle_box)

    # This function creates a linewidth box
    @docstring_append(create_type_doc.format('linewidth'))
    def create_type_linewidth(self):
        # Make a double spinbox for linewidth
        linewidth_box = QW_QDoubleSpinBox()
        linewidth_box.setRange(0, 9999999)
        linewidth_box.setSuffix(" pts")
        set_box_value(linewidth_box, rcParams['lines.linewidth'])
        linewidth_box.setToolTip("Width of the plotted line")
        return(linewidth_box)

    # This function creates a marker box
    @docstring_append(create_type_doc.format('marker'))
    def create_type_marker(self):
        # Obtain list with all supported markers
        markers_lst = [(key, value) for key, value in lineMarkers.items()
                       if(value != 'nothing' and isinstance(key, str))]
        markers_lst.append(('', 'nothing'))
        markers_lst.sort(key=lambda x: x[0])

        # Make combobox for markers
        marker_box = QW_QComboBox()
        for i, (marker, tooltip) in enumerate(markers_lst):
            marker_box.addItem(marker)
            marker_box.setItemData(i, tooltip, QC.Qt.ToolTipRole)
        set_box_value(marker_box, rcParams['lines.marker'])
        marker_box.setToolTip("Marker to be used for the corresponding plot "
                              "type")
        return(marker_box)

    # This function creates a markersize box
    @docstring_append(create_type_doc.format('markersize'))
    def create_type_markersize(self):
        # Make a double spinbox for markersize
        markersize_box = QW_QDoubleSpinBox()
        markersize_box.setRange(0, 9999999)
        markersize_box.setSuffix(" pts")
        markersize_box.setToolTip("Size of the plotted markers")
        set_box_value(markersize_box, rcParams['lines.markersize'])
        return(markersize_box)

    # This function creates a scale box
    def create_type_scale(self, axis):
        """
        Base function for creating the entry types 'xscale' and 'yscale'.

        """

        # Make a combobox for scale
        scale_box = QW_QComboBox()
        scale_box.addItems(['linear', 'log'])
        scale_box.setToolTip("Scale type to use on the %s-axis" % (axis))
        return(scale_box)

    # This function creates a xscale box
    @docstring_append(create_type_doc.format('xscale'))
    def create_type_xscale(self):
        return(self.create_type_scale('x'))

    # This function creates a yscale box
    @docstring_append(create_type_doc.format('yscale'))
    def create_type_yscale(self):
        return(self.create_type_scale('y'))

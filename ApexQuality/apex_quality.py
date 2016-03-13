# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ApexQuality
                                 A QGIS plugin
 Automatic quality assessent of APEX data
                              -------------------
        begin                : 2016-03-11
        git sha              : $Format:%H$
        copyright            : (C) 2016 by RSL
        email                : gillian.milani@geo.uzh.ch
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, QUrl
from PyQt4.QtGui import QAction, QIcon, QToolButton, QMenu, QDesktopServices
# Initialize Qt resources from file resources.py
import resources

# Import the code for the dialog
from apex_quality_dialog import ApexQualityDialog
import os.path
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn import mixture
import numpy as np
from qgis import utils as qgis_utils


from spectral_utils import getSubset
from pyplot_widget import pyPlotWidget
#  from .build_spectral.lib import spectral


from kmeans_widget import KMeanWidget
from qgis_spectral_tool import SpectralTool


class ApexQuality:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'ApexQuality_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = ApexQualityDialog()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Apex Quality Assessment')

        self.spectralTool = SpectralTool(self.iface.mapCanvas())

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('ApexQuality', message)

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        icon_path = ':/plugins/ApexQuality/icon.png'
        self.action1 = QAction(QIcon(icon_path), u"K-Means classification", self.iface.mainWindow())
        self.action2 = QAction(QIcon(icon_path), u"Spectral tool", self.iface.mainWindow())
        self.action3 = QAction(QIcon(icon_path), u"Help", self.iface.mainWindow())
        self.actions.append(self.action1)
        self.actions.append(self.action2)
        self.actions.append(self.action3)
        self.popupMenu = QMenu(self.iface.mainWindow())
        self.popupMenu.addAction(self.action1)
        self.popupMenu.addAction(self.action2)
        self.popupMenu.addSeparator()
        self.popupMenu.addAction(self.action3)
        self.action1.triggered.connect(self.someMethod1)
        self.action2.triggered.connect(self.someMethod2)
        self.action3.triggered.connect(self.someMethod3)
        self.toolButton = QToolButton()
        self.toolButton.setMenu(self.popupMenu)
        self.toolButton.setDefaultAction(self.action1)
        self.toolButton.setPopupMode(QToolButton.InstantPopup)
        self.toolbar1 = self.iface.addToolBarWidget(self.toolButton)

    def someMethod1(self):
        filePath = self.getCurrentImage()
        subset = getSubset(filePath)
        if subset is None:
            return
        dialog = KMeanWidget()
        ok = dialog.exec_()
        if not ok:
            return
        if dialog.euclideanRadioButton.isChecked():
            distance = spectral.L2
        else:
            distance = spectral.L1
        (m, c) = spectral.kmeans(subset, dialog.classSpinBox.value(), dialog.iterationSpinBox.value(), distance = distance)

        c_plot = pyPlotWidget()
        ax = c_plot.figure.add_subplot(121)
        ax.hold(1)
        for i in range(c.shape[0]):
            ax.plot(c[i], color = plt.cm.gist_ncar(i / float(len(c) - 1)))  # @UndefinedVariable
        ax.hold(0)
        uniqu = np.unique(m)
        ax = c_plot.figure.add_subplot(122)
        imgplot = ax.imshow(m, cmap = plt.cm.gist_ncar , vmin = np.min(uniqu), vmax = np.max(uniqu))  # @UndefinedVariable
        c_plot.canvas.draw(); c_plot.show(); c_plot.exec_()

    def someMethod2(self):
        self.iface.mapCanvas().setMapTool(self.spectralTool)
        self.spectralTool.plot = pyPlotWidget()

    def someMethod3(self):
        path = os.path.dirname(os.path.realpath(__file__))
        url = QUrl('file://' + path + '/help/build/html/index.html')
        QDesktopServices.openUrl(url)

    def getCurrentImage(self):
        rlayer = qgis_utils.iface.mapCanvas().currentLayer()
        if rlayer == None:
            return
        else:
            return rlayer.source()

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Apex Quality Assessment'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        self.spectralTool.deactivate()
        del self.toolbar1



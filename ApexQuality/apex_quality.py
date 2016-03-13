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
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, QUrl, QFileInfo, Qt
from PyQt4.QtGui import QAction, QIcon, QToolButton, QMenu, QDesktopServices, QKeySequence
# Initialize Qt resources from file resources.py
import resources  # @UnusedImport

import os.path
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn import mixture
import numpy as np
from qgis import utils as qgis_utils
from qgis import core as qgis_core
from osgeo import gdal
from matplotlib.backends.backend_pdf import PdfPages
from pandas.tools.plotting import table
from pandas import DataFrame

from spectral_utils import getSubset
from pyplot_widget import pyPlotWidget
import customization
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
        self.action1 = QAction(QIcon(icon_path), u"Unsuperviseds classification", self.iface.mainWindow())
        self.action2 = QAction(QIcon(icon_path), u"Spectral tool", self.iface.mainWindow())
        self.action3 = QAction(QIcon(icon_path), u"Help", self.iface.mainWindow())
        self.action1.setShortcut(QKeySequence(Qt.SHIFT + Qt.CTRL + Qt.Key_Y))
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
        path = os.path.dirname(os.path.realpath(__file__))
        filePath = self.getCurrentImage()
        subset, xMin, yMax = getSubset(filePath)
        if subset is None:
            return
        dialog = KMeanWidget()
        ok = dialog.exec_()
        if not ok:
            return

        g = mixture.GMM(n_components = dialog.classSpinBox.value())
        h, w, n_b = subset.shape
        subset = subset.reshape(-1, n_b)
        pca = TruncatedSVD(n_components = dialog.nCompSvdSpinBox.value())
        subset_cp = pca.fit_transform(subset)
        m = g.fit_predict(subset_cp)
        proba = g.predict_proba(subset_cp)
        indicator = np.min(proba, axis = 1)
        indicator = np.reshape(indicator, (h, w))
        m = np.reshape(m, (h, w))
        c = pca.inverse_transform(g.means_)
        c_covar = pca.inverse_transform(g.covars_)

        if dialog.pyplotCB.isChecked():
            c_plot = pyPlotWidget()
            ax = c_plot.figure.add_subplot(321)
            ax.hold(1)
            for i in range(c.shape[0]):
                ax.plot(g.means_[i], color = plt.cm.gist_rainbow(i / float(len(c) - 1)))  # @UndefinedVariable)
            ax = c_plot.figure.add_subplot(322)
            for i in range(c.shape[0]):
                ax.plot(g.covars_[i], color = plt.cm.gist_rainbow(i / float(len(c) - 1)))  # @UndefinedVariable
            ax = c_plot.figure.add_subplot(323)
            for i in range(c.shape[0]):
                ax.plot(c[i], color = plt.cm.gist_rainbow(i / float(len(c) - 1)))  # @UndefinedVariable
            ax.set_ylim([0, 1])
            ax = c_plot.figure.add_subplot(324)
            for i in range(c.shape[0]):
                ax.plot(c_covar[i], color = plt.cm.gist_rainbow(i / float(len(c) - 1)))  # @UndefinedVariable
            ax.hold(0)
            uniqu = np.unique(m)
            ax = c_plot.figure.add_subplot(325)
            ax.imshow(m, cmap = plt.cm.gist_rainbow , vmin = np.min(uniqu), vmax = np.max(uniqu))  # @UndefinedVariable
            ax = c_plot.figure.add_subplot(326)
            imbar = ax.imshow(indicator, cmap = plt.cm.hot)  # @UndefinedVariable
            c_plot.figure.colorbar(imbar)
            c_plot.canvas.draw(); c_plot.show(); c_plot.exec_()

        if dialog.geotiffCB.isChecked():
            dataset1 = gdal.Open(filePath)
            geoTransform = list(dataset1.GetGeoTransform())
            geoTransform[0] += (xMin * geoTransform[1])
            geoTransform[3] += (yMax * geoTransform[5])
            r_save = np.array(m, dtype = np.uint8)
            r_save = np.reshape(r_save, (r_save.shape[0], r_save.shape[1], 1))

            self.WriteGeotiffNBand(r_save, path + '/temp/test.tiff', gdal.GDT_Byte, geoTransform, dataset1.GetProjection())

            fileInfo = QFileInfo(path + '/temp/test.tiff')
            baseName = fileInfo.baseName()
            rlayer = qgis_core.QgsRasterLayer(path + '/temp/test.tiff', baseName)
            qgis_core.QgsMapLayerRegistry.instance().addMapLayer(rlayer)

        if dialog.pdfCB.isChecked():
            outputFile = path + '/temp/test.pdf'
            with PdfPages(outputFile) as pdf:
                c_plot = pyPlotWidget()
                ax = c_plot.figure.add_subplot(221)
                ax.set_title('SVD Classes')
                ax.hold(1)
                for i in range(c.shape[0]):
                    ax.plot(g.means_[i], color = plt.cm.gist_rainbow(i / float(len(c) - 1)))  # @UndefinedVariable)
                ax = c_plot.figure.add_subplot(222)
                ax.set_title('SVD variances')
                for i in range(c.shape[0]):
                    ax.plot(g.covars_[i], color = plt.cm.gist_rainbow(i / float(len(c) - 1)))  # @UndefinedVariable
                ax = c_plot.figure.add_subplot(223)
                ax.set_title('Spectrum Classes')
                for i in range(c.shape[0]):
                    ax.plot(c[i], color = plt.cm.gist_rainbow(i / float(len(c) - 1)))  # @UndefinedVariable
                ax.set_ylim([0, 1])
                ax = c_plot.figure.add_subplot(224)
                ax.set_title('Spectrum Variances')
                for i in range(c.shape[0]):
                    ax.plot(c_covar[i], color = plt.cm.gist_rainbow(i / float(len(c) - 1)))  # @UndefinedVariable
                ax.hold(0)
                uniqu = np.unique(m)
                pdf.savefig(c_plot.figure)

                c_plot = pyPlotWidget()
                ax = c_plot.figure.add_subplot(221)
                ax.set_title('Classification')
                ax.imshow(m, cmap = plt.cm.gist_rainbow , vmin = np.min(uniqu), vmax = np.max(uniqu))  # @UndefinedVariable
                ax = c_plot.figure.add_subplot(222)
                ax.set_title('Minimal distance to class')
                imbar = ax.imshow(indicator, cmap = plt.cm.hot)  # @UndefinedVariable
                c_plot.figure.colorbar(imbar)
                ax = c_plot.figure.add_subplot(223)
                for i in range(c.shape[0]):
                    ax.plot([0], [0], color = plt.cm.gist_rainbow(i / float(len(c) - 1)), label = i)  # @UndefinedVariable
                ax.legend()
                ax.axis('off')
                c_plot.canvas.draw()
                pdf.savefig(c_plot.figure)

                class_list = []
                for class_i in range(np.min(uniqu), np.max(uniqu) + 1):
                    class_list.append(np.reshape(m == class_i, (-1)))

                nn = 12
                t_n = int(np.ceil(n_b / float(nn)))
                for i in range(nn):
                    c_plot = pyPlotWidget()
                    sub = subset[:, (i * t_n):np.min((((i + 1) * t_n), n_b))]
                    for j, class_i in enumerate(range(np.min(uniqu), np.max(uniqu) + 1)):
                        print i, j
                        ax = c_plot.figure.add_subplot(1, np.max(uniqu) + 1, class_i + 1)
                        bool_i = class_list[j]
                        subset_class = sub[bool_i, :]
                        ax.set_title('Class %s' % class_i)
                        ax.axis('off')
                        results, headers = customization.compute_stats_per_class(subset_class)
                        matrix = np.transpose(np.array(results))
                        df = DataFrame(matrix, columns = headers, dtype = np.float32)
                        table(ax, df, rowLabels = range((i * t_n) + 1, np.min((((i + 1) * t_n), n_b)) + 1), loc = 'upper right', colWidths = [1.0 / matrix.shape[1]] * matrix.shape[1])
                        c_plot.canvas.draw()
                    pdf.savefig(c_plot.figure)
                    break
            url = QUrl('file://' + outputFile)
            QDesktopServices.openUrl(url)

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
        self.iface.removeToolBarIcon(self.toolbar1)
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Apex Quality Assessment'),
                action)

            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        self.spectralTool.deactivate()


    def WriteGeotiffNBand(self, raster, filepath, dtype, vectReference, proj):
        nrows, ncols, n_b = np.shape(raster)
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(filepath, ncols, nrows, n_b, dtype, ['COMPRESS=LZW'])
        dst_ds.SetProjection(proj)
        dst_ds.SetGeoTransform(vectReference)
        for i in range(n_b):
            R = np.array(raster[:, :, i], dtype = np.float32)
            dst_ds.GetRasterBand(i + 1).WriteArray(R)  # Red
            dst_ds.GetRasterBand(i + 1).SetNoDataValue(-1)
        dst_ds = None

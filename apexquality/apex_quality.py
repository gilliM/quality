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
from PyQt4.QtGui import QAction, QIcon, QToolButton, QMenu, QDesktopServices, QKeySequence, QColor
# Initialize Qt resources from file resources.py
import resources  # @UnusedImport

import os.path
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn import mixture
import numpy as np
from qgis import utils as qgis_utils
from qgis import core as qgis_core
from qgis import gui as qgis_gui
from osgeo import gdal
from matplotlib.backends.backend_pdf import PdfPages
from pandas.tools.plotting import table
from pandas import DataFrame
import spectral

from spectral_utils import getSubset
from pyplot_widget import pyPlotWidget
import customization


from kmeans_widget import KMeanWidget
from qgis_spectral_tool import SpectralTool


class ApexQuality:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor."""
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
        self.path = os.path.dirname(os.path.realpath(__file__))

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
        dialog = KMeanWidget()
        ok = dialog.exec_()
        if not ok:
            return

        filePath = self.getCurrentImage()
        if filePath is None:
            qgis_utils.iface.messageBar().pushMessage("Error",
                "No Raster selected", level = qgis_gui.QgsMessageBar.CRITICAL, duration = 5)
            return

        n_class = dialog.classSpinBox.value()
        cmap = plt.get_cmap('gist_rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, n_class)]

        if dialog.restrictedRB.isChecked():
            data, xMin, yMax = getSubset(filePath)
            if data is None:
                qgis_utils.iface.messageBar().pushMessage("Error",
                    "Problem of extend", level = qgis_gui.QgsMessageBar.CRITICAL, duration = 5)
                return
            g = mixture.GMM(n_components = n_class)
            h, w, n_b = data.shape
            data = data.reshape(-1, n_b)
            pca = TruncatedSVD(n_components = dialog.nCompSvdSpinBox.value())
            data_cp = pca.fit_transform(data)
            m = g.fit_predict(data_cp)
            indicator = np.max(g.predict_proba(data_cp), axis = 1)
            indicator = np.reshape(indicator, (h, w))
            m = np.reshape(m, (h, w))
            c = pca.inverse_transform(g.means_)
            c_covar = pca.inverse_transform(g.covars_)

        else:
            img = spectral.open_image(filePath.replace('.bsq', '.hdr'))
            data = img.load()
            xMin = 0; yMax = 0

            g = mixture.GMM(n_components = n_class)
            h, w, n_b = data.shape
            data = data.reshape(-1, n_b)
            subset = data[np.random.choice(data.shape[0], 100000)]
            pca = TruncatedSVD(n_components = dialog.nCompSvdSpinBox.value())
            pca.fit(subset)
            subset_pc = pca.transform(subset)
            data_pc = pca.transform(data)
            g.fit(subset_pc)
            m = g.predict(data_pc)
            indicator = np.max(g.predict_proba(data_pc), axis = 1)
            indicator = np.reshape(indicator, (h, w))
            m = np.reshape(m, (h, w))
            c = pca.inverse_transform(g.means_)
            c_covar = pca.inverse_transform(g.covars_)

        if dialog.pyplotCB.isChecked():
            self.c_plot = pyPlotWidget()
            ax = self.c_plot.figure.add_subplot(431)
            ax.hold(1)
            for i in range(c.shape[0]):
                ax.plot(g.means_[i], color = colors[i])
            ax = self.c_plot.figure.add_subplot(432)
            for i in range(c.shape[0]):
                ax.plot(g.covars_[i], color = colors[i])
            ax = self.c_plot.figure.add_subplot(434)
            for i in range(c.shape[0]):
                ax.plot(c[i], color = colors[i])
            ax.set_ylim([0, 1])
            ax = self.c_plot.figure.add_subplot(435)
            for i in range(c.shape[0]):
                ax.plot((c_covar[i]), color = colors[i])
            ax.hold(0)
            uniqu = np.unique(m)
            ax = self.c_plot.figure.add_subplot(437)
            ax.imshow(m, cmap = cmap , vmin = np.min(uniqu), vmax = np.max(uniqu))
            ax = self.c_plot.figure.add_subplot(438)
            imbar = ax.imshow(indicator, cmap = plt.cm.hot)  # @UndefinedVariable
            self.c_plot.figure.colorbar(imbar)
            ax = self.c_plot.figure.add_subplot(4, 3, 10)
            for i in range(c.shape[0]):
                ax.plot([0], [0], color = colors[i], label = i)
            ax.legend()
            ax.axis('off')
            ax = self.c_plot.figure.add_subplot(4, 3, 11)
            for i in range(c.shape[0]):
                ax.plot((c_covar[i] / c[i]), color = colors[i])
            ax.hold(0)
            self.c_plot.canvas.draw();
            self.c_plot.show();
            self.c_plot.raise_()

            class_list = []
            for class_i in range(np.min(uniqu), np.max(uniqu) + 1):
                class_list.append(np.reshape(m == class_i, (-1)))


            colors = [cmap(i) for i in np.linspace(0, 1, np.max(uniqu) + 1 - np.min(uniqu))]
            ax = self.c_plot.figure.add_subplot(1, 3, 3)
            for j, class_i in enumerate(range(n_class)):
                # ax = self.c_plot.figure.add_subplot(np.max(uniqu) + 1 - np.min(uniqu), 3, 3 * (j + 1))
                bool_i = class_list[j]
                data_class = data[bool_i, :]
                # ax.axis('off')
                results, headers = customization.compute_stats_per_class(data_class)
                color = colors[j]
                ax.plot(j + results[1], '-', color = color)
                ax.plot(j + results[1] + results[3], '-', color = color)
                ax.plot(j + results[1] - results[3], '-', color = color)
                ax.plot(j + results[0], '-.', color = color)
                ax.plot(j + results[2], '--', color = color)
                ax.set_ylim([0, j + 1])

                ax.set_axis_off()
                ax.set_frame_on(True)
                ax.set_axis_bgcolor('w')
            # ax.plot([0], [0], 'k-', label = 'mean')
            # ax.plot([0], [0], 'k--', label = 'max')
            # ax.plot([0], [0], 'k-.', label = 'min')
            # ax.legend()
            self.c_plot.figure.subplots_adjust(left = 0.02, right = 0.98, top = 0.98, bottom = 0.1, wspace = 0.05, hspace = 0.05)
            self.c_plot.canvas.draw();
            self.c_plot.showMaximized();
            self.c_plot.exec_()

        if dialog.geotiffCB.isChecked():
            dataset1 = gdal.Open(filePath)
            geoTransform = list(dataset1.GetGeoTransform())
            geoTransform[0] += (xMin * geoTransform[1])
            if geoTransform[5] > 0: geoTransform[5] *= -1
            geoTransform[3] += (yMax * geoTransform[5])
            r_save = np.array(m, dtype = np.uint8)
            r_save = np.reshape(r_save, (r_save.shape[0], r_save.shape[1], 1))

            self.WriteGeotiffNBand(r_save, self.path + '/temp/temp.tiff', gdal.GDT_Byte, geoTransform, dataset1.GetProjection())
            fileInfo = QFileInfo(self.path + '/temp/test.tiff')
            baseName = fileInfo.baseName()
            rlayer = qgis_core.QgsRasterLayer(self.path + '/temp/temp.tiff', baseName)



            fcn = qgis_core.QgsColorRampShader()
            fcn.setColorRampType(qgis_core.QgsColorRampShader.EXACT)
            lst = [ qgis_core.QgsColorRampShader.ColorRampItem(j, QColor(colors[j][0] * 255, colors[j][1] * 255, colors[j][2] * 255)) for j in range(n_class) ]
            fcn.setColorRampItemList(lst)
            shader = qgis_core.QgsRasterShader()
            shader.setRasterShaderFunction(fcn)

            renderer = qgis_core.QgsSingleBandPseudoColorRenderer(rlayer.dataProvider(), 1, shader)
            rlayer.setRenderer(renderer)
            qgis_core.QgsMapLayerRegistry.instance().addMapLayer(rlayer)


            r_save = np.array(indicator, dtype = np.float32)
            r_save = np.reshape(r_save, (r_save.shape[0], r_save.shape[1], 1))
            self.WriteGeotiffNBand(r_save, self.path + '/temp/temp_indicator.tiff', gdal.GDT_Float32, geoTransform, dataset1.GetProjection())
            fileInfo = QFileInfo(self.path + '/temp/test.tiff')
            baseName = fileInfo.baseName()
            rlayer = qgis_core.QgsRasterLayer(self.path + '/temp/temp_indicator.tiff', baseName)
            qgis_core.QgsMapLayerRegistry.instance().addMapLayer(rlayer)



        if dialog.pdfCB.isChecked():
            outputFile = self.path + '/temp/test.pdf'
            with PdfPages(outputFile) as pdf:
                c_plot = pyPlotWidget()
                ax = c_plot.figure.add_subplot(221)
                ax.set_title('SVD Classes')
                ax.hold(1)
                for i in range(c.shape[0]):
                    ax.plot(g.means_[i], color = colors[i])  # @UndefinedVariable)
                ax = c_plot.figure.add_subplot(222)
                ax.set_title('SVD variances')
                for i in range(c.shape[0]):
                    ax.plot(g.covars_[i], color = colors[i])  # @UndefinedVariable
                ax = c_plot.figure.add_subplot(223)
                ax.set_title('Spectrum Classes')
                for i in range(c.shape[0]):
                    ax.plot(c[i], color = colors[i])  # @UndefinedVariable
                ax.set_ylim([0, 1])
                ax = c_plot.figure.add_subplot(224)
                ax.set_title('Spectrum Variances')
                for i in range(c.shape[0]):
                    ax.plot(c_covar[i], color = colors[i])  # @UndefinedVariable
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
                ax.set_title('Classification')
                original = np.reshape(data, (h, w, n_b))
                img = np.transpose(np.array((self.getBand(original, 39), self.getBand(original, 17), self.getBand(original, 6))), (1, 2, 0))
                ax.imshow(img, interpolation = "nearest")

                ax = c_plot.figure.add_subplot(224)
                for i in range(c.shape[0]):
                    ax.plot([0], [0], color = colors[i], label = i)  # @UndefinedVariable
                ax.legend(prop = {'size':int(96.0 / n_class)})
                ax.axis('off')
                c_plot.canvas.draw()
                pdf.savefig(c_plot.figure)

                class_list = []
                for class_i in range(np.min(uniqu), np.max(uniqu) + 1):
                    class_list.append(np.reshape(m == class_i, (-1)))

                nn = 5
                n_pages = int(np.ceil(n_class / float(nn)))
                r = range(np.min(uniqu), np.max(uniqu) + 1)

                for p in range(n_pages):
                    c_plot = pyPlotWidget()
                    ax = c_plot.figure.add_subplot(1, 1, 1)
                    for j, class_i in enumerate(r[(p * nn): np.min([((p + 1) * nn), len(r)])]):
                        # ax = self.c_plot.figure.add_subplot(np.max(uniqu) + 1 - np.min(uniqu), 3, 3 * (j + 1))
                        bool_i = class_list[j + p * nn]
                        data_class = data[bool_i, :]
                        # ax.axis('off')
                        results, headers = customization.compute_stats_per_class(data_class)
                        color = colors[j + p * nn]
                        ax.plot(nn - 1 - j + results[1], '-', color = color)
                        ax.plot(nn - 1 - j + results[1] + results[3], ':', color = color)
                        ax.plot(nn - 1 - j + results[1] - results[3], ':', color = color)
                        ax.plot(nn - 1 - j + results[0], '-.', color = color)
                        ax.plot(nn - 1 - j + results[2], '--', color = color)
                        ax.set_ylim([0, nn])

                        ax.set_axis_off()
                        ax.set_frame_on(True)
                        ax.set_axis_bgcolor('w')
                    pdf.savefig(c_plot.figure)

                """
                ### The following part was used to write tables to the pdf
                nn = 12
                t_n = int(np.ceil(n_b / float(nn)))
                for i in range(nn):
                    c_plot = pyPlotWidget()
                    sub = data[:, (i * t_n):np.min((((i + 1) * t_n), n_b))]
                    for j, class_i in enumerate(range(np.min(uniqu), np.max(uniqu) + 1)):
                        print i, j
                        ax = c_plot.figure.add_subplot(1, np.max(uniqu) + 1, class_i + 1)
                        bool_i = class_list[j]
                        data_class = sub[bool_i, :]
                        ax.set_title('Class %s' % class_i)
                        ax.axis('off')
                        results, headers = customization.compute_stats_per_class(data_class)
                        matrix = np.transpose(np.array(results))
                        df = DataFrame(matrix, columns = headers, dtype = np.float32)
                        table(ax, df, rowLabels = range((i * t_n) + 1, np.min((((i + 1) * t_n), n_b)) + 1), loc = 'upper right', colWidths = [1.0 / matrix.shape[1]] * matrix.shape[1])
                        c_plot.canvas.draw()
                    pdf.savefig(c_plot.figure)
                """
            url = QUrl('file://' + outputFile)
            QDesktopServices.openUrl(url)

    def getBand(self, array, i):
        val = array[:, :, i]
        max = np.percentile(val, 98.0)  #  / 1.5
        min = np.percentile(val, 2.0)  # * 1.5
        val = (val - min) / (max - min)
        val[val < 0] = 0
        val[val > 1] = 1
        val = np.array(np.round(val * 255), dtype = np.uint8)
        return val


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

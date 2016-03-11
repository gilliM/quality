# -*- coding: utf-8 -*-
"""
/***************************************************************************
                                 A QGIS plugin
 general plugin
                             -------------------
        begin                : 2015-06-25
        git sha              : $Format:%H$
        copyright            : (C) 2015 by Gillian Milani at RSL
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

import qgis  # @UnresolvedImport
from qgis.gui import *
from qgis.core import *
from PyQt4.QtCore import Qt, SIGNAL, QSettings
from PyQt4.QtGui import QApplication
import numpy as np

s = QSettings()

class SpectralTool(QgsMapToolEmitPoint):  # @UndefinedVariable
    def __init__(self, canvas):
        self.canvas = canvas
        QgsMapToolEmitPoint.__init__(self, self.canvas)  # @UndefinedVariable
        self.reset()
        self.plot = None

    def reset(self):
        self.startPoint = self.endPoint = None
        self.isEmittingPoint = False

    def canvasPressEvent(self, e):
        rlayer = qgis.utils.iface.mapCanvas().currentLayer()
        if rlayer == None:
            return
        point = self.toMapCoordinates(e.pos())
        self.isEmittingPoint = True
        ident = rlayer.dataProvider().identify(QgsPoint(point), QgsRaster.IdentifyFormatValue)  # @UndefinedVariable
        asort_index = np.argsort(ident.results().keys())
        a_values = np.array(ident.results().values())[asort_index]
        print a_values
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            self.plot.figure.clf()
            self.getCorrelation(a_values)
        else:
            if modifiers == Qt.ControlModifier:
                self.plot.figure.clf()

            ax = self.plot.figure.add_subplot(111)
            ax.plot(asort_index, a_values)
            self.plot.canvas.draw(); self.plot.show(); self.plot.exec_()

    def canvasReleaseEvent(self, e):
        self.isEmittingPoint = False

    def canvasMoveEvent(self, e):
        if not self.isEmittingPoint:
            return
        rlayer = qgis.utils.iface.mapCanvas().currentLayer()
        if rlayer == None:
            return
        point = self.toMapCoordinates(e.pos())

        ident = rlayer.dataProvider().identify(QgsPoint(point), QgsRaster.IdentifyFormatValue)  # @UndefinedVariable
        asort_index = np.argsort(ident.results().keys())
        a_values = np.array(ident.results().values())[asort_index]
        self.plot.figure.clf()
        ax = self.plot.figure.add_subplot(111)
        ax.plot(asort_index, a_values)
        self.plot.canvas.draw(); self.plot.show(); self.plot.exec_()

    def deactivate(self):
        self.plot = None
        try:
            super(SpectralTool, self).deactivate()
        except:
            pass
        try:
            self.emit(SIGNAL("deactivated()"))
        except:
            pass



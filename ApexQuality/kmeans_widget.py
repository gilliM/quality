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

import os
from PyQt4 import QtGui, uic, QtCore, QtSql
s = QtCore.QSettings()

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'uis/spectral_kmean_widget_base.ui'))


class KMeanWidget(QtGui.QDialog, FORM_CLASS):
    def __init__(self, parent = None):
        """Constructor."""
        super(KMeanWidget, self).__init__(parent)
        self.setupUi(self)
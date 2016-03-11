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
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication
from PyQt4.QtGui import QAction, QIcon, QToolButton, QMenu
# Initialize Qt resources from file resources.py
import resources

# Import the code for the dialog
from apex_quality_dialog import ApexQualityDialog
import os.path


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
        self.action1 = QAction(QIcon(icon_path), u"Action 1", self.iface.mainWindow())
        self.action2 = QAction(QIcon(icon_path), u"Action 2", self.iface.mainWindow())
        self.action3 = QAction(QIcon(icon_path), u"Action 3", self.iface.mainWindow())
        self.actions.append(self.action1)
        self.actions.append(self.action2)
        self.actions.append(self.action3)
        self.popupMenu = QMenu(self.iface.mainWindow())
        self.popupMenu.addAction(self.action1)
        self.popupMenu.addAction(self.action2)
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
        pass

    def someMethod2(self):
        pass

    def someMethod3(self):
        pass


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Apex Quality Assessment'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar1


    def run(self):
        """Run method that performs all the real work"""
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass

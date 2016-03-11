# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ApexQuality
                                 A QGIS plugin
 Automatic quality assessent of APEX data
                             -------------------
        begin                : 2016-03-11
        copyright            : (C) 2016 by RSL
        email                : gillian.milani@geo.uzh.ch
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load ApexQuality class from file ApexQuality.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .apex_quality import ApexQuality
    return ApexQuality(iface)

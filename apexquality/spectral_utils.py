# -*- coding: utf-8 -*-
'''
Created on Mar 11, 2016

@author: gmilani
'''

import spectral
import gdal
from qgis import utils as qgis_utils
from PyQt4.QtGui import QMessageBox

def world2Pixel(geoMatrix, x, y):
    # Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    # the pixel location of a geospatial coordinate
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / (-1 * yDist))
    return (pixel, line)

def getSubset(filePath, extend = None):
    if extend is None:
        iface = qgis_utils.iface
        e = iface.mapCanvas().extent()
        xMax = e.xMaximum()
        yMax = e.yMaximum()
        xMin = e.xMinimum()
        yMin = e.yMinimum()
        srcImage = gdal.Open(filePath)
        geoTrans = srcImage.GetGeoTransform()
        print geoTrans
        if geoTrans[0] == 0 and geoTrans[3] == 0:
            # apparently no projection
            minImage = [int(xMin), -int(yMin)]
            maxImage = [int(xMax), -int(yMax)]
            print minImage
            print maxImage
        else:
            # image has projection
            minImage = list(world2Pixel(geoTrans, xMin, yMin))
            maxImage = list(world2Pixel(geoTrans, xMax, yMax))
            if minImage[0] < 0: minImage[0] = 0
            if minImage[1] < 0: minImage[1] = 0
            if maxImage[0] > srcImage.RasterXSize: maxImage[0] = srcImage.RasterXSize
            if maxImage[1] > srcImage.RasterYSize: maxImage[1] = srcImage.RasterYSize
    else:
        minImage = extend[0:2]
        maxImage = extend[2:4]

    if ((minImage[1] - maxImage[1]) * (maxImage[0] - minImage[0])) > 10 ** 7:
        QMessageBox.critical(None, u"Size error", u"The current extent is too large")
        return None
    if '.bsq' in filePath:
        f = filePath.replace('.bsq', '.hdr')
    else:
        f = filePath + '.hdr'
    return spectral.open_image(f).read_subregion([maxImage[1], minImage[1]], [minImage[0], maxImage[0]]), minImage[0], maxImage[1]

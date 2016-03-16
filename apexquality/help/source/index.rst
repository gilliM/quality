.. ApexQuality documentation master file, created by
   sphinx-quickstart on Sun Feb 12 17:11:03 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ApexQuality's documentation!
============================================

.. toctree::
   :maxdepth: 2


Unsupervised classification
===========================
The current unsupervised classification allow to work on the current extend of the QGIS canvas. The classification is performed on the current selected layer of the layer tree.

A truncated SVD is apply to reduce to data dimenstionality. A Gaussian mixture model is used as classifier.

Unconsistencies in the data can be observed trough 2 means:

* Form of the class
* High variance of a class
* Per-pixel analysis of the distance to the closer class

Form of the class
-------------------

If a lot of pixel exhibits inconsistency, but are all similar, the problem can be observred directly in the mean of spectra of the class (5th graph)

High variance of class
------------------------

The high variance of a class can mean that different landcover are assigned to this class. From some experience of the classifier, weird spectra are often included in mixed classes. The variance can be observed in the 2nd and 4th graph

Per-pixel analysis of the distance to the closer class
-------------------------------------------------------

Each pixel is assigned to the class to which its probability of belonging is the higher. If the highest probability is low, it means that the given pixel is far from every centroid, so far from most of the pixel. It can be considered as an outlier. The highest probability is shown on the 6th graph. Lower is the probability, further the pixel is from any class.


Spectral tool
=============
The spectral tool can be used in three modes:

* click: plot the spectra of the given pixel. A second click add a second spectra to the previous one
* click and drag: reset the plot at each drawing
* click + CTRL (CMD on mac): reset the plot at each drawing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. Magnify documentation master file, created by
   sphinx-quickstart on Wed Feb 22 12:01:48 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. toctree::
   :maxdepth: 1
   :caption: API:

   assay


Magnify: Image Processing for Microfluidics
===========================================
Magnify is a Python library for processing images of microfluidic experiments.
The API is designed to be simple and intuitive, while also being expressive enough to
implement complex image processing pipelines. The library has built-in pipelines for
bead images and mitomi devices.

Setup
-----
To install Magnify, run:

.. code-block:: console

    $ pip install magnify

Usage
-----
.. code-block:: python

    import magnify

    # Load the pipeline for generic bead finding.
    pipe = magnify.load("beads")

    # Process microscopy images taken over multiple timesteps.
    assay = pipe("images.ome.tiff", search_on="egfp")

    # Print out the channels: ["egfp", "mcherry"].
    print(assay.channels)

    # Print the mean intensity of the beads on timestep 10 in the egfp channel.
    print(assay.fg_mean[10, 0])

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




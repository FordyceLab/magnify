# Magnify
A Python toolkit for processing microscopy images. Magnify makes it easy to work with terabyte-scale imaging datasets on your laptop. It provides a unified interface for any task that involves finding and processing markers of interest such as [**beads**](https://www.nature.com/articles/s41378-020-00220-3), [**droplets**](https://pubs.acs.org/doi/pdf/10.1021/acs.analchem.0c02499), **cells**, and [**microfluidic device components**](https://www.science.org/doi/full/10.1126/science.abf8761).

Magnify comes with predefined [processing pipelines](https://github.com/FordyceLab/magnify/blob/main/src/magnify/registry.py) that support **file-parsing**, **image stitching**, **flat-field correction**, **image segmentation**, **tag identification**, and **marker filtering** across many different marker types. Magnify's pipelines allow you to process your images in just a few lines of code, while also being easy to extend with your own custom pipeline components.

## Setup
```sh
pip install magnify
```

## Usage
Here's a minimal example of how to use magnify to find, analyze, and visualize lanthanide-encoded beads given a microscopy image.
```python
import magnify as mg
import magnify.plot as mp

# Process the experiment and get the output as an xarray dataset.
xp = mg.mrbles("example.ome.tif", search_channel="620", min_bead_radius=10)

# Get the mean bead area and intensity.
print("Mean Bead Area:", xp.fg.sum(dim=["roi_x", "roi_y"]).mean())
print("Mean Bead Intensity:", xp.where(xp.fg).roi.mean())

# Show all the beads and how they were segmented.
mp.imshow(xp)
```
![](static/imshow.gif)

## Core Concepts
### Output Format
Magnify outputs its results as xarray datasets. If you are unfamiliar with xarrays you might want to read [this quick overview](https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html) after you're setup with magnify. An xarray dataset is essentially a dictionary of arrays with named dimensions. Let's look at the Jupyter notebook output for a simple example where we've only used magnify to segment beads.

![](static/xarray.png)

In most cases the actual data only consists of the processed images and regions of interest (ROI) around segmented markers. We also have coordinates which are arrays that represent metadata in our dataset, such as the location of the foreground (fg) and background (bg) in each ROI. The image below shows a graphical illustration of these concepts.
![](static/xarray-components.png)

In this example the image array was 2-dimensional (`image_height x image_width`) and the ROI array was 3-dimensional (`num_markers x roi_height x roi_width`). However, magnify can also process stacks of images that were collected across multiple timepoints and color channels, so the image array can have up to 4 dimensions (`num_timepoints x num_channels x ROI_height x ROI_width`) and the ROI array can have up to 5 dimensions (`num_markers x num_timepoints x num_channels x ROI_height x ROI_width`).

Also important for large datasets is how the data is stored. The `fg`, `bg`, `roi`, `image` arrays are stored on your hard drive rather than on RAM using [Dask](https://docs.dask.org/en/stable/presentations.html). This allows you to interact with much larger datasets at the cost of slower access times. You usually don't need to worry about this since Dask operates on subsets of the array in an intelligent way. But if you're finding that your analysis takes too long you might want to compute some summary information (e.g. the mean intensity of each marker) that fits in RAM, load that array into memory with [`compute`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.compute.html), and interact primarily with that array moving forward.

### File Parsing
Since a single experiment can consist of many files spread out across many folders, magnify allows you to retrieve many files using a single string. For example, let's say you've acquired an image across multiple channels stored in the following folder structure:
```text
.
├── egfp/
│   └── image1.tif
├── cy5/
│   └── image2.tif
└── dapi/
    └── image3.tif
```

You can load all these images into magnify with
```python
xp = pipe("(channel)/*.tif")
```
The search string supports [globs](https://en.wikipedia.org/wiki/Glob_(programming)) so `*` expands to match anything that matches the pattern. `(channel)` also expands like `*` but it also saves the segment of the file path it matches in the resulting dataset as the  channel name. The specifiers that allow you to read metadata from the file path are:
- `(assay)`: The name of distinct experiments, if this is provided magnify returns a list of datasets (rather than a single dataset).
- `(time)`: The time at which the image was acquired in format YYYYmmDD-HHMMSS. If your files specify acquisition times in a different format you can write `(time|FORMAT)` where `FORMAT` is a [legal format code for Python's strptime](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) (e.g. `(time|%H:%M:%S)`).
- `(channel)`: The channel in which the image was acquired.
- `(row)` and `(col)`: In the case of a tiled image these two specifiers indicate the row and column of the subimages. Magnify will stitch all these tiles into one large image.
- Alternate coordinates: You can also attach additional information to each coordinate using a specifier that looks like: `(INFO_COORD)` where `COORD` is the name of the original coordinate and `INFO` is the name for the attached information for example, `(concentration_time)`. By default magnify encodes the information as strings but you can specify alternate formats using `(INFO__COORD|FORMAT)` where `FORMAT` can be: `time`, `int`, or `float`.

Magnify can read any [TIFF](https://en.wikipedia.org/wiki/TIFF) image file. It can also read [OME-TIFF](https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/) files that were generated by [micromanager](https://micro-manager.org/). We plan to add support for other input formats as needed.

### Pipelines
If you don't need customized pipelines and just want to use the predefined pipelines you can skip this section.
Magnify's pipeline system is heavily inspired by [spaCy's pipelines](https://spacy.io/), so if you're familiar with that library you might only need to skim this section.

TODO: Write this. For now you can read the [spacy pipeline docs](https://spacy.io/usage/processing-pipelines) to get an idea of the design philosophy in magnify.

### Plotting
Magnify includes a plotting sublibrary which you can import with `import magnify.plot as mp`. It is designed to enable rapid prototyping of interactive visualization, primarily for the purpose of troubleshooting experiments rather than creating publication-ready figures. The plotting library is still under development and isn't currently stable.

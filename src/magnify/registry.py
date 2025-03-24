import functools
import inspect
from io import StringIO
from os import PathLike

import catalogue
import xarray as xr
from numpy.typing import ArrayLike

from magnify.pipeline import Pipeline

readers = catalogue.create("magnify", "readers")
components = catalogue.create("magnify", "components")


def component(name):
    def component_decorator(func):
        @functools.wraps(func)
        def component_factory(*args, **kwargs):
            return functools.partial(func, *args, **kwargs)

        # Make the factory signature identical to func's except for the first argument.
        signature = inspect.signature(func)
        signature = signature.replace(parameters=list(signature.parameters.values())[1:])
        component_factory.__signature__ = signature
        components.register(name)(component_factory)
        return func

    return component_decorator


def microfluidic_chip(
    data: ArrayLike | str,
    shape: tuple[int, int] = (8, 8),
    pinlist: str | None = None,
    blank: str | list[str] | None = None,
    overlap: int = 102,
    rotation: int = 0,
    row_dist: float = 375 / 1.61,
    col_dist: float = 400 / 1.61,
    chip_type: str | None = None,
    min_button_diameter: int = 8,
    max_button_diameter: int = 30,
    chamber_diameter: int = 60,
    top_chamber: int | None = None,
    left_chamber: int | None = None,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.2,
    cluster_penalty: float = 50,
    roi_length: int | None = None,
    progress_bar: bool = False,
    search_timestep: int | list[int] = 0,
    search_channel: str | list[str] | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
    interactive: bool = False,
) -> xr.Dataset | list[xr.Dataset]:
    """
    Find buttons in minichip images and standardize the resulting data in an xarray.Dataset

    Parameters
    ----------
    data :
        The input image data to be processed. It can be one of the following:
        - A file path or glob to image data.
        - An `xarray.DataArray` or `xarray.Dataset` containing image data.
        - A sequence of file paths, `xarray.DataArray`, or `xarray.Dataset`.
    shape :
        The shape of the button array, specifying the number of rows and columns in the image grid.
    pinlist :
        A file path to a CSV file that describes the tag to be assigned to each chamber on the chip. The CSV file must include
        a column called `Indices` that contains row and column pairs in the format `(row, col)`, and a
        `MutantID` column that contains the names of the buttons. Either `pinlist` or `shape` must be provided.
    blank :
        Values representing "blank" or non-expressed buttons in the pinlist, which will be replaced
        with empty strings in the dataset. Defaults to `["", "blank", "BLANK"]`.
    overlap :
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
    rotation :
        The degree of rotation to apply to the stitched image.
    row_dist, col_dist :
        The distance between rows/columns of buttons in pixels.
    chip_type :
        The type of microfluidic chip that was imaged. Can be one of ["minichip"|"pc"|"ps"], if `chip_type` is not None then it will override `row_dist` and `col_dist`.
    min_button_diameter, max_button_diameter :
        The minimum/maximum diameter in pixels for detecting buttons in the image.
    chamber_diameter :
        The diameter in pixels of the chamber around each button.
    top_chamber, left_chamber :
        The pixel offset of the edge of the top/leftmost chamber on the chip. If set to `None` the offset is automatically found.
    low_edge_quantile, high_edge_quantile :
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons. Must be between 0 and 1.
    num_iter :
        The maximum number of iterations to perform the button detection process using RANSAC. A higher value will take longer but will find buttons more accurately.
    min_roundness :
        The minimum roundness value for detected buttons. Buttons that do not meet this roundness threshold are excluded. Must be between 0 and 1.
    cluster_penalty :
        A penalty number that balances two factors when identifying clusters: penalizing high inter-cluster variance and
        penalizing deviations from the expected number of items in a cluster. A higher value places more emphasis on the second factor
    roi_length :
        The length in pixels of the region of interest (ROI) around detected buttons. If None, the ROI length is set to 1.2 * `chamber_diameter`.
    progress_bar :
        If True, display a progress bar during processing to track the progress of the pipeline.
    search_timestep :
        The timesteps on which to search for buttons. A timestep that isn't in search_timestep will use the same button locations
        as the closest searched timestep before it, or if there isn't one the closest timestep after it.
    search_channel :
        The channel or list of channels to use for button detection and expression filtering. If `None`, all channels will be used.
    squeeze :
        If True, removes any dimensions of length 1.
    roi_only :
        If True, only returns the region of interest from the dataset.
    drop_tiles :
        If True, removes the "tile" variable from the dataset after stitching.
    interactive:
        If True, open a window to visualize and tune image processing step-by-step.

    Returns
    -------
    Processed image(s): xr.Dataset | list[xr.Dataset]
        The processed image in xr.Dataset from executed pipeline as the outcome of the image pipeline workflow.

    Notes
    -----
    This function uses a pipeline architecture to process image data. The steps in the pipeline consist of:

    - 'identify_buttons' : Identifies buttons based on the provided `pinlist` or `shape` parameters and assigns valid markers.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'rotate' : Rotates the stitched image by the specified angle.
    - 'find_buttons' : Detects buttons based on edge detection and clustering.
    - 'filter_expression' : Filters buttons based on foreground-background contrast differences using a minimum contrast threshold.
    - 'filter_leaky' : Filters buttons that are determined to be leaky.
    - 'drop' : Optionally removes tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> processed_mini_chip = mini_chip(
    ...     data=my_image_data, channels=[0], pinlist="pinlist.csv", overlap=100, rotation=45
    ... )

    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, rotating the image by 45 degrees, using channel 0 for detection, and using the chamber layout from 'pinlist.csv'.
    """
    pipe = microfluidic_chip_pipe(
        shape=shape,
        pinlist=pinlist,
        blank=blank,
        overlap=overlap,
        rotation=rotation,
        row_dist=row_dist,
        col_dist=col_dist,
        chip_type=chip_type,
        min_button_diameter=min_button_diameter,
        max_button_diameter=max_button_diameter,
        chamber_diameter=chamber_diameter,
        top_chamber=top_chamber,
        left_chamber=left_chamber,
        low_edge_quantile=low_edge_quantile,
        high_edge_quantile=high_edge_quantile,
        num_iter=num_iter,
        min_roundness=min_roundness,
        cluster_penalty=cluster_penalty,
        roi_length=roi_length,
        progress_bar=progress_bar,
        search_timestep=search_timestep,
        search_channel=search_channel,
        squeeze=squeeze,
        roi_only=roi_only,
        drop_tiles=drop_tiles,
        interactive=interactive,
    )
    return pipe(data=data)


def microfluidic_chip_pipe(
    shape: tuple[int, int] = (8, 8),
    pinlist: str | None = None,
    blank: str | list[str] | None = None,
    overlap: int = 102,
    rotation: int = 0,
    row_dist: float = 375 / 1.61,
    col_dist: float = 400 / 1.61,
    chip_type: str | None = None,
    min_button_diameter: int = 8,
    max_button_diameter: int = 30,
    chamber_diameter: int = 60,
    top_chamber: int | None = None,
    left_chamber: int | None = None,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.2,
    cluster_penalty: float = 50,
    roi_length: int | None = None,
    progress_bar: bool = False,
    search_timestep: int | list[int] = 0,
    search_channel: str | list[str] | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
    interactive: bool = False,
) -> Pipeline:
    """
    Build a Pipeline object that detects buttons in images of microfluidic chips and standardizes the resulting data in an xarray.Dataset.

    Reference
    -------
    For detailed information on how to use this pipeline, refer to :func:`microfluidic_chip`.
    """
    if chip_type is not None:
        if chip_type == "minichip":
            row_dist, col_dist = 375 / 1.61, 400 / 1.61
        elif chip_type == "pc":
            row_dist, col_dist = 406 / 3.22, 750 / 3.22
        elif chip_type == "ps":
            row_dist, col_dist = 375 / 3.22, 655 / 3.22
        else:
            raise ValueError(
                f"Invalid chip type: {chip_type}. Must be one of ['pc', 'ps', 'minichip']"
            )

    pipe = Pipeline("read")
    pipe.add_pipe("identify_buttons", shape=shape, pinlist=pinlist, blank=blank)
    pipe.add_pipe("stitch", overlap=overlap)
    pipe.add_pipe("rotate", rotation=rotation)
    pipe.add_pipe(
        "find_buttons",
        row_dist=row_dist,
        col_dist=col_dist,
        min_button_diameter=min_button_diameter,
        max_button_diameter=max_button_diameter,
        chamber_diameter=chamber_diameter,
        top_chamber=top_chamber,
        left_chamber=left_chamber,
        low_edge_quantile=low_edge_quantile,
        high_edge_quantile=high_edge_quantile,
        num_iter=num_iter,
        min_roundness=min_roundness,
        cluster_penalty=cluster_penalty,
        roi_length=roi_length,
        progress_bar=progress_bar,
        search_timestep=search_timestep,
        search_channel=search_channel,
        interactive=interactive,
    )
    pipe.add_pipe("drop", squeeze=squeeze, roi_only=roi_only, drop_tiles=drop_tiles)

    return pipe


def mrbles(
    data: ArrayLike | str,
    spectra: str | PathLike | StringIO,
    codes: str | PathLike | StringIO,
    flatfield: float = 1.0,
    darkfield: float = 0.0,
    overlap: int = 102,
    min_bead_diameter: int = 10,
    max_bead_diameter: int = 50,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.3,
    roi_length: int | None = None,
    search_channel: str | list[str] | None = None,
    reference: str = "eu",
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
    interactive: bool = False,
) -> xr.Dataset | list[xr.Dataset]:
    """
    Find MRBLEs(Microspheres with Ratiometric Barcode Lanthanide Encoding Beaads) in images and standardize the resulting data in an xarray.Dataset.

    Parameters
    ----------
    data :
        The input image data to be processed. It can be one of the following:
        - A file path or glob to image data.
        - An `xarray.DataArray` or `xarray.Dataset` containing image data.
        - A sequence of file paths, `xarray.DataArray`, or `xarray.Dataset`.
    spectra :
        A path to a csv file that contains the reference spectrum of each lanthanide that will be used for decoding.
    codes :
        A path to a csv file that contains the name and lanthanide content of each code.
    flatfield :
        The flatfield correction factor or path to a flatfield correction image.
    darkfield :
        The darkfield correction factor or path to a darkfield correction image.
    overlap :
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
    min_bead_diameter, max_bead_diameter :
        The minimum/maximum diameter in pixels for detecting beads in the image.
    low_edge_quantile, high_edge_quantile :
        The lower/upper quantile for edge detection, tunes the sensitivity of how dim edges can be when detecting beads. Must be between 0 and 1.
    num_iter :
        The maximum number of iterations to perform the bead detection process using RANSAC. A higher value will take longer but will find beads more accurately.
    min_roundness :
        The minimum roundness value for detected beads. Beads that do not meet this roundness threshold are excluded. Must be between 0 and 1.
    roi_length :
        The length in pixels of the region of interest (ROI) around detected beadeads. If None, the ROI length is set to 2 * `max_bead_diameter`.
    search_channel :
        The channel or list of channels to use for bead detection. If `None`, all channels will be used.
    squeeze :
        If True, removes any dimensions of length 1.
    roi_only :
        If True, only returns the region of interest from the dataset.
    drop_tiles :
        If True, removes the "tile" variable from the dataset after stitching.
    interactive:
        If True, open a window to visualize and tune image processing step-by-step.
    reference :
        The reference material or standard used for spectral decoding. The default is "eu" (Europium), which is typically used in MRBLEs for comparison in spectral analysis.
    squeeze :
        If True, remove dimensions of size 1.
    roi_only :
        If True, only returns the region of interest (ROI) from the dataset, ignoring other parts of the image.
    drop_tiles :
        If True, removes the "tile" variable from the dataset after stitching.
    interactive:
        If True, open a window to visualize and tune image processing step-by-step.

    Returns
    -------
    Processed image(s): xr.Dataset | list[xr.Dataset]
        The processed images and ROI.

    Notes
    -----
    This function uses a pipeline architecture to process MRBLEs image data. The steps in the pipeline include:
    - 'flatfield_correct' : Applies flatfield and darkfield corrections to the image data.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'find_beads' : Detects beads based on specified diameter, roundness, and other parameters.
    - 'identify_mrbles' : Assign a code to each bead by matching their spectral signatures to the provided reference spectra.
    - 'drop' : Optionally removes tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> processed_mrbles = mrbles(
    ...     spectra=my_spectra, codes=my_codes, data=my_image_data, channels=[0], overlap=100
    ... )

    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, using channel 0 for analysis, and matching bead spectral signatures against `my_spectra` and `my_codes`.
    """
    pipe = mrbles_pipe(
        spectra=spectra,
        codes=codes,
        flatfield=flatfield,
        darkfield=darkfield,
        overlap=overlap,
        min_bead_diameter=min_bead_diameter,
        max_bead_diameter=max_bead_diameter,
        low_edge_quantile=low_edge_quantile,
        high_edge_quantile=high_edge_quantile,
        num_iter=num_iter,
        min_roundness=min_roundness,
        roi_length=roi_length,
        search_channel=search_channel,
        reference=reference,
        squeeze=squeeze,
        roi_only=roi_only,
        drop_tiles=drop_tiles,
        interactive=interactive,
    )
    return pipe(data=data)


def mrbles_pipe(
    spectra: str | PathLike | StringIO,
    codes: str | PathLike | StringIO,
    flatfield: float = 1.0,
    darkfield: float = 0.0,
    overlap: int = 102,
    min_bead_diameter: int = 10,
    max_bead_diameter: int = 50,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.3,
    roi_length: int | None = None,
    search_channel: str | list[str] | None = None,
    reference: str = "eu",
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
    interactive: bool = False,
) -> Pipeline:
    """
    Build a Pipeline object that can detect MRBLEs(Microspheres with Ratiometric Barcode Lanthanide Encoding Beaads) in images and standardize the resulting data in an xarray.Dataset.

    Reference
    -------
    This function builds the necessary pipeline for detecting beads in mrbles images.
    For detailed information on how to use this pipeline, refer to :func:`mrbles`.
    """
    pipe = Pipeline("read")
    pipe.add_pipe("flatfield_correct", flatfield=flatfield, darkfield=darkfield)
    pipe.add_pipe("stitch", overlap=overlap)
    pipe.add_pipe(
        "find_beads",
        min_bead_diameter=min_bead_diameter,
        max_bead_diameter=max_bead_diameter,
        low_edge_quantile=low_edge_quantile,
        high_edge_quantile=high_edge_quantile,
        num_iter=num_iter,
        min_roundness=min_roundness,
        roi_length=roi_length,
        search_channel=search_channel,
        interactive=interactive,
    )
    pipe.add_pipe("identify_mrbles", spectra=spectra, codes=codes, reference=reference)
    pipe.add_pipe("drop", squeeze=squeeze, roi_only=roi_only, drop_tiles=drop_tiles)

    return pipe


def beads(
    data: ArrayLike | str,
    flatfield: float = 1.0,
    darkfield: float = 0.0,
    overlap: int = 102,
    min_bead_diameter: int = 10,
    max_bead_diameter: int = 50,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.3,
    roi_length: int | None = None,
    search_channel: str | list[str] | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
    interactive: bool = False,
) -> xr.Dataset | list[xr.Dataset]:
    """
    Find beads in images and standardize the resulting data in an xarray.Dataset.

    Parameters
    ----------
    data :
        The input image data to be processed. It can be one of the following:
        - A file path or glob to image data.
        - An `xarray.DataArray` or `xarray.Dataset` containing image data.
        - A sequence of file paths, `xarray.DataArray`, or `xarray.Dataset`.
    flatfield :
        The flatfield correction factor or path to a flatfield correction image.
    darkfield :
        The darkfield correction factor or path to a darkfield correction image.
    overlap :
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
    min_bead_diameter, max_bead_diameter :
        The minimum/maximum diameter in pixels for detecting beads in the image.
    low_edge_quantile, high_edge_quantile :
        The lower/upper quantile for edge detection, tunes the sensitivity of how dim edges can be when detecting beads. Must be between 0 and 1.
    num_iter :
        The maximum number of iterations to perform the bead detection process using RANSAC. A higher value will take longer but will find beads more accurately.
    min_roundness :
        The minimum roundness value for detected beads. Beads that do not meet this roundness threshold are excluded. Must be between 0 and 1.
    roi_length :
        The length in pixels of the region of interest (ROI) around detected beadeads. If None, the ROI length is set to 2 * `max_bead_diameter`.
    search_channel :
        The channel or list of channels to use for bead detection. If `None`, all channels will be used.
    squeeze :
        If True, removes any dimensions of length 1.
    roi_only :
        If True, only returns the region of interest from the dataset.
    drop_tiles :
        If True, removes the "tile" variable from the dataset after stitching.
    interactive:
        If True, open a window to visualize and tune image processing step-by-step.
    squeeze :
        If True, remove dimensions of size 1.
    roi_only :
        If True, only returns the region of interest (ROI) from the dataset, ignoring other parts of the image.
    drop_tiles :
        If True, removes the "tile" variable from the dataset after stitching.
    interactive:
        If True, open a window to visualize and tune image processing step-by-step.

    Returns
    -------
    Processed image(s): xr.Dataset | list[xr.Dataset]
        The processed images and ROI.

    Notes
    -----
    This function uses a pipeline architecture to process image data. The steps in the pipeline include:

    - 'flatfield_correct' : Applies flatfield and darkfield corrections to the image data.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'find_beads' : Detects beads in the image based on specified diameters and other detection parameters.
    - 'drop' : Optionally removes tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> bead_image = beads(
    ...     data=my_image_data,
    ...     channels=[0, 1],
    ...     overlap=100,
    ...     min_bead_diameter=10,
    ...     max_bead_diameter=40,
    ... )

    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, using only channels 0 and 1, and detects beads with a diameter between 10 and 40 pixels.
    """
    pipe = beads_pipe(
        darkfield=darkfield,
        overlap=overlap,
        min_bead_diameter=min_bead_diameter,
        max_bead_diameter=max_bead_diameter,
        low_edge_quantile=low_edge_quantile,
        high_edge_quantile=high_edge_quantile,
        num_iter=num_iter,
        min_roundness=min_roundness,
        roi_length=roi_length,
        search_channel=search_channel,
        squeeze=squeeze,
        roi_only=roi_only,
        drop_tiles=drop_tiles,
        interactive=interactive,
    )
    return pipe(data=data)


def beads_pipe(
    flatfield: float = 1.0,
    darkfield: float = 0.0,
    overlap: int = 102,
    min_bead_diameter: int = 5,
    max_bead_diameter: int = 25,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.3,
    roi_length: int | None = None,
    search_channel: str | list[str] | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
    interactive: bool = False,
) -> Pipeline:
    """
    Build a Pipeline object that can detect beads in images and standardize the resulting data in an xarray.Dataset.

    Reference
    -------
    This function builds the necessary pipeline for detecting beads.
    For detailed information on how to use this pipeline, refer to :func:`beads`.
    """
    pipe = Pipeline("read")
    pipe.add_pipe("flatfield_correct", flatfield=flatfield, darkfield=darkfield)
    pipe.add_pipe("stitch", overlap=overlap)
    pipe.add_pipe(
        "find_beads",
        min_bead_diameter=min_bead_diameter,
        max_bead_diameter=max_bead_diameter,
        low_edge_quantile=low_edge_quantile,
        high_edge_quantile=high_edge_quantile,
        num_iter=num_iter,
        min_roundness=min_roundness,
        roi_length=roi_length,
        search_chanel=search_channel,
        interactive=interactive,
    )
    pipe.add_pipe("drop", squeeze=squeeze, roi_only=roi_only, drop_tiles=drop_tiles)

    return pipe


def image(
    data: ArrayLike | str,
    overlap: int = 102,
    rotation: float = 0,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> xr.Dataset | list[xr.Dataset]:
    """
    Read in images and standardize the resulting data in an xarray.Dataset.

    Parameters
    ----------
    data :
        The input image data to be processed. It can be one of the following:
        - A file path or glob to image data.
        - An `xarray.DataArray` or `xarray.Dataset` containing image data.
        - A sequence of file paths, `xarray.DataArray`, or `xarray.Dataset`.
    overlap :
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
    squeeze :
        If True, removes any dimensions of length 1.
    roi_only :
        If True, only returns the region of interest from the dataset.
    drop_tiles :
        If True, removes the "tile" variable from the dataset after stitching.
    Returns
    -------
    Processed image(s) :
        The processed images and ROI.

    Notes
    -----
    This function uses a pipeline architecture to process image data. The steps in the pipeline include:

    - 'stitch' : Stitch together image tiles based on the overlap parameter.
    - 'rotate' : Rotate the image by the specified angle.
    - 'drop' : Depending on the options for `squeeze`, `roi_only`, and `drop_tiles`, unnecessary or unused
      tiles are removed, and the data is optionally simplified.

    Examples
    --------
    >>> processed_image = image(
    ...     data=my_image_data, channels=[0, 1], overlap=100, rotation=45, squeeze=True
    ... )
    """
    pipe = image_pipe(
        overlap=overlap,
        rotation=rotation,
        squeeze=squeeze,
        roi_only=roi_only,
        drop_tiles=drop_tiles,
    )
    return pipe(data=data)


def image_pipe(
    overlap: int = 102,
    rotation: float = 0,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> Pipeline:
    """
    Build a Pipeline object that reads in an image and standardizes the resulting data in an xarray.Dataset.

    Reference
    -------
    This function builds the necessary pipeline for customizing an image-processing pipeline.
    For detailed information on how to use this pipeline, refer to :func:`image`.
    """
    pipe = Pipeline("read")
    pipe.add_pipe("stitch", overlap=overlap)
    pipe.add_pipe("rotate", rotation=rotation)
    pipe.add_pipe("drop", squeeze=squeeze, roi_only=roi_only, drop_tiles=drop_tiles)
    return pipe

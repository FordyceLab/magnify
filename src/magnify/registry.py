from __future__ import annotations
import functools
import inspect
from typing import Callable, Sequence
from os import PathLike
from io import StringIO

from numpy.typing import ArrayLike
import catalogue
from functools import wraps
import xarray as xr

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

def mini_chip(
    data: ArrayLike | str,
    times: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    debug: bool = False,
    shape: tuple[int, int] = (8, 8),
    pinlist: str | None = None,
    blank: str | list[str] | None = None,
    overlap: int = 102,
    rotation: int = 0,
    row_dist: float = 375 / 1.61,
    col_dist: float = 400 / 1.61,
    min_button_radius: int = 4,
    max_button_radius: int = 15,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.2,
    cluster_penalty: float = 50,
    roi_length: int = 61,
    progress_bar: bool = False,
    search_timestep: list[int] | None = None,
    search_channel: str | list[str] | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> xr.Dataset | list[xr.Dataset]:
    pipe = mini_chip_pipe(debug=debug, shape=shape, pinlist=pinlist, blank=blank, 
                          overlap=overlap, rotation=rotation, row_dist=row_dist, col_dist=col_dist,
                          min_button_radius=min_button_radius, max_button_radius=max_button_radius, 
                          low_edge_quantile=low_edge_quantile, high_edge_quantile=high_edge_quantile, 
                          num_iter=num_iter, min_roundness=min_roundness, cluster_penalty=cluster_penalty,
                          roi_length=roi_length, progress_bar=progress_bar, search_timestep=search_timestep,
                          search_channel= search_channel, squeeze=squeeze, roi_only=roi_only, drop_tiles=drop_tiles)
    return pipe(data=data, times=times, channels=channels)

"""
    Processes image data for a miniaturized chip with button detection, stitching, rotation, and multiple filtering steps.

    Parameters
    ----------
    data : 
        The input image data to be processed. It can be one of the following:
        - A file path (string) to image data.
        - An `xarray.DataArray` or `xarray.Dataset` containing image data.
        - A sequence (list or tuple) of file paths, `xarray.DataArray`, or `xarray.Dataset`.
        The function can handle multiple formats and will standardize them into an `xarray.Dataset` for processing.
    times : 
        A list or sequence of time points to extract from the dataset. If specified, only the data
        corresponding to these time points will be processed. If `None`, all available time points
        in the data will be used.
    channels :
        A list or sequence of channel names to extract from the dataset. If specified, only the data
        from the specified channels will be processed. If `None`, all available channels will be used.
    debug :
        If True, set the logger to debug mode for error logging.
    shape : tuple[int, int], optional
        The shape of the button array, specifying the number of rows and columns in the image grid.
        If provided, it will be used to assign default button tags. The default is `(8, 8)` for mini chips.
    pinlist :
        A file path to a CSV file that describes the pin layout on the chip. The CSV file must include
        a column called `Indices` that contains row and column pairs in the format `(row, col)`, and a
        `MutantID` column that contains the names of the buttons. This is used to map the buttons to physical
        locations on the chip. Either `pinlist` or `shape` must be provided.
    blank : 
        Values representing "blank" or non-expressed buttons in the pinlist, which will be replaced
        with empty strings in the dataset. Defaults to `["", "blank", "BLANK"]`.
    overlap :
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
        This overlap value is subtracted from both the vertical (y) and horizontal (x) dimensions
        of the tiles to remove redundant or overlapping areas between adjacent tiles.
    rotation : 
        The degree of rotation to apply to the stitched image.
    row_dist / col_dist: 
        The distance between rows/columns of buttons in pixels. This is converted to pixels based on the pixel-to-micron conversion rate, assuming 1.61 pixels.
    min_button_radius / max_button_radius : 
        The minimum/maximum radius (in pixels) for detecting buttons in the image.
    low_edge_quantile / high_edge_quantile : 
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons.
    num_iter : 
        The maximum number of iterations to perform during the button detection process, controlling the depth of search.
    min_roundness : 
        The minimum roundness value for detected buttons. Buttons that do not meet this roundness threshold are excluded.
    cluster_penalty : 
        The penalty factor for clustering during button detection. This parameter helps control how strongly nearby buttons are grouped together.
    roi_length : 
        The length (in pixels) of the region of interest (ROI) around detected buttons.
    progress_bar : 
        If True, display a progress bar during processing to track the progress of the pipeline.
    search_timestep : 
        A list of time steps to focus on for button detection. If provided, only the specified time steps are used. If `None`, all time steps are used.
    search_channel : 
        The channel or list of channels to use for button detection and expression filtering. If `None`, all channels will be used. Channels are used to compute foreground and background values for contrast-based filtering.
    squeeze : 
        If True, removes singleton dimensions from the dataset(dimensions of size 1).
    roi_only : 
        If True, only returns the region of interest(roi) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    Processed image(s): xr.Dataset | list[xr.Dataset]
        The processed image in xr.Dataset from executed pipeline as the outcome of the image pipeline workflow.

    Notes
    -----
    This function uses a pipeline architecture to process image data. The steps in the pipeline include:
    
    - 'identify_buttons' : Identifies buttons based on the provided `pinlist` or `shape` parameters and assigns valid markers.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'rotate' : Rotates the stitched image by the specified angle.
    - 'find_buttons' : Detects buttons based on edge detection and clustering.
    - 'filter_expression' : Filters buttons based on foreground-background contrast differences using a minimum contrast threshold.
    - 'filter_nonround' : Filters out buttons that do not meet the roundness criteria.
    - 'filter_leaky' : Removes buttons that are determined to be leaky or poorly segmented.
    - 'drop' : Optionally removes unnecessary tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> processed_mini_chip = mini_chip(data=my_image_data, channels=[0], pinlist='pinlist.csv', overlap=100, rotation=45)
    
    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, rotating the image by 45 degrees, using channel 0 for detection, and utilizing a pin layout from 'pinlist.csv'.
    """


def mini_chip_pipe(
    debug: bool = False,
    shape: tuple[int, int] = (8, 8),
    pinlist: str | None = None,
    blank: str | list[str] | None = None,
    overlap: int = 102,
    rotation: int = 0,
    row_dist: float = 375 / 1.61,
    col_dist: float = 400 / 1.61,
    min_button_radius: int = 4,
    max_button_radius: int = 15,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.2,
    cluster_penalty: float = 50,
    roi_length: int = 61,
    progress_bar: bool = False,
    search_timestep: list[int] | None = None,
    search_channel: str | list[str] | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> Pipeline:
    """
    Build the pipe to processes image data for a miniaturized chip with button detection, stitching, rotation, and multiple filtering steps.

    Parameters
    ----------
    debug :
        If True, set the logger to debug mode for error logging.
    shape : tuple[int, int], optional
        The shape of the button array, specifying the number of rows and columns in the image grid.
        If provided, it will be used to assign default button tags. The default is `(8, 8)` for mini chips.
    pinlist :
        A file path to a CSV file that describes the pin layout on the chip. The CSV file must include
        a column called `Indices` that contains row and column pairs in the format `(row, col)`, and a
        `MutantID` column that contains the names of the buttons. This is used to map the buttons to physical
        locations on the chip. Either `pinlist` or `shape` must be provided.
    blank : 
        Values representing "blank" or non-expressed buttons in the pinlist, which will be replaced
        with empty strings in the dataset. Defaults to `["", "blank", "BLANK"]`.
    overlap :
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
        This overlap value is subtracted from both the vertical (y) and horizontal (x) dimensions
        of the tiles to remove redundant or overlapping areas between adjacent tiles.
    rotation : 
        The degree of rotation to apply to the stitched image.
    row_dist / col_dist: 
        The distance between rows/columns of buttons in pixels. This is converted to pixels based on the pixel-to-micron conversion rate, assuming 1.61 pixels.
    min_button_radius / max_button_radius : 
        The minimum/maximum radius (in pixels) for detecting buttons in the image.
    low_edge_quantile / high_edge_quantile : 
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons.
    num_iter : 
        The maximum number of iterations to perform during the button detection process, controlling the depth of search.
    min_roundness : 
        The minimum roundness value for detected buttons. Buttons that do not meet this roundness threshold are excluded.
    cluster_penalty : 
        The penalty factor for clustering during button detection. This parameter helps control how strongly nearby buttons are grouped together.
    roi_length : 
        The length (in pixels) of the region of interest (ROI) around detected buttons.
    progress_bar : 
        If True, display a progress bar during processing to track the progress of the pipeline.
    search_timestep : 
        A list of time steps to focus on for button detection. If provided, only the specified time steps are used. If `None`, all time steps are used.
    search_channel : 
        The channel or list of channels to use for button detection and expression filtering. If `None`, all channels will be used. Channels are used to compute foreground and background values for contrast-based filtering.
    squeeze : 
        If True, removes singleton dimensions from the dataset(dimensions of size 1).
    roi_only : 
        If True, only returns the region of interest(roi) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    pipe : Pipeline
        The constructed pipeline object, either configured and returned if `return_pipe` is True, or executed as part of the image processing workflow.

    Notes
    -----
    This function constructs a pipeline architecture to process image data. The steps in the pipeline include:
    
    - 'identify_buttons' : Identifies buttons based on the provided `pinlist` or `shape` parameters and assigns valid markers.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'rotate' : Rotates the stitched image by the specified angle.
    - 'find_buttons' : Detects buttons based on edge detection and clustering.
    - 'filter_expression' : Filters buttons based on foreground-background contrast differences using a minimum contrast threshold.
    - 'filter_nonround' : Filters out buttons that do not meet the roundness criteria.
    - 'filter_leaky' : Removes buttons that are determined to be leaky or poorly segmented.
    - 'drop' : Optionally removes unnecessary tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> mini_chip_pipe = mini_chip(channels=[0], pinlist='pinlist.csv', overlap=100, rotation=45)
    >>> mini_chip = mini_chip_pipe(data=my_image_data)
    
    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, rotating the image by 45 degrees, using channel 0 for detection, and utilizing a pin layout from 'pinlist.csv'.
    """
    # Button centers are apart 375um vertically and 655um horizontally.
    # Assuming a 4x objective and 1x1 binning each pixel is 1.61um.
    config = {key: value for key, value in locals().items()}  # Construct the configuration dict

    pipe = Pipeline("read", config=config)
    pipe.add_pipe("identify_buttons")
    pipe.add_pipe("stitch")
    pipe.add_pipe("rotate")
    pipe.add_pipe("find_buttons")
    pipe.add_pipe("filter_expression")
    pipe.add_pipe("filter_nonround")
    pipe.add_pipe("filter_leaky")
    pipe.add_pipe("drop")

    return pipe

def ps_chip(
    data: ArrayLike | str,
    times: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    debug: bool = False,
    shape: tuple[int, int] | None = None,
    pinlist: str | None = None,
    blank: str | list[str] | None = None,
    overlap: int = 102,
    rotation: int = 0,
    row_dist: float = 375 / 3.22,
    col_dist: float = 655 / 3.22,
    min_button_radius: int = 4,
    max_button_radius: int = 15,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.2,
    cluster_penalty: float = 50,
    roi_length: int = 61,
    progress_bar: bool = False,
    search_timestep: list[int] | None = None,
    search_channel: str | list[str] | None = None,
    min_contrast: int | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> xr.Dataset | list[xr.Dataset]:
    pipe = ps_chip_pipe(debug=debug, shape=shape, pinlist=pinlist, blank=blank, 
                          overlap=overlap, rotation=rotation, row_dist=row_dist, col_dist=col_dist,
                          min_button_radius=min_button_radius, max_button_radius=max_button_radius, 
                          low_edge_quantile=low_edge_quantile, high_edge_quantile=high_edge_quantile, 
                          num_iter=num_iter, min_roundness=min_roundness, cluster_penalty=cluster_penalty,
                          roi_length=roi_length, progress_bar=progress_bar, search_timestep=search_timestep,
                          search_channel= search_channel, min_contrast=min_contrast,squeeze=squeeze, 
                          roi_only=roi_only, drop_tiles=drop_tiles)
    return pipe(data=data, times=times, channels=channels)
"""
    Processes image data to detect buttons on a PS (Polystyrene) chip, applying stitching, rotation, and multiple filtering steps.

    Parameters
    ----------
    data : 
        The input image data to be processed. It can be one of the following:
        - A file path (string) to image data.
        - An `xarray.DataArray` or `xarray.Dataset` containing image data.
        - A sequence (list or tuple) of file paths, `xarray.DataArray`, or `xarray.Dataset`.
        The function can handle multiple formats and will standardize them into an `xarray.Dataset` for processing.
    times : 
        A list or sequence of time points to extract from the dataset. If specified, only the data
        corresponding to these time points will be processed. If `None`, all available time points
        in the data will be used.
    channels : 
        A list or sequence of channel names to extract from the dataset. If specified, only the data
        from the specified channels will be processed. If `None`, all available channels will be used.
    shape : tuple[int, int], optional
        The shape of the button array, specifying the number of rows and columns in the image grid.
        If provided, it will be used to assign default button tags. Either `shape` or `pinlist` must be provided.
    debug : 
        If True, set the logger to debug mode for error logging.
    pinlist : 
        A file path to a CSV file that describes the pin layout on the chip. The CSV file must include
        a column called `Indices` that contains row and column pairs in the format `(row, col)`, and a
        `MutantID` column that contains the names of the buttons. This is used to map the buttons to physical
        locations on the chip. Either `pinlist` or `shape` must be provided.
    blank : 
        Values representing "blank" or non-expressed buttons in the pinlist, which will be replaced
        with empty strings in the dataset. Defaults to `["", "blank", "BLANK"]`.
    overlap : 
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
        This overlap value is subtracted from both the vertical (y) and horizontal (x) dimensions
        of the tiles to remove redundant or overlapping areas between adjacent tiles.
    rotation :
        The degree of rotation to apply to the stitched image.
    row_dist / col_dist: 
        The distance between rows/columns of buttons in pixels. This is converted to pixels based on the pixel-to-micron conversion rate, assuming 1.61 pixels.
    min_button_radius / max_button_radius : 
        The minimum/maximum radius (in pixels) for detecting buttons in the image.
    low_edge_quantile / high_edge_quantile : 
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons.
    num_iter : 
        The maximum number of iterations to perform during the button detection process, controlling the depth of search.
    min_roundness : 
        The minimum roundness value required for buttons to be considered valid.
    cluster_penalty : 
        The penalty factor for clustering during button detection. This parameter helps control how strongly nearby buttons are grouped together.
    roi_length : 
        The length (in pixels) of the region of interest (ROI) around detected buttons.
    progress_bar : 
        If True, display a progress bar during processing to track the progress of the pipeline.
    search_timestep : 
        A list of time steps to focus on for button detection. If provided, only the specified time steps are used. If `None`, all time steps are used.
    search_channel : 
        The channel or list of channels to use for button detection and expression filtering. If `None`, 
        all channels will be used. Channels are used to compute foreground and background values for 
        contrast-based filtering.
    min_contrast : 
        The minimum contrast threshold for button detection and expression filtering. This value is used 
        to determine the intensity difference between the foreground and background of buttons. If `None`, 
        the contrast threshold is dynamically determined based on the standard deviation of background differences.
    squeeze : 
        If True, removes singleton dimensions from the dataset(dimensions of size 1).
    roi_only : 
        If True, only returns the region of interest(roi) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    Processed image(s): xr.Dataset | list[xr.Dataset]
        The processed image in xr.Dataset from executed pipeline as the outcome of the image pipeline workflow.

    Notes
    -----
    This function uses a pipeline architecture to process image data. The steps in the pipeline include:
    
    - 'identify_buttons' : Identifies buttons based on the provided parameters, such as radii, contrast, and roundness.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'rotate' : Rotates the stitched image by the specified angle.
    - 'find_buttons' : Detects buttons based on edge detection and clustering.
    - 'filter_expression' : Filters buttons based on predefined expression rules.
    - 'filter_nonround' : Filters out buttons that do not meet the roundness criteria.
    - 'filter_leaky' : Removes buttons that are determined to be leaky or poorly segmented.
    - 'drop' : Optionally removes unnecessary tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> processed_chip = ps_chip(data=my_image_data, channels=[0], pinlist='pinlist.csv', overlap=100, rotation=45)
    
    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, rotating the image by 45 degrees, using channel 0 for detection, and utilizing a pin layout from 'pinlist.csv'.
    """


def ps_chip_pipe(
    debug: bool = False,
    shape: tuple[int, int] | None = None,
    pinlist: str | None = None,
    blank: str | list[str] | None = None,
    overlap: int = 102,
    rotation: int = 0,
    row_dist: float = 375 / 3.22,
    col_dist: float = 655 / 3.22,
    min_button_radius: int = 4,
    max_button_radius: int = 15,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.2,
    cluster_penalty: float = 50,
    roi_length: int = 61,
    progress_bar: bool = False,
    search_timestep: list[int] | None = None,
    search_channel: str | list[str] | None = None,
    min_contrast: int | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> Pipeline:
    """
    Build the pipe to processes image data to detect buttons on a PS (Polystyrene) chip, applying stitching, rotation, and multiple filtering steps.

    Parameters
    ----------
    shape : tuple[int, int], optional
        The shape of the button array, specifying the number of rows and columns in the image grid.
        If provided, it will be used to assign default button tags. Either `shape` or `pinlist` must be provided.
    debug : 
        If True, set the logger to debug mode for error logging.
    pinlist : 
        A file path to a CSV file that describes the pin layout on the chip. The CSV file must include
        a column called `Indices` that contains row and column pairs in the format `(row, col)`, and a
        `MutantID` column that contains the names of the buttons. This is used to map the buttons to physical
        locations on the chip. Either `pinlist` or `shape` must be provided.
    blank : 
        Values representing "blank" or non-expressed buttons in the pinlist, which will be replaced
        with empty strings in the dataset. Defaults to `["", "blank", "BLANK"]`.
    overlap : 
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
        This overlap value is subtracted from both the vertical (y) and horizontal (x) dimensions
        of the tiles to remove redundant or overlapping areas between adjacent tiles.
    rotation :
        The degree of rotation to apply to the stitched image.
    row_dist : 
        The distance between rows of buttons in micrometers (um). This is converted to pixels based on the pixel-to-micron conversion rate.
    col_dist : 
        The distance between columns of buttons in micrometers (um). This is converted to pixels based on the pixel-to-micron conversion rate.
    min_button_radius / max_button_radius : 
        The minimum/maximum radius (in pixels) for detecting buttons in the image.
    low_edge_quantile / high_edge_quantile: 
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons.
    num_iter : 
        The maximum number of iterations to perform during the button detection process, controlling the depth of search.
    min_roundness : 
        The minimum roundness value required for buttons to be considered valid.
    cluster_penalty : 
        The penalty factor for clustering during button detection. This parameter helps control how strongly nearby buttons are grouped together.
    roi_length : 
        The length (in pixels) of the region of interest (ROI) around detected buttons.
    progress_bar : 
        If True, display a progress bar during processing to track the progress of the pipeline.
    search_timestep : 
        A list of time steps to focus on for button detection. If provided, only the specified time steps are used. If `None`, all time steps are used.
    search_channel : 
        The channel or list of channels to use for button detection and expression filtering. If `None`, 
        all channels will be used. Channels are used to compute foreground and background values for 
        contrast-based filtering.
    min_contrast : 
        The minimum contrast threshold for button detection and expression filtering. This value is used 
        to determine the intensity difference between the foreground and background of buttons. If `None`, 
        the contrast threshold is dynamically determined based on the standard deviation of background differences.
    squeeze : 
        If True, removes singleton dimensions from the dataset(dimensions of size 1).
    roi_only : 
        If True, only returns the region of interest(roi) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    pipe : Pipeline
        The constructed pipeline object, either configured and returned if `return_pipe` is True, or executed as part of the image processing workflow.

    Notes
    -----
    This function constructs a pipeline architecture to process image data. The steps in the pipeline include:
    
    - 'identify_buttons' : Identifies buttons based on the provided parameters, such as radii, contrast, and roundness.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'rotate' : Rotates the stitched image by the specified angle.
    - 'find_buttons' : Detects buttons based on edge detection and clustering.
    - 'filter_expression' : Filters buttons based on predefined expression rules.
    - 'filter_nonround' : Filters out buttons that do not meet the roundness criteria.
    - 'filter_leaky' : Removes buttons that are determined to be leaky or poorly segmented.
    - 'drop' : Optionally removes unnecessary tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> ps_chip_pipeline = ps_chip(channels=[0], pinlist='pinlist.csv', overlap=100, rotation=45)
    >>> ps_chip = ps_chip_pipeline(data=my_image_data)
    
    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, rotating the image by 45 degrees, using channel 0 for detection, and utilizing a pin layout from 'pinlist.csv'.
    """
    # Button centers are apart 375um vertically and 655um horizontally.
    # Assuming a 4x objective and 2x2 binning each pixel is 3.22um.
    config = {key: value for key, value in locals().items()}

    pipe = Pipeline("read", config=config)
    pipe.add_pipe("identify_buttons")
    pipe.add_pipe("stitch")
    pipe.add_pipe("rotate")
    pipe.add_pipe("find_buttons")
    pipe.add_pipe("filter_expression")
    pipe.add_pipe("filter_nonround")
    pipe.add_pipe("filter_leaky")
    pipe.add_pipe("drop")

    return pipe

def pc_chip(
    data: ArrayLike | str,
    times: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    debug: bool = False,
    shape: tuple[int, int] | None = None,
    pinlist: str | None = None,
    blank: str | list[str] | None = None,
    overlap: int = 102,
    rotation: int | float = 0,
    row_dist: float = 406 / 3.22,
    col_dist: float = 750 / 3.22,
    min_button_radius: int = 4,
    max_button_radius: int = 15,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.2,
    cluster_penalty: float = 50,
    roi_length: int = 61,
    progress_bar: bool = False,
    search_timestep: list[int] | None = None,
    search_channel: str | list[str] | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> xr.Dataset | list[xr.Dataset]:
    pipe = pc_chip_pipe(debug=debug, shape=shape, pinlist=pinlist, blank=blank, 
                          overlap=overlap, rotation=rotation, row_dist=row_dist, col_dist=col_dist,
                          min_button_radius=min_button_radius, max_button_radius=max_button_radius, 
                          low_edge_quantile=low_edge_quantile, high_edge_quantile=high_edge_quantile, 
                          num_iter=num_iter, min_roundness=min_roundness, cluster_penalty=cluster_penalty,
                          roi_length=roi_length, progress_bar=progress_bar, search_timestep=search_timestep,
                          search_channel= search_channel,squeeze=squeeze, roi_only=roi_only, drop_tiles=drop_tiles)
    return pipe(data=data, times=times, channels=channels)

"""
    Processes image data to detect buttons on a PC (Polycarbonate) chip, applying stitching, rotation, horizontal flips, and multiple filtering steps.

    Parameters
    ----------
    data : 
        The input image data to be processed. It can be one of the following:
        - A file path (string) to image data.
        - An `xarray.DataArray` or `xarray.Dataset` containing image data.
        - A sequence (list or tuple) of file paths, `xarray.DataArray`, or `xarray.Dataset`.
        The function can handle multiple formats and will standardize them into an `xarray.Dataset` for processing.
    times : 
        A list or sequence of time points to extract from the dataset. If specified, only the data
        corresponding to these time points will be processed. If `None`, all available time points
        in the data will be used.
    channels : 
        A list or sequence of channel names to extract from the dataset. If specified, only the data
        from the specified channels will be processed. If `None`, all available channels will be used.
    debug : 
        If True, set the logger to debug mode for error logging.
    shape : 
        The shape of the button array, specifying the number of rows and columns in the image grid.
        If provided, it will be used to assign default button tags. Either `shape` or `pinlist` must be provided.
    pinlist : 
        A file path to a CSV file that describes the pin layout on the chip. The CSV file must include
        a column called `Indices` that contains row and column pairs in the format `(row, col)`, and a
        `MutantID` column that contains the names of the buttons. This is used to map the buttons to physical
        locations on the chip. Either `pinlist` or `shape` must be provided.
    blank : 
        Values representing "blank" or non-expressed buttons in the pinlist, which will be replaced
        with empty strings in the dataset. Defaults to `["", "blank", "BLANK"]`.
    overlap : 
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
        This overlap value is subtracted from both the vertical (y) and horizontal (x) dimensions
        of the tiles to remove redundant or overlapping areas between adjacent tiles.
    rotation : 
        The degree of rotation to apply to the stitched image.
    row_dist : 
        The distance between rows of buttons in micrometers (um). This is converted to pixels based on the pixel-to-micron conversion rate.
    col_dist : 
        The distance between columns of buttons in micrometers (um). This is converted to pixels based on the pixel-to-micron conversion rate.
    min_button_radius / max_button_radius : 
        The minimum/maximum radius (in pixels) for detecting buttons in the image.
    low_edge_quantile / high_edge_quantile: 
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons.
    num_iter : 
        The maximum number of iterations to perform during the button detection process, controlling the depth of search.
    min_roundness : 
        The minimum roundness value for detected buttons. Buttons that do not meet this roundness threshold are excluded.
    cluster_penalty : 
        The penalty factor for clustering during button detection. This parameter helps control how strongly nearby buttons are grouped together.
    roi_length : 
        The length (in pixels) of the region of interest (ROI) around detected buttons.
    progress_bar : 
        If True, display a progress bar during processing to track the progress of the pipeline.
    search_timestep : 
        A list of time steps to focus on for button detection. If provided, only the specified time steps are used. If `None`, all time steps are used.
    search_channel :
        The channel or list of channels to use for button detection and expression filtering. If `None`, all channels will be used. Channels are used to compute foreground and background values for contrast-based filtering.
    squeeze : 
        If True, removes singleton dimensions from the dataset (dimensions of size 1), simplifying the data structure.
    roi_only : 
        If True, only returns the region of interest (ROI) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    Processed image(s): xr.Dataset | list[xr.Dataset]
        The processed image in xr.Dataset from executed pipeline as the outcome of the image pipeline workflow.

    Notes
    -----
    This function uses a pipeline architecture to process image data. The steps in the pipeline include:
    
    - 'identify_buttons' : Identifies buttons based on the provided `pinlist` or `shape` parameters and assigns valid markers.
    - 'horizontal_flip' : Flips the image horizontally before and after stitching to align the chip layout.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'rotate' : Rotates the stitched image by the specified angle.
    - 'find_buttons' : Detects buttons based on edge detection and clustering.
    - 'filter_expression' : Filters buttons based on foreground-background contrast differences using a minimum contrast threshold.
    - 'filter_nonround' : Filters out buttons that do not meet the roundness criteria.
    - 'filter_leaky' : Removes buttons that are determined to be leaky or poorly segmented.
    - 'drop' : Optionally removes unnecessary tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> processed_chip = pc_chip(data=my_image_data, channels=[0], pinlist='pinlist.csv', overlap=100, rotation=45)
    
    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, rotating the image by 45 degrees, using channel 0 for detection, and utilizing a pin layout from 'pinlist.csv'.
    """

def pc_chip_pipe(
    debug: bool = False,
    shape: tuple[int, int] | None = None,
    pinlist: str | None = None,
    blank: str | list[str] | None = None,
    overlap: int = 102,
    rotation: int | float = 0,
    row_dist: float = 406 / 3.22,
    col_dist: float = 750 / 3.22,
    min_button_radius: int = 4,
    max_button_radius: int = 15,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.2,
    cluster_penalty: float = 50,
    roi_length: int = 61,
    progress_bar: bool = False,
    search_timestep: list[int] | None = None,
    search_channel: str | list[str] | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> Pipeline:
    """
    Build the pipe to processes image data to detect buttons on a PC (Polycarbonate) chip, applying stitching, rotation, horizontal flips, and multiple filtering steps.

    Parameters
    ----------
    debug : 
        If True, set the logger to debug mode for error logging.
    shape : 
        The shape of the button array, specifying the number of rows and columns in the image grid.
        If provided, it will be used to assign default button tags. Either `shape` or `pinlist` must be provided.
    pinlist : 
        A file path to a CSV file that describes the pin layout on the chip. The CSV file must include
        a column called `Indices` that contains row and column pairs in the format `(row, col)`, and a
        `MutantID` column that contains the names of the buttons. This is used to map the buttons to physical
        locations on the chip. Either `pinlist` or `shape` must be provided.
    blank : 
        Values representing "blank" or non-expressed buttons in the pinlist, which will be replaced
        with empty strings in the dataset. Defaults to `["", "blank", "BLANK"]`.
    overlap : 
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
        This overlap value is subtracted from both the vertical (y) and horizontal (x) dimensions
        of the tiles to remove redundant or overlapping areas between adjacent tiles.
    rotation : 
        The degree of rotation to apply to the stitched image.
    row_dist : 
        The distance between rows of buttons in micrometers (um). This is converted to pixels based on the pixel-to-micron conversion rate.
    col_dist : 
        The distance between columns of buttons in micrometers (um). This is converted to pixels based on the pixel-to-micron conversion rate.
    min_button_radius / max_button_radius : 
        The minimum/maximum radius (in pixels) for detecting buttons in the image.
    low_edge_quantile / high_edge_quantile: 
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons.
    num_iter : 
        The maximum number of iterations to perform during the button detection process, controlling the depth of search.
    min_roundness : 
        The minimum roundness value for detected buttons. Buttons that do not meet this roundness threshold are excluded.
    cluster_penalty : 
        The penalty factor for clustering during button detection. This parameter helps control how strongly nearby buttons are grouped together.
    roi_length : 
        The length (in pixels) of the region of interest(roi)) around detected buttons.
    progress_bar : 
        If True, display a progress bar during processing to track the progress of the pipeline.
    search_timestep : 
        A list of time steps to focus on for button detection. If provided, only the specified time steps are used. If `None`, all time steps are used.
    search_channel :
        The channel or list of channels to use for button detection and expression filtering. If `None`, all channels will be used. Channels are used to compute foreground and background values for contrast-based filtering.
    squeeze : 
        If True, removes singleton dimensions from the dataset (dimensions of size 1), simplifying the data structure.
    roi_only : 
        If True, only returns the region of interest (ROI) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    pipe : Pipeline
        The constructed pipeline object, either configured and returned if `return_pipe` is True, or executed as part of the image processing workflow.

    Notes
    -----
    This function constructs a pipeline architecture to process image data. The steps in the pipeline include:
    
    - 'identify_buttons' : Identifies buttons based on the provided `pinlist` or `shape` parameters and assigns valid markers.
    - 'horizontal_flip' : Flips the image horizontally before and after stitching to align the chip layout.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'rotate' : Rotates the stitched image by the specified angle.
    - 'find_buttons' : Detects buttons based on edge detection and clustering.
    - 'filter_expression' : Filters buttons based on foreground-background contrast differences using a minimum contrast threshold.
    - 'filter_nonround' : Filters out buttons that do not meet the roundness criteria.
    - 'filter_leaky' : Removes buttons that are determined to be leaky or poorly segmented.
    - 'drop' : Optionally removes unnecessary tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> processed_chip_pipe = pc_chip(channels=[0], pinlist='pinlist.csv', overlap=100, rotation=45)
    >>> process_chip = processed_chip_pipe(data=my_image_data)
    
    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, rotating the image by 45 degrees, using channel 0 for detection, and utilizing a pin layout from 'pinlist.csv'.
    """
    # Button centers are apart 412um vertically and 760um horizontally.
    # Assuming a 4x objective and 2x2 binning each pixel is 3.22um.
    config = {key: value for key, value in locals().items()}

    pipe = Pipeline("read", config=config)
    pipe.add_pipe("identify_buttons")
    pipe.add_pipe("horizontal_flip")
    pipe.add_pipe("stitch")
    pipe.add_pipe("horizontal_flip")
    pipe.add_pipe("rotate")
    pipe.add_pipe("find_buttons")
    pipe.add_pipe("filter_expression")
    pipe.add_pipe("filter_nonround")
    pipe.add_pipe("filter_leaky")
    pipe.add_pipe("drop")

    return pipe

def mrbles(
    spectra: str | PathLike | StringIO,
    codes: str | PathLike | StringIO,
    data: ArrayLike | str,
    times: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    flatfield: float = 1.0,
    darkfield: float = 0.0,
    overlap: int = 102,
    min_bead_radius: int = 5,
    max_bead_radius: int = 25,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.3,
    roi_length: int = 61,
    search_channel: str | list[str] | None = None,
    reference: str = "eu",
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> xr.Dataset | list[xr.Dataset]:
    pipe = mrbles_pipe(spectra=spectra, codes=codes, flatfield=flatfield, darkfield=darkfield, overlap=overlap,
                       min_bead_radius=min_bead_radius, max_bead_radius=max_bead_radius, low_edge_quantile=low_edge_quantile,
                       high_edge_quantile=high_edge_quantile, num_iter=num_iter, min_roundness=min_roundness,
                       roi_length=roi_length, search_channel=search_channel, reference=reference, squeeze=squeeze,
                       roi_only=roi_only, drop_tiles=drop_tiles)
    return pipe(data=data, times=times, channels=channels)

"""
    Processes image data for MRBLEs (Microsphere-based Rolling Circle Amplification Beads), applying flatfield correction, bead detection, and spectral decoding.

    Parameters
    ----------
    spectra : 
        The reference spectra data for the MRBLEs. This is used to identify the spectral signatures of the beads.
    codes : 
        The codes corresponding to the reference spectra. These codes are used to map specific spectral signatures to particular bead identities.
    data : 
        The input image data to be processed. It can be one of the following:
        - A file path (string) to image data.
        - An `xarray.DataArray` or `xarray.Dataset` containing image data.
        - A sequence (list or tuple) of file paths, `xarray.DataArray`, or `xarray.Dataset`.
        The function can handle multiple formats and will standardize them into an `xarray.Dataset` for processing.
    times : 
        A list or sequence of time points to extract from the dataset. If specified, only the data
        corresponding to these time points will be processed. If `None`, all available time points
        in the data will be used.
    channels : 
        A list or sequence of channel names to extract from the dataset. If specified, only the data
        from the specified channels will be processed. If `None`, all available channels will be used.
    flatfield : 
        The flatfield correction factor or path to a flatfield correction image. If a file path is provided, 
        the image will be loaded from the specified file (e.g., a TIFF or Zarr file). Flatfield correction 
        is used to account for uneven illumination across the image. If set to a numeric value (e.g., 1.0), 
        no flatfield correction will be applied.
    darkfield : 
        The darkfield correction factor or path to a darkfield correction image. If a file path is provided, 
        the image will be loaded from the specified file (e.g., a TIFF or Zarr file). Darkfield correction 
        is used to account for background noise in the image. If set to a numeric value (e.g., 0.0), no 
        darkfield correction will be applied.
    overlap : 
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
        This overlap value is subtracted from both the vertical (y) and horizontal (x) dimensions
        of the tiles to remove redundant or overlapping areas between adjacent tiles.
    min_bead_radius : 
        The minimum radius (in pixels) for detecting beads in the image.
    min_button_radius / max_button_radius : 
        The minimum/maximum radius (in pixels) for detecting buttons in the image.
    low_edge_quantile / high_edge_quantile: 
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons.
    num_iter : 
        The maximum number of iterations to perform during the bead detection process, controlling the depth of search.
    min_roundness : 
        The minimum roundness value for detected beads. Beads that do not meet this roundness threshold are excluded.
    roi_length : 
        The length (in pixels) of the region of interest (ROI) around detected beads.
    search_channel : 
        The channel or list of channels to use for bead detection and analysis. If `None`, all channels are used.
    reference : 
        The reference material or standard used for spectral decoding. The default is "eu" (Europium), which is typically used in MRBLEs for comparison in spectral analysis.
    squeeze :
        If True, removes singleton dimensions from the dataset (dimensions of size 1), simplifying the data structure.
    roi_only : 
        If True, only returns the region of interest (ROI) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    Processed image(s): xr.Dataset | list[xr.Dataset]
        The processed image in xr.Dataset from executed pipeline as the outcome of the image pipeline workflow.

    Notes
    -----
    This function uses a pipeline architecture to process MRBLEs image data. The steps in the pipeline include:
    
    - 'flatfield_correct' : Applies flatfield and darkfield corrections to the image data.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'find_beads' : Detects beads based on specified radii, roundness, and other parameters.
    - 'identify_mrbles' : Identifies MRBLEs beads by matching their spectral signatures to the provided reference spectra.
    - 'drop' : Optionally removes unnecessary tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> processed_mrbles = mrbles(spectra=my_spectra, codes=my_codes, data=my_image_data, channels=[0], overlap=100)
    
    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, using channel 0 for analysis, and matching bead spectral signatures against `my_spectra` and `my_codes`.
    """


def mrbles_pipe(
    spectra: str | PathLike | StringIO,
    codes: str | PathLike | StringIO,
    flatfield: float = 1.0,
    darkfield: float = 0.0,
    overlap: int = 102,
    min_bead_radius: int = 5,
    max_bead_radius: int = 25,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.3,
    roi_length: int = 61,
    search_channel: str | list[str] | None = None,
    reference: str = "eu",
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> Pipeline:
    """
    Build the pipe to processes image data for MRBLEs (Microsphere-based Rolling Circle Amplification Beads), applying flatfield correction, bead detection, and spectral decoding.

    Parameters
    ----------
    spectra : 
        The reference spectra data for the MRBLEs. This is used to identify the spectral signatures of the beads.
    codes : 
        The codes corresponding to the reference spectra. These codes are used to map specific spectral signatures to particular bead identities.
    flatfield : 
        The flatfield correction factor or path to a flatfield correction image. If a file path is provided, 
        the image will be loaded from the specified file (e.g., a TIFF or Zarr file). Flatfield correction 
        is used to account for uneven illumination across the image. If set to a numeric value (e.g., 1.0), 
        no flatfield correction will be applied.
    darkfield : 
        The darkfield correction factor or path to a darkfield correction image. If a file path is provided, 
        the image will be loaded from the specified file (e.g., a TIFF or Zarr file). Darkfield correction 
        is used to account for background noise in the image. If set to a numeric value (e.g., 0.0), no 
        darkfield correction will be applied.
    overlap : 
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
        This overlap value is subtracted from both the vertical (y) and horizontal (x) dimensions
        of the tiles to remove redundant or overlapping areas between adjacent tiles.
    min_button_radius / max_button_radius : 
        The minimum/maximum radius (in pixels) for detecting buttons in the image.
    low_edge_quantile / high_edge_quantile: 
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons.
    num_iter : 
        The maximum number of iterations to perform during the bead detection process, controlling the depth of search.
    min_roundness : 
        The minimum roundness value for detected beads. Beads that do not meet this roundness threshold are excluded.
    roi_length : 
        The length (in pixels) of the region of interest (ROI) around detected beads.
    search_channel : 
        The channel or list of channels to use for bead detection and analysis. If `None`, all channels are used.
    reference : 
        The reference material or standard used for spectral decoding. The default is "eu" (Europium), which is typically used in MRBLEs for comparison in spectral analysis.
    squeeze :
        If True, removes singleton dimensions from the dataset (dimensions of size 1), simplifying the data structure.
    roi_only : 
        If True, only returns the region of interest (ROI) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    pipe : Pipeline
        The constructed pipeline object, either configured and returned if `return_pipe` is True, or executed as part of the image processing workflow.

    Notes
    -----
    This function constructs a pipeline architecture to process Marbles image data. The steps in the pipeline include:
    
    - 'flatfield_correct' : Applies flatfield and darkfield corrections to the image data.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'find_beads' : Detects beads based on specified radii, roundness, and other parameters.
    - 'identify_mrbles' : Identifies MRBLEs beads by matching their spectral signatures to the provided reference spectra.
    - 'drop' : Optionally removes unnecessary tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> mrbles_pipe = mrbles(spectra=my_spectra, codes=my_codes, channels=[0], overlap=100)
    >>> processed_mrbles = mrbles_pipe(data=my_image_data)
    
    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, using channel 0 for analysis, and matching bead spectral signatures against `my_spectra` and `my_codes`.
    """
    config = {key: value for key, value in locals().items()}

    pipe = Pipeline("read", config=config)
    pipe.add_pipe("flatfield_correct")
    pipe.add_pipe("stitch")
    pipe.add_pipe("find_beads")
    pipe.add_pipe("identify_mrbles")
    pipe.add_pipe("drop")

    return pipe

def beads(
    data: ArrayLike | str,
    times: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    debug: bool = False,
    flatfield:float = 1.0,
    darkfield:float = 0.0,
    overlap: int = 102,
    min_bead_radius: int = 5,
    max_bead_radius: int = 25,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.3,
    roi_length: int = 61,
    search_channel: str | list[str] | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True
) -> xr.Dataset | list[xr.Dataset]:
    pipe = beads_pipe(debug=debug, flatfield=flatfield, darkfield=darkfield, overlap=overlap,
                      min_bead_radius=min_bead_radius, max_bead_radius=max_bead_radius,
                      low_edge_quantile=low_edge_quantile, high_edge_quantile=high_edge_quantile,
                      num_iter=num_iter, min_roundness=min_roundness, roi_length=roi_length,
                      search_channel=search_channel, squeeze=squeeze, roi_only=roi_only, drop_tiles=drop_tiles)
    return pipe(data=data, times=times, channels=channels)

"""
    Processes image data to detect beads, applying flatfield correction, stitching, and optional tile dropping.

    Parameters
    ----------
    data : 
        The input image data to be processed. It can be one of the following:
        - A file path (string) to image data.
        - An `xarray.DataArray` or `xarray.Dataset` containing image data.
        - A sequence (list or tuple) of file paths, `xarray.DataArray`, or `xarray.Dataset`.
        The function can handle multiple formats and will standardize them into an `xarray.Dataset` for processing.
    times : 
        A list or sequence of time points to extract from the dataset. If specified, only the data
        corresponding to these time points will be processed. If `None`, all available time points
        in the data will be used.
    channels : 
        A list or sequence of channel names to extract from the dataset. If specified, only the data
        from the specified channels will be processed. If `None`, all available channels will be used.
    debug : 
        If True, set the logger to debug mode for error logging.
    flatfield : 
        The flatfield correction factor or path to a flatfield correction image. If a file path is provided, 
        the image will be loaded from the specified file (e.g., a TIFF or Zarr file). Flatfield correction 
        is used to account for uneven illumination across the image. If set to a numeric value (e.g., 1.0), 
        no flatfield correction will be applied.
    darkfield : 
        The darkfield correction factor or path to a darkfield correction image. If a file path is provided, 
        the image will be loaded from the specified file (e.g., a TIFF or Zarr file). Darkfield correction 
        is used to account for background noise in the image. If set to a numeric value (e.g., 0.0), no 
        darkfield correction will be applied.
    overlap : 
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
        This overlap value is subtracted from both the vertical (y) and horizontal (x) dimensions
        of the tiles to remove redundant or overlapping areas between adjacent tiles.
    min_button_radius / max_button_radius : 
        The minimum/maximum radius (in pixels) for detecting buttons in the image.
    low_edge_quantile / high_edge_quantile: 
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons.
    num_iter : 
        The maximum number of iterations to perform during the bead detection process, controlling the depth of search.
    min_roundness : 
        The minimum roundness value for beads to be detected. A higher value enforces stricter roundness requirements.
    roi_length : 
        The length (in pixels) of the region of interest (ROI) around detected beads. This determines the size of the sub-image extracted around each detected bead.
    search_channel : 
        The channel or list of channels to use for bead detection. If `None`, all channels will be used for the search.
    squeeze :
        If True, removes singleton dimensions from the dataset (dimensions of size 1), simplifying the data structure.
    roi_only : 
        If True, only returns the region of interest (ROI) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    Processed image(s): xr.Dataset | list[xr.Dataset]
        The processed image in xr.Dataset from executed pipeline as the outcome of the image pipeline workflow.

    Notes
    -----
    This function uses a pipeline architecture to process image data. The steps in the pipeline include:
    
    - 'flatfield_correct' : Applies flatfield and darkfield corrections to the image data.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'find_beads' : Detects beads in the image based on specified radii and other detection parameters.
    - 'drop' : Optionally removes unnecessary tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> bead_image = beads(data=my_image_data, channels=[0, 1], overlap=100, min_bead_radius=5, max_bead_radius=20)
    
    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, using only channels 0 and 1, and detects beads with a radius between 5 and 20 pixels.
    """

def beads_pipe(
    debug:bool = False,
    flatfield:float = 1.0,
    darkfield:float = 0.0,
    overlap: int = 102,
    min_bead_radius: int = 5,
    max_bead_radius: int = 25,
    low_edge_quantile: float = 0.1,
    high_edge_quantile: float = 0.9,
    num_iter: int = 5000000,
    min_roundness: float = 0.3,
    roi_length: int = 61,
    search_channel: str | list[str] | None = None,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> Pipeline:
    """
    Build the pipe to processes image data to detect beads, applying flatfield correction, stitching, and optional tile dropping.

    Parameters
    ----------
    debug : 
        If True, set the logger to debug mode for error logging.
    flatfield : 
        The flatfield correction factor or path to a flatfield correction image. If a file path is provided, 
        the image will be loaded from the specified file (e.g., a TIFF or Zarr file). Flatfield correction 
        is used to account for uneven illumination across the image. If set to a numeric value (e.g., 1.0), 
        no flatfield correction will be applied.
    darkfield : 
        The darkfield correction factor or path to a darkfield correction image. If a file path is provided, 
        the image will be loaded from the specified file (e.g., a TIFF or Zarr file). Darkfield correction 
        is used to account for background noise in the image. If set to a numeric value (e.g., 0.0), no 
        darkfield correction will be applied.
    overlap : 
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
        This overlap value is subtracted from both the vertical (y) and horizontal (x) dimensions
        of the tiles to remove redundant or overlapping areas between adjacent tiles.
    min_button_radius / max_button_radius : 
        The minimum/maximum radius (in pixels) for detecting buttons in the image.
    low_edge_quantile / high_edge_quantile: 
        The lower/upper quantile for edge detection, used to identify the dimmest edges when detecting buttons.
    num_iter : 
        The maximum number of iterations to perform during the bead detection process, controlling the depth of search.
    min_roundness : 
        The minimum roundness value for beads to be detected. A higher value enforces stricter roundness requirements.
    roi_length : 
        The length (in pixels) of the region of interest (ROI) around detected beads. This determines the size of the sub-image extracted around each detected bead.
    search_channel : 
        The channel or list of channels to use for bead detection. If `None`, all channels will be used for the search.
    squeeze :
        If True, removes singleton dimensions from the dataset (dimensions of size 1), simplifying the data structure.
    roi_only : 
        If True, only returns the region of interest (ROI) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    pipe : Pipeline
        The constructed Pipeline object, either configured and returned if `return_pipe` is True, or executed to return an xarray.

    Notes
    -----
    This function constructs a pipeline architecture to process image data. The steps in the pipeline include:
    - 'flatfield_correct' : Applies flatfield and darkfield corrections to the image data.
    - 'stitch' : Stitches image tiles based on the overlap parameter.
    - 'find_beads' : Detects beads in the image based on specified radii and other detection parameters.
    - 'drop' : Optionally removes unnecessary tiles and simplifies the dataset based on the `squeeze`, `roi_only`, and `drop_tiles` options.

    Examples
    --------
    >>> bead_pipe = beads_pipe(channels=[0, 1], overlap=100, min_bead_radius=5, max_bead_radius=20)
    >>> beads = bead_pipe(data=my_image_data)
    
    This processes `my_image_data` by stitching tiles with 100 pixels of overlap, using only channels 0 and 1, and detects beads with a radius between 5 and 20 pixels.
    """
    config = {key: value for key, value in locals().items()}
    pipe = Pipeline("read", config=config)
    pipe.add_pipe("flatfield_correct")
    pipe.add_pipe("stitch")
    pipe.add_pipe("find_beads")
    pipe.add_pipe("drop")

    return pipe 

def image(
    data: ArrayLike | str,
    times: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    debug:bool = False,
    overlap: int = 102,
    rotation:float = 0,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> xr.Dataset | list[xr.Dataset]:
    pipe = image_pipe(debug=debug, overlap=overlap, rotation=rotation, squeeze=squeeze, roi_only=roi_only, drop_tiles=drop_tiles)
    return pipe(data=data, times=times, channels=channels)
"""
    Create an imaging pipeline with the specified configuration.

    Parameters
    ----------
    data : 
        The input image data to be processed. It can be one of the following:
        - A file path (string) to image data.
        - An `xarray.DataArray` or `xarray.Dataset` containing image data.
        - A sequence (list or tuple) of file paths, `xarray.DataArray`, or `xarray.Dataset`.
        The function can handle multiple formats and will standardize them into an `xarray.Dataset` for processing.
    times : 
        A list or sequence of time points to extract from the dataset. If specified, only the data
        corresponding to these time points will be processed. If `None`, all available time points
        in the data will be used.
    channels : 
        A list or sequence of channel names to extract from the dataset. If specified, only the data
        from the specified channels will be processed. If `None`, all available channels will be used.
    debug : 
        If True, set the logger to debug mode for error logging.
    overlap : 
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
    rotation : 
        The degree of rotation to apply to the image.
    squeeze :
        If True, removes singleton dimensions from the dataset (dimensions of size 1), simplifying the data structure.
    roi_only : 
        If True, only returns the region of interest (ROI) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    Processed image(s): xr.Dataset | list[xr.Dataset]
        The processed image in xr.Dataset from executed pipeline as the outcome of the image pipeline workflow.
    
    Notes
    -----
    This function uses a pipeline architecture to process image data. The steps in the pipeline include:
    
    - 'stitch' : Stitch together image tiles based on the overlap parameter.
    - 'rotate' : Rotate the image by the specified angle.
    - 'drop' : Depending on the options for `squeeze`, `roi_only`, and `drop_tiles`, unnecessary or unused 
      tiles are removed, and the data is optionally simplified.

    Examples
    --------
    >>> processed_image = image(data=my_image_data, channels=[0, 1], overlap=100, rotation=45, squeeze=True)
    """

def image_pipe(
    debug:bool = False,
    overlap: int = 102,
    rotation:float = 0,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
) -> Pipeline:
    """
    Create an imaging pipeline with the specified configuration.

    Parameters
    ----------
    debug : 
        If True, set the logger to debug mode for error logging.
    overlap : 
        The number of pixels to exclude from the edges of adjacent tiles during the stitching process.
    rotation : 
        The degree of rotation to apply to the image.
    squeeze :
        If True, removes singleton dimensions from the dataset (dimensions of size 1), simplifying the data structure.
    roi_only : 
        If True, only returns the region of interest (ROI) from the dataset, ignoring other parts of the image.
    drop_tiles : 
        If True, removes the "tile" variable from the dataset after stitching, effectively dropping individual tile data.

    Returns
    -------
    pipe : Pipeline
        The constructed pipeline object, either configured and returned if `return_pipe` is True, or executed to return an xarray.
    
    Notes
    -----
    This function uses a pipeline architecture to process image data. The steps in the pipeline include:
    
    - 'stitch' : Stitch together image tiles based on the overlap parameter.
    - 'rotate' : Rotate the image by the specified angle.
    - 'drop' : Depending on the options for `squeeze`, `roi_only`, and `drop_tiles`, unnecessary or unused 
      tiles are removed, and the data is optionally simplified.

    Examples
    --------
    >>> image_pipe = image(channels=[0, 1], overlap=100, rotation=45, squeeze=True)
    >>> processed_image = image_pipe(data=my_image_data)
    """
    config = {key: value for key, value in locals().items()}
    pipe = Pipeline("read", config=config)
    pipe.add_pipe("stitch")
    pipe.add_pipe("rotate")
    pipe.add_pipe("drop")
    return pipe

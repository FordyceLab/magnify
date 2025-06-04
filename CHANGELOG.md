#Changelog

## Unreleased

### Added

### Fixed

### Removed

### Updated


## Magnify 0.12.2 (4 June 2025)

### Fixed
 - Fix an issue where cluster finding would fail when all clusters have the same mean in a given dimension. [48b9c77](https://github.com/FordyceLab/magnify/commit/48b9c77c67822a3c49b84ebb06508f6c352e88ce)

### Updated
 - Update image stitching to take the middle part of each image rather than the left. Thanks to @palmhjell. [#24](https://github.com/FordyceLab/magnify/pull/24)

## Magnify 0.12.1 (3 April 2025)

### Updated
 - Change opencv dependency to be opencv-headless. This fixes issues with Qt conflicts. [fcfda8c](https://github.com/FordyceLab/magnify/commit/fcfda8cf961191f257deca0505d64dff6589cbb8)

### Fixed
 - Fix the typo in registry.py that prevented `mg.beads` from running. [3e7d825](https://github.com/FordyceLab/magnify/commit/3e7d82583191f6cc751eb623fca226630e50f2e5)

## Magnify 0.12.0 (3 April 2025)

### Added
 - Add a `circle_mask` component which zeroes out everything outside/inside a given circle. [5c45cbb](https://github.com/FordyceLab/magnify/commit/5c45cbbe820121e7cf1e38a492ee9f573ff1e7f7)
 - Add the ability to pass in callables to `add_pipes` and to specify a custom name for components to prevent name clashes. [cc19deb](https://github.com/FordyceLab/magnify/commit/cc19debac94aff56edb217b1ad83acbd2a11ba33)
 - Export the component decorator as a public magnify function. [b79c10b](https://github.com/FordyceLab/magnify/commit/b79c10ba24e751bd86ab4a4628708729378cbcb9)

### Fixed
 - Make the array caching mechanism work with zarr v3. [7b654c7](https://github.com/FordyceLab/magnify/commit/7b654c744a917bc087ae5023aa6ed655d0cbb722)
 - Fix the minimum bead distance to be equal to the minimum bead radius rather than the diameter. [0c90dd1](https://github.com/FordyceLab/magnify/commit/0c90dd1a856390304581f02db3f61900273a9bd3)
 - Make sure magnify doesn't crash when it doesn't find any circles in the circle finding subroutine. [4fee6d8](https://github.com/FordyceLab/magnify/commit/4fee6d8e7572bca75b60f26d58fb5333a15dfc52).
 - Ensure magnify correctly handles non uint16 input images in button finding. [cbb5c8d](https://github.com/FordyceLab/magnify/commit/cbb5c8d9da2c63f2b6849ec1808104e5bc407784)

### Updated
 - Remove Python 3.10 support and added Python 3.13 support. [18321c7](https://github.com/FordyceLab/magnify/commit/18321c7101d481ec8b5017cca6454e2f8f4d56ab)
 - fg and bg no longer have an unnecessary channel dimension. [01a4f2a](https://github.com/FordyceLab/magnify/commit/01a4f2adb47bf72ebdabd0dafd22d6b53be31add)
 - Parameters now get directly passed to `add_pipe` rather than the Pipeline constructor which makes pipeline construction simpler to understand. [2a27c93](https://github.com/FordyceLab/magnify/commit/2a27c93e6d375568e9ec0c4b6652509f6d61dc7e)
 - Move the option to specify custom channel/time dimension names from the reader to a seperate component called `rename_labels` that is significantly more flexible. [dc741c6](https://github.com/FordyceLab/magnify/commit/dc741c613456d97bb59084f40bc41e10ca012f80)
 - Move array format standardization and restoration into their own pipeline components called `standardize_format` and `restore_format`. [74fb7a3](https://github.com/FordyceLab/magnify/commit/74fb7a3ffcbc2dfbbe6bda90dd4876686a326851)


## Magnify 0.11.5 (20 March 2025)

### Added
 - Add the option to specify the left or top boundaries of a chip when calling `microfluidic_chip`. [a6a0ad6](https://github.com/FordyceLab/magnify/commit/a6a0ad64be723be07e573c3146c683a026bdc8b9)
 - Add support for [BaSiC](https://github.com/peng-lab/BaSiCPy) correction. Because BaSiCPy restricts the version of many dependencies we intentionally don't add basicpy as a dependency for magnify. [938e020](https://github.com/FordyceLab/magnify/commit/938e0207e4ccca5d767d3eb44e508e2b2b140280)


## Magnify 0.11.4 (19 March 2025)

### Fixed
 - Set the default value of `roi_length` in `microfluidic_chip_pipe` to `None` rather than a fixed value. [59a4eb4](https://github.com/FordyceLab/magnify/commit/59a4eb48338d24ee3faeb31487940d5061978db5) and [08d8f82](https://github.com/FordyceLab/magnify/commit/08d8f82abb0e194e27e8def9b6e6c444a8a29c97).

### Removed
 - Dropped pyqt dependencies since this often causes conflicts with prior QT installations. [f8e8cc6](https://github.com/FordyceLab/magnify/commit/f8e8cc67a1838ab7ecb592af4f1b104f122b077d)

### Updated
 - Make `imshow` run faster when an image has many markers. [2ff4d67](https://github.com/FordyceLab/magnify/commit/2ff4d678d153d82caa912518bdf750d4b382d6db)
 - Render large images as [multiscale images](https://napari.org/stable/howtos/layers/image.html#multiscale-images) in `imshow`. [6d3d041](https://github.com/FordyceLab/magnify/commit/6d3d041ecb8cd9df737a151715e5252e9c9f2b7b)
 - Updated `imshow` to properly handle xarray datasets with `mark_row` and `mark_col` dimensions. [ed1b3ee](https://github.com/FordyceLab/magnify/commit/ed1b3ee0017e3c91034942a6ad2531f32bfaa103)

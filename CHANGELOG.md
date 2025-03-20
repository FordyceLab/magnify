# Changelog

## Unreleased

### Added

### Fixed

### Removed

### Updated

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

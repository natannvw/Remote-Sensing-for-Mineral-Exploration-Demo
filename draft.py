from collections import namedtuple

import numpy as np
import pandas as pd
import rasterio

Raster = namedtuple("Raster", ["datacube", "profile", "name", "path"])
Spectrum = namedtuple("Spectrum", ["wavelength", "reflectance"])


def create_raster(
    datacube: np.array, profile: dict, path: str = None, name: str = None
) -> Raster:
    raster = Raster(datacube=datacube, profile=profile, path=path, name=name)

    return raster


def create_spectrum(wavelength: np.array, reflectance: np.array) -> Spectrum:
    spectrum = Spectrum(wavelength=wavelength, reflectance=reflectance)

    return spectrum


def get_spectrum(mineral_name: str) -> Spectrum:
    if mineral_name == "kaolinite":  # TODO add more minerals
        path = r"Spectral Library\Kaolinite.txt"
        df = pd.read_csv(
            path, skiprows=2, header=None, names=["wavelength", mineral_name]
        )

        wavelength = df["wavelength"].to_numpy()
        reflectance = df[mineral_name].to_numpy()

    spectrum = create_spectrum(wavelength=wavelength, reflectance=reflectance)

    return spectrum


def continuum_remove(raster: Raster) -> Raster:
    # TODO
    pass


def sam(raster: Raster, ref_spectrum: Spectrum) -> float:
    # TODO
    pass


def spectralMatch(raster: Raster, ref_spectrum: Spectrum, method: str = "sam") -> float:
    # TODO
    pass


def resample_spectrum(spectrum, desired_wavelengths):
    resampled = np.interp(
        desired_wavelengths, spectrum.wavelength, spectrum.reflectance
    )

    return resampled


def preprocess_datacube(datacube):
    # TODO remove bands by range
    pass


if __name__ == "__main__":
    path = r"Data\Cuprite Nevada\ENMAP01-____L1C-DT0000025905_20230707T192008Z_001_V010303_20230922T131737Z-SPECTRAL_IMAGE.TIF"

    with rasterio.open(path) as src:
        profile = src.profile
        datacube = src.read()

    raster = create_raster(datacube=datacube, profile=profile, path=path)

    ref_spectrum = get_spectrum(mineral_name="kaolinite")

    raster.datacube = preprocess_datacube(raster.datacube)  # TODO

    raster = continuum_remove(raster)  # TODO

    ref_spectrum.reflectance = resample_spectrum(
        spectrum=ref_spectrum, desired_wavelengths=raster.wavelength
    )

    sam_score = spectralMatch(raster, ref_spectrum, method="sam")  # TODO

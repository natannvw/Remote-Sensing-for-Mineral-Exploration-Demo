import os
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import xmltodict
from scipy.interpolate import interp1d
from torchmetrics.image import SpectralAngleMapper


class Raster:
    def __init__(self, wavelength, datacube, metadata, profile, name=None, path=None):
        self.wavelength = wavelength
        self._datacube = datacube
        self.metadata = metadata
        self.profile = profile
        self.name = name
        self.path = path

    @property
    def datacube(self):
        return self._datacube

    @datacube.setter
    def datacube(self, value):
        self._datacube = value


Spectrum = namedtuple("Spectrum", ["wavelength", "reflectance"])


def create_raster(
    datacube: np.array,
    profile: dict,
    wavelength: np.array = None,
    metadata: dict = None,
    path: str = None,
    name: str = None,
) -> Raster:
    raster = Raster(
        wavelength=wavelength,
        datacube=datacube,
        metadata=metadata,
        profile=profile,
        path=path,
        name=name,
    )

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


def continuum_removal(raster: Raster) -> Raster:
    # TODO
    pass


def sam(raster: Raster, ref_spectrum: Spectrum) -> float:
    # TODO
    pass


def spectralMatch(raster: Raster, ref_spectrum: Spectrum, method: str = "sam") -> float:
    if method == "sam":
        raster.datacube = torch.tensor(raster.datacube, dtype=torch.float32)
        ref_spectrum.reflectance = torch.tensor(
            ref_spectrum.reflectance, dtype=torch.float32
        )

        sam = SpectralAngleMapper()

        score = sam(raster.datacube, ref_spectrum.reflectance, reduction="none")

    return score


def resample_spectrum(spectrum, desired_wavelengths):
    resampled = np.interp(
        desired_wavelengths, spectrum.wavelength, spectrum.reflectance
    )

    return resampled


def preprocess(raster: Raster):
    # TODO remove bands by range
    # TODO remove bands by wavelength
    # TODO remove bands by index
    # TODO consider smoothing by Savitzky-Golay filter
    # TODO consider normalizing by continuum removal
    # TODO consider rescale each band to reflectance
    # TODO convert tu um

    raster.datacube = continuum_removal(raster.datacube)  # TODO

    pass


def find_metadata_file(tiff_file_path):
    directory = os.path.dirname(tiff_file_path)

    xml_filename = os.path.basename(tiff_file_path).replace(
        "SPECTRAL_IMAGE.TIF", "METADATA.XML"
    )

    # Search for the XML file in the same directory
    for file in os.listdir(directory):
        if file == xml_filename:
            xml_file_path = os.path.join(directory, file)
            break
    else:
        raise FileNotFoundError("XML file not found.")

    return xml_file_path


def get_metadata(tiff_file_path):
    xml_file_path = find_metadata_file(tiff_file_path)

    with open(xml_file_path, "r", encoding="utf-8") as file:
        my_xml = file.read()

    # Use xmltodict to parse and convert the XML document
    metadata_dict = xmltodict.parse(my_xml)

    return metadata_dict


def get_wavelengths(metadata_dict):
    band_characterisation = metadata_dict["level_X"]["specific"]["bandCharacterisation"]

    band_ids = band_characterisation["bandID"]

    # Extracting wavelengthCenterOfBand values
    wavelengths = [band["wavelengthCenterOfBand"] for band in band_ids]

    return np.array(wavelengths, dtype=np.float32)


def replace_bad_bands_reflectance(datacube):
    # Assuming datacube is a 3D numpy array with shape (bands, rows, columns)
    # and -32768 indicates a bad value that needs to be replaced

    # Function to interpolate a single spectrum
    def interpolate_spectrum(spectrum, bad_value=-32768):
        # Identify the bad values
        bad_indices = np.where(spectrum == bad_value)[0]

        # If all values are bad or no bad values are found, return the original spectrum
        if bad_indices.size == 0 or bad_indices.size == spectrum.size:
            return spectrum

        # Identify the good values
        good_indices = np.where(spectrum != bad_value)[0]
        good_values = spectrum[good_indices]

        # Create the interpolation function
        f_interp = interp1d(
            good_indices,
            good_values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        # Interpolate the bad values
        interpolated_values = f_interp(bad_indices)

        # Replace the bad values in the spectrum
        spectrum[bad_indices] = interpolated_values

        return spectrum

    # Apply interpolation to each pixel
    for row in range(datacube.shape[1]):
        for col in range(datacube.shape[2]):
            datacube[:, row, col] = interpolate_spectrum(datacube[:, row, col])

    return datacube


def get_gains_and_offsets(metadata_dict):
    band_characterisation = metadata_dict["level_X"]["specific"]["bandCharacterisation"]

    band_ids = band_characterisation["bandID"]

    # Extracting gain values
    gains = [band["GainOfBand"] for band in band_ids]

    # Extracting offset values
    offsets = [band["OffsetOfBand"] for band in band_ids]

    return np.array(gains, dtype=np.float32), np.array(offsets, dtype=np.float32)


def rescale(raster: Raster):
    gains, offsets = get_gains_and_offsets(raster.metadata)

    raster.datacube = raster.datacube * gains[:, None, None] + offsets[:, None, None]

    raster.datacube = datacube


if __name__ == "__main__":
    filename = "ENMAP01-____L2A-DT0000025905_20230707T192008Z_001_V010303_20230922T131734Z-SPECTRAL_IMAGE.TIF"
    data_folder = "Data"
    cuprite_nevada_folder = "Cuprite Nevada"
    path = os.path.join(data_folder, cuprite_nevada_folder, filename)

    with rasterio.open(path) as src:
        profile = src.profile
        datacube = src.read()
        metadata = get_metadata(path)
        wavelength = get_wavelengths(metadata)

    # wavelength =
    raster = Raster(
        wavelength=wavelength,
        datacube=datacube,
        metadata=metadata,
        profile=profile,
        path=path,
    )
    plt.figure()
    plt.plot(raster.wavelength, raster.datacube[:, 500, 500])

    raster = rescale(raster)

    replace_bad_bands_reflectance(raster.datacube)

    # plt.figure()
    # plt.plot(raster.wavelength, datacube[:, 500, 500])

    raster = preprocess(raster)  # TODO

    ref_spectrum = get_spectrum(mineral_name="kaolinite")

    plt.plot(ref_spectrum.wavelength, ref_spectrum.reflectance)

    ref_spectrum.reflectance = resample_spectrum(
        spectrum=ref_spectrum, desired_wavelengths=raster.wavelength
    )

    sam_score = spectralMatch(raster, ref_spectrum, method="sam")  # TODO

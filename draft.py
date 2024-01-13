import os
from typing import List, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysptools.spectro as spectro
import rasterio
import torch
import xmltodict
from numba import jit, prange
from pyproj.crs import CRS
from rasterio.io import MemoryFile
from rasterio.mask import mask
from scipy.spatial import ConvexHull
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


class Spectrum:
    def __init__(self, wavelength, reflectance):
        self._wavelength = wavelength
        self._reflectance = reflectance

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value

    @property
    def reflectance(self):
        return self._reflectance

    @reflectance.setter
    def reflectance(self, value):
        self._reflectance = value


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


# @ray.remote
def process_pixel(pixel, wavelengths):
    return spectro.convex_hull_removal(pixel, wavelengths)


def continuum_remove(wvlns, reflectance):
    """
    Remove continuum from reflectance spectrum.

    :param wvlns: Wavelengths (1D array).
    :param reflectance: Reflectance values (1D array).
    :return: Continuum removed reflectance (1D array).
    """
    small = 10 * np.finfo(float).eps

    # Extend the wavelengths and reflectance vectors
    x_ext = np.concatenate(([wvlns[0] - small], wvlns, [wvlns[-1] + small]))
    v_ext = np.concatenate(([0], reflectance, [0]))

    # Compute the convex hull
    points = np.vstack((x_ext, v_ext)).T
    hull = ConvexHull(points)
    k = hull.vertices

    # Remove the first and last points of the hull and sort
    k = np.delete(k, [0, -1])
    k.sort()
    k -= 1

    # Linear interpolation
    resampled_spectrum = np.interp(wvlns, wvlns[k], reflectance[k])

    # Continuum removal
    CR = reflectance / resampled_spectrum

    return CR


def continuum_removal(raster: Raster) -> np.array:
    # from PerformanceMonitor import PerformanceMonitor

    bands, rows, cols = raster.datacube.shape
    reshaped_data = np.reshape(raster.datacube.transpose(1, 2, 0), (rows * cols, bands))

    for index in range(rows * cols):
        # for i in range(1):
        # i, j = 500, 500
        # index = i * cols + j
        # raster.datacube[:, i, j], _, _ = spectro.convex_hull_removal(
        #     raster.datacube[:, i, j], raster.wavelength
        # )
        # perf_monitor = PerformanceMonitor()
        # perf_monitor.start()
        reshaped_data[index, :] = continuum_remove(
            raster.wavelength,
            reshaped_data[index, :],
        )
        # perf_monitor.stop()

    # Reshape back to original shape
    raster.datacube = np.reshape(reshaped_data, (rows, cols, bands)).transpose(2, 0, 1)

    return raster


def sam(raster: Raster, ref_spectrum: Spectrum) -> float:
    # TODO
    pass


def spectralMatch(raster: Raster, ref_spectrum: Spectrum, method: str = "sam") -> float:
    # TODO make a check for raster or datacube like in my code
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
    spectrum.wavelength = desired_wavelengths
    spectrum.reflectance = resampled

    return spectrum


def nm2um(wavelength):
    return wavelength / 1000


def removeBands(
    object: Optional[Union[Raster, Spectrum]],
    wave_or_band: str,
    wlrange_or_bandrange: Union[float, int, List[Union[float, int]]],
) -> Raster:
    if type(object) == Raster:
        if wave_or_band == "Wavelength":
            wavelength_indices = np.where(
                (object.wavelength >= wlrange_or_bandrange[0])
                & (object.wavelength <= wlrange_or_bandrange[1])
            )[0]

            object.wavelength = object.wavelength[wavelength_indices]
            object.datacube = object.datacube[wavelength_indices, :, :]

        elif wave_or_band == "BandNumber":
            object.wavelength = object.wavelength[
                wlrange_or_bandrange[0] : wlrange_or_bandrange[1]
            ]
            object.datacube = object.datacube[
                wlrange_or_bandrange[0] : wlrange_or_bandrange[1], :, :
            ]

    elif type(object) == Spectrum:
        if wave_or_band == "Wavelength":
            wavelength_indices = np.where(
                (object.wavelength >= wlrange_or_bandrange[0])
                & (object.wavelength <= wlrange_or_bandrange[1])
            )[0]

            object.wavelength = object.wavelength[wavelength_indices]
            object.reflectance = object.reflectance[wavelength_indices]

        elif wave_or_band == "BandNumber":
            object.wavelength = object.wavelength[
                wlrange_or_bandrange[0] : wlrange_or_bandrange[1]
            ]
            object.reflectance = object.reflectance[
                wlrange_or_bandrange[0] : wlrange_or_bandrange[1]
            ]

    return object


def preprocess(raster: Raster):
    # TODO consider smoothing by Savitzky-Golay filter

    raster.datacube = replace_bad_bands_reflectance(raster.datacube)
    raster = rescale(raster)

    raster.wavelength = nm2um(raster.wavelength)

    raster = removeBands(raster, "Wavelength", [1, 2.5])

    return raster


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


@jit(nopython=True)
def linear_interpolate(indices, values, query_points):
    # Custom linear interpolation logic
    result = np.empty(len(query_points))
    for i, x in enumerate(query_points):
        if x <= indices[0]:
            result[i] = values[0]
        elif x >= indices[-1]:
            result[i] = values[-1]
        else:
            # Find the interval x is in
            for j in range(len(indices) - 1):
                if x <= indices[j + 1]:
                    break
            # Linearly interpolate
            x0, x1 = indices[j], indices[j + 1]
            y0, y1 = values[j], values[j + 1]
            result[i] = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    return result


@jit(nopython=True)
def interpolate_spectrum(spectrum, bad_value=-32768):
    bad_indices = np.where(spectrum == bad_value)[0]

    if bad_indices.size == 0 or bad_indices.size == spectrum.size:
        return spectrum

    good_indices = np.where(spectrum != bad_value)[0]
    good_values = spectrum[good_indices]

    # Use the custom linear interpolation
    interpolated_values = linear_interpolate(good_indices, good_values, bad_indices)

    spectrum[bad_indices] = interpolated_values

    return spectrum


#     # Identify the bad values
#     bad_indices = np.where(spectrum == bad_value)[0]

#     # If all values are bad or no bad values are found, return the original spectrum
#     if bad_indices.size == 0 or bad_indices.size == spectrum.size:
#         return spectrum

#     # Identify the good values
#     good_indices = np.where(spectrum != bad_value)[0]
#     good_values = spectrum[good_indices]

#     # Create the interpolation function
#     f_interp = interp1d(
#         good_indices,
#         good_values,
#         kind="linear",
#         bounds_error=False,
#         fill_value="extrapolate",
#     )

#     # Interpolate the bad values
#     interpolated_values = f_interp(bad_indices)

#     # Replace the bad values in the spectrum
#     spectrum[bad_indices] = interpolated_values

#     return spectrum


@jit(nopython=True, parallel=True)
def replace_bad_bands_reflectance(datacube):
    # Assuming datacube is a 3D numpy array with shape (bands, rows, columns)
    # and -32768 indicates a bad value that needs to be replaced

    # Apply interpolation to each pixel
    for row in prange(datacube.shape[1]):
        for col in prange(datacube.shape[2]):
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


def rescale(raster: Raster) -> Raster:
    gains, offsets = get_gains_and_offsets(raster.metadata)

    raster.datacube = raster.datacube * gains[:, None, None] + offsets[:, None, None]

    return raster


def clip_raster(raster: Raster, polygon: gpd.GeoDataFrame) -> Raster:
    polygon = reproject_gdf(polygon, dst_crs=raster.profile["crs"])

    with MemoryFile() as memfile:  # the mask() function expects a Rasterio dataset as the src argument
        with memfile.open(**raster.profile) as src:
            src.write(raster.datacube)

            clipped_data, clipped_transform = mask(src, polygon.geometry, crop=True)

    # Update the profile
    clipped_profile = raster.profile.copy()
    clipped_profile.update(
        {
            "height": clipped_data.shape[1],
            "width": clipped_data.shape[2],
            "transform": clipped_transform,
        }
    )

    # Assign the clipped data to the object
    raster.datacube = clipped_data
    raster.profile = clipped_profile

    return raster


def reproject_gdf(
    gdf: gpd.GeoDataFrame, dst_crs=CRS.from_epsg(4326)
) -> gpd.GeoDataFrame:
    """
    This function takes a GeoDataFrame and a target CRS as input,
    checks if the CRS of the GeoDataFrame is the same as the target CRS,
    if not, reprojects the GeoDataFrame to the target CRS.
    """
    # Reproject the GeoDataFrame if the CRS does not match
    if CRS(gdf.crs) != dst_crs:
        print(f"Reprojecting GeoDataFrame from {gdf.crs} to {dst_crs}")
        gdf = gdf.to_crs(dst_crs)

    return gdf


def spectrum_preprocess(spectrum: Spectrum, raster: Raster):
    spectrum = resample_spectrum(
        spectrum=ref_spectrum, desired_wavelengths=raster.wavelength
    )
    # spectrum = removeBands(spectrum, "Wavelength", [1, 2.5]) # dont need because it was resampled to raster.wavelength already
    return spectrum


if __name__ == "__main__":
    filename = "ENMAP01-____L2A-DT0000025905_20230707T192008Z_001_V010303_20230922T131734Z-SPECTRAL_IMAGE.TIF"
    data_folder = "Data"
    cuprite_nevada_folder = "Cuprite Nevada"
    raster_path = os.path.join(data_folder, cuprite_nevada_folder, filename)

    with rasterio.open(raster_path) as src:
        profile = src.profile
        datacube = src.read()
        metadata = get_metadata(raster_path)
        wavelength = get_wavelengths(metadata)

    # wavelength =
    raster = Raster(
        wavelength=wavelength,
        datacube=datacube,
        metadata=metadata,
        profile=profile,
        path=raster_path,
    )
    plt.figure()
    plt.plot(raster.wavelength, raster.datacube[:, 500, 500])

    polygon_path = os.path.join(data_folder, cuprite_nevada_folder, "ROI.geojson")

    polygon = gpd.read_file(polygon_path)
    raster = clip_raster(raster, polygon)

    raster = preprocess(raster)  # TODO

    ref_spectrum = get_spectrum(mineral_name="kaolinite")
    ref_spectrum = spectrum_preprocess(ref_spectrum, raster)

    plt.plot(ref_spectrum.wavelength, ref_spectrum.reflectance)

    # Normalize by continuum removal
    cr_datacube = continuum_removal(raster)

    sam_score = spectralMatch(raster, ref_spectrum, method="sam")  # TODO

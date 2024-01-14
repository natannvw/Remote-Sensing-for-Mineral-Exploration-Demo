import os
from typing import List, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch
from numba import jit, prange
from pysptools.spectro import convex_hull_removal
from scipy.spatial import ConvexHull
from torchmetrics.image import SpectralAngleMapper

from Raster import Raster
from Spectrum import Spectrum
from vector_utils import clip_raster


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
    bands, rows, cols = raster.datacube.shape
    reshaped_data = np.reshape(raster.datacube.transpose(1, 2, 0), (rows * cols, bands))

    for index in range(rows * cols):
        # for i in range(1):
        # i, j = 500, 500
        # index = i * cols + j
        # raster.datacube[:, i, j], _, _ = convex_hull_removal(
        #     raster.datacube[:, i, j], raster.wavelength
        # )
        reshaped_data[index, :], _, _ = convex_hull_removal(
            reshaped_data[index, :],
            raster.wavelength,
        )

    # Reshape back to original shape
    raster.datacube = np.reshape(reshaped_data, (rows, cols, bands)).transpose(2, 0, 1)

    return raster


def spectralMatch(raster: Raster, ref_spectrum: Spectrum, method: str = "sam") -> float:
    if method == "sam":
        raster.datacube = torch.tensor(raster.datacube, dtype=torch.float32)
        ref_spectrum.reflectance = torch.tensor(
            ref_spectrum.reflectance, dtype=torch.float32
        )
        bands, rows, cols = raster.datacube.shape

        sam = SpectralAngleMapper(reduction="none")

        score = sam(
            raster.datacube.unsqueeze(0),
            ref_spectrum.reflectance.unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, rows, cols),
        )

        # Get the array inside the tensor
        score_array = score.numpy().squeeze()

    return score_array


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
    raster.rescale()

    raster.wavelength = nm2um(raster.wavelength)

    raster = removeBands(raster, "Wavelength", [1, 2.5])

    return raster


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


def get_rgb_indices(raster):
    rgb_wavelengths = [620, 550, 450]  # Replace with the actual RGB wavelengths

    rgb_indices = []
    for wavelength in rgb_wavelengths:
        index = np.abs(raster.wavelength - wavelength).argmin()
        rgb_indices.append(index)

    return rgb_indices


if __name__ == "__main__":
    data_folder = "Data"
    cuprite_nevada_folder = "Cuprite Nevada"

    filename = "ENMAP01-____L2A-DT0000025905_20230707T192008Z_001_V010303_20230922T131734Z-SPECTRAL_IMAGE.TIF"

    raster_path = os.path.join(data_folder, cuprite_nevada_folder, filename)

    # wavelength =
    raster = Raster(path=raster_path)

    polygon_path = os.path.join(data_folder, cuprite_nevada_folder, "ROI.geojson")

    polygon = gpd.read_file(polygon_path)
    raster = clip_raster(raster, polygon)

    raster = preprocess(raster)

    ref_spectrum = Spectrum(mineral_name="kaolinite")
    ref_spectrum.preprocess(desired_wavelengths=raster.wavelength)

    plt.plot(ref_spectrum.wavelength, ref_spectrum.reflectance)

    # Normalize by continuum removal
    # from PerformanceMonitor import PerformanceMonitor

    # perf_monitor = PerformanceMonitor()
    # perf_monitor.start()
    # cr_datacube = continuum_removal(raster)   # TODO need to be parallelized
    # perf_monitor.stop()

    sam_score = spectralMatch(raster, ref_spectrum, method="sam")  # TODO
    threshold = 0.07
    masked_sam_score = np.ma.masked_greater(sam_score, threshold)

    # Plot the results
    # RGB from the hyperspectral image
    raster_for_rgb = Raster(path=raster_path)
    raster_for_rgb = clip_raster(raster_for_rgb, polygon)

    rgb_indices = get_rgb_indices(raster_for_rgb)

    red = raster_for_rgb.datacube[rgb_indices[0], :, :]
    green = raster_for_rgb.datacube[rgb_indices[1], :, :]
    blue = raster_for_rgb.datacube[rgb_indices[2], :, :]

    raster_for_rgb.datacube = np.stack([red, green, blue])

    # Normalize the RGB datacube
    raster_data = raster_for_rgb.datacube
    raster_data = raster_data.astype(float)  # Convert to float for normalization
    raster_data /= raster_data.max()  # Normalize to 0-1

    # Rearrange the axes to (height, width, channels)
    rgb_image = np.transpose(raster_data, (1, 2, 0))

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.imshow(masked_sam_score, cmap="turbo_r")

    target_spec = raster.datacube[:, 230, 260]
    plt.figure()
    plt.plot(raster.wavelength, target_spec)
    plt.plot(raster.wavelength, ref_spectrum.reflectance)
    plt.legend(["Target", "Reference Spectrum - Kaolinite"])
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Reflectance")

    # TODO map with basemap

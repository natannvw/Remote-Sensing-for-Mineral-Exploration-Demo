import numpy as np


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

    def rescale(self: "Raster"):
        gains, offsets = self.get_gains_and_offsets()

        self.datacube = self.datacube * gains[:, None, None] + offsets[:, None, None]

        return self

    def get_gains_and_offsets(self: "Raster"):
        band_characterisation = self.metadata["level_X"]["specific"][
            "bandCharacterisation"
        ]

        band_ids = band_characterisation["bandID"]

        # Extracting gain values
        gains = [band["GainOfBand"] for band in band_ids]

        # Extracting offset values
        offsets = [band["OffsetOfBand"] for band in band_ids]

        return np.array(gains, dtype=np.float32), np.array(offsets, dtype=np.float32)

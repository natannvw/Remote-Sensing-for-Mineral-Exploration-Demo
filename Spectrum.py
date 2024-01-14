import numpy as np
import pandas as pd


class Spectrum:
    def __init__(self, mineral_name: str = None):
        self.mineral_name = mineral_name

        if self.mineral_name:
            if self.mineral_name == "kaolinite":  # TODO add more minerals
                path = r"Spectral Library\Kaolinite.txt"
                df = pd.read_csv(
                    path,
                    skiprows=2,
                    header=None,
                    names=["wavelength", self.mineral_name],
                )

                self._wavelength = df["wavelength"].to_numpy()
                self._reflectance = df[self.mineral_name].to_numpy()

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

    def resample(self, desired_wavelengths):
        resampled = np.interp(desired_wavelengths, self.wavelength, self.reflectance)
        self.wavelength = desired_wavelengths
        self.reflectance = resampled

        return self

    def preprocess(self, desired_wavelengths):
        self.resample(desired_wavelengths=desired_wavelengths)
        # self = removeBands(self, "Wavelength", [1, 2.5]) # dont need because it was resampled to raster.wavelength already
        return self

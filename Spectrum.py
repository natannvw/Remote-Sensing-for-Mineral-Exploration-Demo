import numpy as np
import pandas as pd


class Spectrum:
    """
    A class representing a spectrum of a mineral.

    Args:
        mineral_name (str, optional): The name of the mineral. Defaults to None.

    Attributes:
        mineral_name (str): The name of the mineral.
        wavelength (ndarray): The array of wavelengths.
        reflectance (ndarray): The array of reflectance values.

    Methods:
        resample(desired_wavelengths): Resamples the spectrum to desired wavelengths.
        preprocess(desired_wavelengths): Preprocesses the spectrum by resampling to desired wavelengths.

    Usage Example:
        # Create a Spectrum object for kaolinite mineral
        spectrum = Spectrum(mineral_name="kaolinite")

        # Resample the spectrum to desired wavelengths
        desired_wavelengths = [400, 500, 600, 700, 800]
        spectrum.resample(desired_wavelengths)

        # Preprocess the spectrum by resampling and removing unwanted bands
        spectrum.preprocess(desired_wavelengths)
    """

    def __init__(self, mineral_name: str = None):
        self.mineral_name = mineral_name
        self._wavelength = None
        self._reflectance = None

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
        """
        Resamples the spectrum to desired wavelengths.

        Args:
            desired_wavelengths (list): The list of desired wavelengths.

        Returns:
            self (Spectrum): The resampled Spectrum object.
        """
        resampled = np.interp(desired_wavelengths, self.wavelength, self.reflectance)
        self.wavelength = desired_wavelengths
        self.reflectance = resampled

        return self

    def preprocess(self, desired_wavelengths):
        """
        Preprocesses the spectrum by resampling to desired wavelengths.

        Args:
            desired_wavelengths (list): The list of desired wavelengths.

        Returns:
            self (Spectrum): The preprocessed Spectrum object.
        """
        self.resample(desired_wavelengths=desired_wavelengths)
        # self = removeBands(self, "Wavelength", [1, 2.5]) # dont need because it was resampled to raster.wavelength already
        return self

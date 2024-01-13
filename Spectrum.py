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

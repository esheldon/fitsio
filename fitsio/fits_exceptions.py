class FITSFormatError(Exception):
    """
    Format error in FITS file
    """
    def __init__(self, value):
        super(FITSFormatError, self).__init__(value)
        self.value = value

    def __str__(self):
        return str(self.value)

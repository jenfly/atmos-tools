class InputError(Exception):
    """
    Exception raised for errors in the input.

    Attributes:
        msg : explanation of the error
    """

    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

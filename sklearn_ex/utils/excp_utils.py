

class phMLNotImplError(RuntimeError):
    """Custom exception to capture algorithm not implemented errors."""
    pass

class phMLRequestNotValidError(RuntimeError):
    """Custom exception to capture request not valid errors."""
    pass

class phMLAppServerAccessError(RuntimeError):
    """Custom exception to capture AppServer access errors."""
    pass

class phMLPreCheckError(RuntimeError):
    """Custom exception to capture input data pre-check errors."""
    pass


if __name__ == '__main__':
    pass
import sys
import hoomd


def get_device() -> hoomd.device:
    try:
        device = hoomd.device.GPU()
        print(device.get_available_devices(), device.is_available())
    except:
        device = hoomd.device.CPU()
        print('No GPU found, using CPU')

    return device


def is_debug() -> bool:
    """
    Function to see if the script is running in debug mode.
    """
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True

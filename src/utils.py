import sys
import hoomd


def get_device(notice_level=3):
    """
    Initialise HOOMD on the CPU or GPU, based on availability
    """

    try:
        device = hoomd.device.GPU(notice_level=notice_level)

        print("HOOMD is running on the following GPU(s):")
        print(device.device)

    except RuntimeError:
        device = hoomd.device.CPU(notice_level=notice_level)

        print("HOOMD is running on the CPU")

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

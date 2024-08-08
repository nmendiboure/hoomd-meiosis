import sys
import hoomd
import gsd.hoomd


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


def get_gsd_snapshot(snap):
    """
    Convert HOOMD snapshots to assignable GSD snapshots
    """

    snap_gsd = gsd.hoomd.Frame()

    for attr in snap_gsd.__dict__:
        data_gsd = getattr(snap_gsd, attr)

        if hasattr(snap, attr):
            data = getattr(snap, attr)

            if hasattr(data_gsd, '__dict__'):
                for prop in data_gsd.__dict__:
                    if hasattr(data, prop):
                        setattr(data_gsd, prop, getattr(data, prop))

    return snap_gsd


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

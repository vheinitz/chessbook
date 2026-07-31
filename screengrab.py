"""
Silent screen capture.

PIL's ``ImageGrab.grab()`` has no native backend on Linux: it shells out to
``gnome-screenshot``, which makes the whole screen flash white and plays the
camera shutter sound on every single capture - and it takes about a second.

This module captures without flash and without sound.  Backends are tried in
order and the first one that works is remembered:

1. GNOME Shell's screenshot D-Bus API.  Works on Wayland *and* X11 sessions,
   takes ~0.1 s, and the flash is an explicit parameter which we set to False.
2. Pillow's built-in X11/XCB grabber (``xdisplay=""``).  Silent, but only sees
   X11 windows, so it is useless on a Wayland session.
3. Plain ``ImageGrab.grab()`` - the flashing, beeping fallback, used only when
   nothing better is available.
"""

import cv2
import numpy as np
from PIL import ImageGrab

# D-Bus RequestName reply codes / flags
_DBUS_NAME_FLAG_DO_NOT_QUEUE = 4
_DBUS_REQUEST_NAME_REPLY_PRIMARY_OWNER = 1
_DBUS_REQUEST_NAME_REPLY_ALREADY_OWNER = 4


class GnomeShellGrabber:
    """Screenshots through GNOME Shell's D-Bus API - no flash, no sound."""

    name = "gnome-shell-dbus"

    def __init__(self):
        import gi
        gi.require_version("Gio", "2.0")
        from gi.repository import Gio, GLib

        self._Gio = Gio
        self._GLib = GLib
        self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)

        # Fail fast if the interface is not there at all.
        self._bus.call_sync(
            "org.gnome.Shell", "/org/gnome/Shell/Screenshot",
            "org.freedesktop.DBus.Introspectable", "Introspect", None,
            GLib.VariantType("(s)"), Gio.DBusCallFlags.NONE, 5000, None)

    def _own_screenshot_name(self, acquire):
        """
        Acquire/release the ``org.gnome.Screenshot`` bus name.

        GNOME Shell serves the screenshot interface only to a few well known
        callers (the media keys daemon, the desktop portal, gnome-screenshot).
        Owning gnome-screenshot's name puts us on that list.  The name is held
        only for the duration of a capture so the real gnome-screenshot keeps
        working while this program runs.
        """
        method = "RequestName" if acquire else "ReleaseName"
        args = (self._GLib.Variant("(su)", ("org.gnome.Screenshot",
                                            _DBUS_NAME_FLAG_DO_NOT_QUEUE))
                if acquire else self._GLib.Variant("(s)", ("org.gnome.Screenshot",)))
        reply = self._bus.call_sync(
            "org.freedesktop.DBus", "/org/freedesktop/DBus",
            "org.freedesktop.DBus", method, args,
            self._GLib.VariantType("(u)"), self._Gio.DBusCallFlags.NONE, 5000, None)
        return reply.unpack()[0]

    def grab(self, path):
        code = self._own_screenshot_name(True)
        if code not in (_DBUS_REQUEST_NAME_REPLY_PRIMARY_OWNER,
                        _DBUS_REQUEST_NAME_REPLY_ALREADY_OWNER):
            raise RuntimeError("org.gnome.Screenshot is held by another process")
        try:
            result = self._bus.call_sync(
                "org.gnome.Shell", "/org/gnome/Shell/Screenshot",
                "org.gnome.Shell.Screenshot", "Screenshot",
                # include_cursor=False, flash=False -> silent and invisible
                self._GLib.Variant("(bbs)", (False, False, path)),
                self._GLib.VariantType("(bs)"), self._Gio.DBusCallFlags.NONE,
                10000, None)
        finally:
            self._own_screenshot_name(False)

        success, _ = result.unpack()
        if not success:
            raise RuntimeError("GNOME Shell refused to take the screenshot")

        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"screenshot was not written to {path}")
        return image


class PillowX11Grabber:
    """Pillow's XCB grabber - silent, but blind to Wayland-native windows."""

    name = "pillow-x11"

    def __init__(self):
        self._probe()

    def _probe(self):
        image = ImageGrab.grab(xdisplay="")
        if image.size[0] < 2 or image.size[1] < 2:
            raise RuntimeError("X11 grab returned an empty image")

    def grab(self, path):
        image = np.array(ImageGrab.grab(xdisplay="").convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
        return image


class PillowDefaultGrabber:
    """Whatever PIL does by default - on GNOME this flashes and beeps."""

    name = "pillow-default"

    def grab(self, path):
        image = np.array(ImageGrab.grab().convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
        return image


_BACKENDS = (GnomeShellGrabber, PillowX11Grabber, PillowDefaultGrabber)

_grabber = None
_next_backend = 0


def _next_grabber(verbose):
    """Instantiates the next backend that is available on this system."""
    global _next_backend

    while _next_backend < len(_BACKENDS):
        factory = _BACKENDS[_next_backend]
        _next_backend += 1
        try:
            grabber = factory()
        except Exception as exc:
            if verbose:
                print(f"screengrab: {factory.name} unavailable ({exc})")
            continue
        if verbose:
            print(f"screengrab: using {grabber.name}")
        return grabber

    raise RuntimeError("no screen capture backend available")


def grab_screen(path, verbose=True):
    """
    Captures the whole screen, writes it to ``path`` and returns it as a BGR
    OpenCV image.  A backend that starts failing is dropped for good and the
    next one in the list takes over.
    """
    global _grabber

    while True:
        if _grabber is None:
            _grabber = _next_grabber(verbose)
        try:
            return _grabber.grab(path)
        except Exception as exc:
            print(f"screengrab: {_grabber.name} failed ({exc})")
            _grabber = None


if __name__ == "__main__":
    import time

    start = time.time()
    img = grab_screen("/tmp/screengrab_test.png")
    print(f"captured {img.shape} in {time.time() - start:.2f}s "
          f"-> /tmp/screengrab_test.png")

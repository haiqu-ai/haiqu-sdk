"""Haiqu matplotlib style utilities."""

import importlib.resources as _res


def set_haiqu_mpl_style(font_path: str | None = None) -> None:
    """Apply the Haiqu matplotlib style to the current session.

    When ``haiqu.sdk`` is imported inside a Jupyter environment this function
    is called automatically. Outside of Jupyter — for example in a script or a
    testing harness — call it explicitly after importing ``haiqu.sdk``.

    The function makes the following changes to the active matplotlib session:

    * **Colour cycle** — sets the default series colours to the Haiqu brand
      palette (Morpho Blue, Orange, Red, Light Blue, Black, Neutral Grey).
    * **Colormaps** — registers two named colormaps:

      * ``haiqu_blue`` — sequential, white → Light Blue → Morpho Blue.
      * ``haiqu`` — diverging, Orange → white → Morpho Blue.

      ``haiqu`` is set as the default image colormap (``image.cmap``).

    * **Typography** — if available, registers and activates provided font as the
      monospace typeface used for tick labels, axis labels, and captions. Typically
      this should be the *Suisse Int'l Mono* font, if it is available.
      The font HAIQU_MPL_FONT_PATH environment variable or the font_path arg should
      provide the path.
    * **Style** — applies the bundled ``haiqu.mplstyle`` sheet (minimal axes,
      no top/right spines, dashed grid, square scatter markers, high-DPI save).

    Args:
        font_path (str | None): The path to the font. Defaults to ``None``. If
                                ``font_path`` is not specified, the
                                ``HAIQU_MPL_FONT_PATH`` environment variable
                                is checked. If it is not set, the
                                matplotlib default font is used.

    Examples:
        Automatic application inside a notebook (nothing extra required):

        >>> import haiqu.sdk
        >>> import matplotlib.pyplot as plt
        >>> plt.plot([1, 2, 3])   # rendered with Haiqu style

        Explicit application in a script:

        >>> import haiqu.sdk
        >>> haiqu.sdk.set_haiqu_mpl_style()
        >>> plt.plot([1, 2, 3])   # rendered with Haiqu style

        Using the registered colormaps directly:

        >>> import numpy as np
        >>> plt.imshow(np.random.rand(10, 10))               # haiqu (default)
        >>> plt.imshow(np.random.rand(10, 10), cmap='haiqu_blue') # diverging
    """
    import matplotlib as _mpl
    import matplotlib.colors as _mcolors
    import matplotlib.font_manager as _fm
    import matplotlib.style as _mpl_style
    import os

    _font_file = font_path or os.environ.get("HAIQU_MPL_FONT_PATH")
    if _font_file:
        _fm.fontManager.addfont(_font_file)

    for _name, _stops in [
        ("haiqu_blue", ["#FFFFFF", "#C4D5FF", "#093188"]),
        ("haiqu", ["#FEA450", "#FFFFFF", "#093188"]),
    ]:
        _cmap = _mcolors.LinearSegmentedColormap.from_list(_name, _stops)
        if hasattr(_mpl, "colormaps"):
            _mpl.colormaps.register(_cmap, name=_name, force=True)
        else:
            import matplotlib.cm as _cm

            _cm.register_cmap(name=_name, cmap=_cmap)

    _mpl_style.use(str(_res.files("haiqu.sdk").joinpath("haiqu.mplstyle")))
    _mpl.rcParams["image.cmap"] = "haiqu"


def unset_haiqu_mpl_style() -> None:
    """Restore matplotlib to its default style, undoing :func:`set_haiqu_mpl_style`.

    Resets all rcParams to the matplotlib defaults (equivalent to calling
    ``matplotlib.rcdefaults()``). The two brand colormaps (``haiqu_blue`` and
    ``haiqu``) remain available in the colormap registry and can still be
    referenced by name, but they are no longer the active defaults.

    Use this function when you need to produce a plot without the Haiqu style
    in the same session where ``haiqu.sdk`` was imported.

    Examples:
        Temporarily disable the Haiqu style for a single plot:

        >>> import haiqu.sdk
        >>> import matplotlib.pyplot as plt
        >>> haiqu.sdk.unset_haiqu_mpl_style()
        >>> plt.plot([1, 2, 3])  # rendered with matplotlib defaults

        Re-apply afterwards:

        >>> haiqu.sdk.set_haiqu_mpl_style()
        >>> plt.plot([1, 2, 3])  # rendered with Haiqu style again
    """
    import matplotlib as _mpl

    _mpl.rcdefaults()

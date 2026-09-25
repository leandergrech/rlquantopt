"""Figure helpers for the docs: save each panel of a multi-panel figure as its own image.

Wide figures with several panels side by side are unreadable on a page; the docs show the panels
stacked instead. ``save_panels`` crops each axes (with its title, labels and legend) out of the
already-drawn figure, so the panels stay identical to the combined figure.
"""
import os


def save_panels(fig, axes, path, dpi=150, pad=0.1):
    """Save ``axes`` of ``fig`` as ``<path stem>_a.png``, ``_b.png``, ...; returns the file names."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    stem = os.path.splitext(path)[0]
    out = []
    for i, ax in enumerate(axes):
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(pad)
        name = f"{stem}_{'abcdefgh'[i]}.png"
        fig.savefig(name, dpi=dpi, bbox_inches=bbox)
        out.append(name)
    return out

"""Printed-figure sizing for the Phase 3 matplotlib figures.

The matplotlib twin of results/_figure_theme.R, and it exists for the same
reason: these figures are drawn on a 5-15 in canvas and then placed into a
2-3 in slot in Google Slides, so 10 pt text prints at 1.5-2.8 pt on the page.

    printed pt = fontsize x placed width / figure width

Fonts are in points and do not shrink with the canvas, so the lever is the
canvas. Shrink the figure by K and the type grows relative to the plot by 1/K.

Three things scale differently, exactly as in the R module:

  * TEXT (ticks, labels, titles, legend)  -- unchanged, held at print size;
    that is the whole point.
  * MARKS -- bars and heatmap cells are in data/axes coordinates and rescale
    on their own, so unlike ggplot there is nothing to do. Only genuinely
    absolute things (line widths) take K.
  * IN-PLOT ANNOTATIONS (the numbers written inside heatmap cells) sit inside
    the data geometry, which shrank; at full size they collide. They take
    sqrt(K), same compromise as geom_text.

HEIGHT: the furniture -- tick labels, title, colourbar, legend -- is a fixed
number of inches whatever K is, so shrinking takes the height out of the axes
and squashes the plot. `stretch_height` gives it back, measured from the
rendered figure rather than assumed.
"""
import matplotlib.pyplot as plt

K = 0.55           # canvas multiplier. 1.0 = the original figure, exactly.
STRETCH_MAX = 1.60


def scaled_figsize(w_in, h_in, k=None):
    k = K if k is None else k
    """The canvas to draw on, before any height correction."""
    return (w_in * k, h_in * k)


def annot_size(base_pt, k=None):
    """In-cell annotation size.

    A matplotlib font is absolute points on a canvas that just shrank by k, so
    left alone it is already 1/k bigger RELATIVE to the cell it sits in. We want
    in-cell numbers to grow by only 1/sqrt(k) -- the same compromise geom_text
    gets in the R module -- so multiply by sqrt(k):

        relative size = base*sqrt(k) / (W*k) = (base/W) / sqrt(k)
    """
    k = K if k is None else k
    return base_pt * (k ** 0.5)


def stretch_height(fig, h_in_scaled, k=None, cap=STRETCH_MAX):
    """Extra height factor that keeps the AXES the size they were on the page.

    Measures what the figure actually gave to axes vs furniture, then applies
        stretch = 1 + (F/h)(1/k - 1)
    with F the furniture height in inches -- the same closed form the R module
    uses. Call after the axes exist and tight_layout() has run.
    """
    k = K if k is None else k
    fig.canvas.draw()
    fig_h = fig.get_size_inches()[1]
    axes_h = sum(ax.get_position().height for ax in fig.axes if ax.get_visible())
    # get_position is a fraction of the figure; collapse overlapping rows by
    # taking the union extent instead of the sum when axes share rows
    tops = [ax.get_position().y1 for ax in fig.axes if ax.get_visible()]
    bots = [ax.get_position().y0 for ax in fig.axes if ax.get_visible()]
    if not tops:
        return 1.0
    used = (max(tops) - min(bots)) * fig_h            # inches of drawing area
    F = max(0.0, fig_h - used)                        # furniture, inches
    h_orig = h_in_scaled / k                          # the original height
    if F <= 0 or F >= h_orig:
        return 1.0
    return min(cap, max(1.0, 1 + (F / h_orig) * (1 / k - 1)))


def report(name, w0, h0, w1, h1, placed_mm=None):
    line = (f"  [fig] {name:38s} {w0:5.2f}x{h0:<5.2f} -> {w1:5.2f}x{h1:<5.2f} in "
            f"(k {K:.2f})  aspect {h0/w0:.3f} -> {h1/w1:.3f}")
    if placed_mm:
        pl = placed_mm / 25.4
        line += f"   10 pt: {10*pl/w0:.1f} -> {10*pl/w1:.1f} pt on the page"
    print(line)

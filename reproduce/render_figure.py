#!/usr/bin/env python3
"""Run one figure build script and write its outputs into OUT_DIR.

The build scripts (code/manuscript/publication_manuscript/figures/...) import the
vendored scipilot figure helpers and expect Liberation Sans, the Arial-metric font
every figure is set in. This wrapper makes both available, then runs the script
with OUT_DIR as the working directory, because export_figure() writes there.

Liberation Sans is found, in order, from LIBERATION_FONTS (a folder of .ttf files),
the fonts bundled with R's fontLiberation package, or an installed system font.

The grayscale proof is made from the PNG already written. (The helper's own
version re-saves the PNG with a 0.1 in pad, which would make it 60 px larger than
the published file.)

Usage:  python3 render_figure.py <build_script.py> <OUT_DIR>
"""
import glob
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "code/manuscript/publication_manuscript/figures/_vendor/scipilot"
FONT_GLOBS = [
    os.path.join(os.environ.get("LIBERATION_FONTS", "/nonexistent"), "LiberationSans-*.ttf"),
    "/Library/Frameworks/R.framework/Versions/*/Resources/library/fontLiberation/fonts/liberation-fonts/LiberationSans-*.ttf",
    "/usr/lib/R/site-library/fontLiberation/fonts/liberation-fonts/LiberationSans-*.ttf",
    "/usr/local/lib/R/site-library/fontLiberation/fonts/liberation-fonts/LiberationSans-*.ttf",
    os.path.expanduser("~/R/*/*/fontLiberation/fonts/liberation-fonts/LiberationSans-*.ttf"),
]


def main():
    script, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SCIPILOT_SCRIPTS", str(VENDOR))
    sys.path.insert(0, os.environ["SCIPILOT_SCRIPTS"])
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    files = sorted({f for g in FONT_GLOBS for f in glob.glob(g)})
    for f in files:
        font_manager.fontManager.addfont(f)
    if not files and not any("Liberation Sans" in f.name for f in font_manager.fontManager.ttflist):
        sys.exit("Liberation Sans not found: install it, or set LIBERATION_FONTS, "
                 "or install the R package fontLiberation")

    import export_figure

    def grayscale_from_png(fig, basename, dpi):
        from PIL import Image
        gray = f"{basename}_grayscale.png"
        Image.open(f"{basename}.png").convert("L").save(gray)
        return gray

    export_figure._grayscale_from = grayscale_from_png
    os.chdir(out)
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

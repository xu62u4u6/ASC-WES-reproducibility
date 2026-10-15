"""Compose S20 from saved 100-seed means using the checked manuscript layout.

Pass --source (completed importance directory) and --output (new staging path).
Uses the existing composition script without fitting models or changing data.
"""

import os
from pathlib import Path
import runpy
import tempfile


if __name__ == "__main__":
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="asc-s20-mpl-")
    script = Path(__file__).resolve().parents[1] / "render_s20_manuscript.py"
    runpy.run_path(str(script), run_name="__main__")

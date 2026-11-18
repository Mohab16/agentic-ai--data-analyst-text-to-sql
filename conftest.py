"""Root conftest: redirect pytest's temporary directories to a writable path.

Some CI environments restrict access to the system temp folder, causing pytest's
`tmp_path` fixture to raise PermissionError. By setting the standard temp
environment variables and monkey‑patching ``tempfile`` before any pytest code
runs we guarantee a writable place for all temporary files.
"""

import os
from pathlib import Path
import tempfile

# Choose a location inside the repository – `tmp` at the workspace root.
# Create it if missing.
TMP_ROOT = Path(__file__).resolve().parent / "tmp"
TMP_ROOT.mkdir(exist_ok=True)

# Override the environment variables if they are not already set to a safe value.
# Using `setdefault` keeps any explicit user value.
os.environ.setdefault("TMPDIR", str(TMP_ROOT))
os.environ.setdefault("TEMP", str(TMP_ROOT))
os.environ.setdefault("TMP", str(TMP_ROOT))
os.environ.setdefault("PYTEST_TMPDIR", str(TMP_ROOT))

# Ensure the tempfile module also uses our temporary directory. This is important
# because pytest may call ``tempfile.gettempdir()`` before our environment variables
# are applied.
tempfile.tempdir = str(TMP_ROOT)
tempfile.gettempdir = lambda: str(TMP_ROOT)
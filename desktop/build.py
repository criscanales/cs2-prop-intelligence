"""Build a double-click desktop app with PyInstaller (run on the target OS: Windows -> .exe, macOS -> .app).

pip install -r requirements.txt pyinstaller pywebview
python desktop/build.py
"""
import os
import sys
from pathlib import Path

import PyInstaller.__main__

root = Path(__file__).resolve().parents[1]
sep = os.pathsep
args = [
    str(root / "cs2model" / "desktop.py"),
    "--name", "CS2 Props",
    "--windowed",
    "--noconfirm",
    "--paths", str(root),
    "--add-data", f"{root / 'cs2model'}{sep}cs2model",
    "--add-data", f"{root / 'config'}{sep}config",
    "--collect-all", "streamlit",
    "--collect-all", "plotly",
    "--collect-data", "lightgbm",
    "--collect-binaries", "lightgbm",
    "--collect-data", "cv2",  # Haar face cascade used for portrait framing
    "--copy-metadata", "streamlit",
    "--collect-submodules", "cs2model",
    "--collect-submodules", "webview",
    "--hidden-import", "sklearn.linear_model",
    "--hidden-import", "statsmodels.api",
    "--distpath", str(root / "dist"),
    "--workpath", str(root / "build"),
    "--specpath", str(root / "build"),
]
if sys.platform == "darwin":
    args += ["--osx-bundle-identifier", "com.cs2props.app"]
PyInstaller.__main__.run(args)
print("built:", root / "dist")

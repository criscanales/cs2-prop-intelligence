"""Desktop launcher: runs the Streamlit dashboard in a background process and shows it in a native window.

python -m cs2model.desktop          (from source)
"CS2 Props" app / exe               (PyInstaller build, see desktop/build.py)
"""
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
BUNDLE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
HOME = Path(os.environ.get("CS2_HOME") or (Path.home() / "CS2Props" if FROZEN else BUNDLE))


def _setup_home():
    os.environ["CS2_HOME"] = str(HOME)
    for sub in ("config", "data", "models"):
        (HOME / sub).mkdir(parents=True, exist_ok=True)
    for f in (BUNDLE / "config").glob("*.yaml"):  # first run: copy editable config next to the user's data
        if not (HOME / "config" / f.name).exists():
            shutil.copy(f, HOME / "config" / f.name)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(port: int):
    from streamlit.web import bootstrap
    flags = {"server.port": port, "server.address": "127.0.0.1", "server.headless": True,
             "global.developmentMode": False, "browser.gatherUsageStats": False,
             "theme.base": "dark", "theme.primaryColor": "#22c55e", "theme.backgroundColor": "#0b0f14",
             "theme.secondaryBackgroundColor": "#141a22", "theme.textColor": "#e5e7eb"}
    bootstrap.load_config_options(flag_options=flags)
    bootstrap.run(str(BUNDLE / "cs2model" / "dashboard" / "app.py"), False, [], flags)


def _wait(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if urllib.request.urlopen(url + "/_stcore/health", timeout=2).status == 200:
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main():
    import multiprocessing
    multiprocessing.freeze_support()  # frozen app: pricing worker processes re-enter here
    _setup_home()
    if "--serve" in sys.argv:
        serve(int(sys.argv[sys.argv.index("--serve") + 1]))
        return
    port = _free_port()
    cmd = [sys.executable] + ([] if FROZEN else ["-m", "cs2model.desktop"]) + ["--serve", str(port)]
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}  # CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, cwd=str(BUNDLE), env=os.environ.copy(), **kw)
    url = f"http://127.0.0.1:{port}"
    try:
        if not _wait(url):
            raise SystemExit("dashboard failed to start")
        try:
            import webview
            webview.create_window("CS2 Props +EV", url, width=1440, height=900, min_size=(1000, 650))
            webview.start()
        except Exception:  # noqa: BLE001  no native webview available: fall back to the default browser
            webbrowser.open(url)
            proc.wait()
    finally:
        proc.terminate()


def _bundle_hints():  # never called; makes PyInstaller bundle what the Streamlit script imports at runtime
    import cs2model.cli  # noqa: F401
    import cs2model.dashboard  # noqa: F401
    import cs2model.pipeline  # noqa: F401
    import cs2model.training.train  # noqa: F401
    import plotly.graph_objects  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    import statsmodels.api  # noqa: F401


if __name__ == "__main__":
    main()

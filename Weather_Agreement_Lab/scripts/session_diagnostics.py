import sys
import os
import platform
import subprocess
from datetime import datetime
import pkg_resources


def run_session_diagnostics():

    print("===== SESSION DIAGNOSTICS =====\n")

    print("Timestamp:", datetime.now())

    print("\n--- Python ---")
    print("Python version:", sys.version)
    print("Python executable:", sys.executable)

    print("\n--- Virtual Environment ---")
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))

    print("\n--- Working Directory ---")
    print("Current working directory:", os.getcwd())

    print("\n--- Platform ---")
    print("System:", platform.system())
    print("Release:", platform.release())
    print("Machine:", platform.machine())
    print("Processor:", platform.processor())

    print("\n--- Installed Key Packages ---")

    for pkg in ["pandas", "requests", "python-dotenv"]:
        try:
            version = pkg_resources.get_distribution(pkg).version
            print(f"{pkg}: {version}")
        except Exception:
            print(f"{pkg}: NOT INSTALLED")

    print("\n--- Git Status ---")

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True
        )
        print("Git branch:", result.stdout.strip())

    except Exception:
        print("Git not available")

    print("\n==============================")


"""Shared pytest setup: DLL location, golden-file options."""
import os
import sys
from pathlib import Path

import pytest

PYTHON_DIR = Path(__file__).resolve().parents[1]
REPO_PROJECT_DIR = PYTHON_DIR.parent  # .../CuSuperHelium (holds the .sln)
DEFAULT_DLL = REPO_PROJECT_DIR / "x64" / "Release" / "CuSuperHelium.dll"

# Temporary until the Python package exists (reorganization Phase 6).
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Overwrite the golden files with the current DLL output instead of comparing.",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Visual tests open windows only with CUSH_SHOW_PLOTS=1; otherwise render off-screen.
    if os.environ.get("CUSH_SHOW_PLOTS") != "1":
        import matplotlib

        matplotlib.use("Agg")
    config.addinivalue_line("markers", "visual: test that also produces a figure for a person to check")


@pytest.fixture(scope="session")
def dll_path() -> Path:
    path = Path(os.environ.get("CUSUPERHELIUM_DLL", DEFAULT_DLL))
    if not path.is_file():
        pytest.fail(f"CuSuperHelium.dll not found at {path}. Build it with scripts/build.ps1 or set CUSUPERHELIUM_DLL.")
    return path


@pytest.fixture(scope="session")
def sim(dll_path: Path):
    import integration.rhs as rhs

    return rhs.SimulationManager(str(dll_path))


@pytest.fixture(scope="session")
def update_goldens(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-goldens"))

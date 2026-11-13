"""Shared pytest setup: DLL location, golden-file options."""
import os
from pathlib import Path

import pytest

try:
    import cusuperhelium
except ImportError as error:  # pragma: no cover - only without the package installed
    raise ImportError("The cusuperhelium package is not installed. From the repository root run: "
                      "pip install -e CuSuperHelium/Python[test]") from error


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
    """CUSUPERHELIUM_DLL, otherwise this checkout's build output (see cusuperhelium.library)."""
    try:
        return cusuperhelium.find_dll()
    except FileNotFoundError as error:
        pytest.fail(str(error))


@pytest.fixture(scope="session")
def sim(dll_path: Path):
    return cusuperhelium.SimulationManager(dll_path)


@pytest.fixture(scope="session")
def update_goldens(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-goldens"))

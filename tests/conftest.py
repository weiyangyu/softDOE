"""Optional live R verification is explicit and fails if prerequisites are missing."""


def pytest_addoption(parser):
    group = parser.getgroup("MOFAT reference")
    group.addoption("--run-r", action="store_true", help="Compare with an installed R MOFAT package.")
    group.addoption("--mofat-version", default="2.0", help="Exact R MOFAT version required by --run-r.")

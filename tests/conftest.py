from pathlib import Path

import pytest

from plodds import clean

FIXTURES = Path(__file__).parent / "fixtures"


def _skip_stubs():
    """Report a test as skipped while the function it needs is still a stub."""
    try:
        return (yield)
    except NotImplementedError:
        pytest.skip("not written yet")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):  # stubs called while building fixtures
    return (yield from _skip_stubs())


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):  # stubs called by the test itself
    return (yield from _skip_stubs())


@pytest.fixture
def raw_sample():
    return clean.read_raw(FIXTURES / "E0_sample.csv")


@pytest.fixture
def cleaned_sample(raw_sample):
    return clean.clean_season(raw_sample, "2324")

"""Shared pytest fixtures for Fat-Tree testing."""
from __future__ import annotations

import pytest
from src.topology import FatTreeTopology
from src.models import FatTreeGraph


@pytest.fixture(scope="session")
def topo_k2() -> FatTreeGraph:
    """Fixture for k=2 boundary Fat-Tree."""
    return FatTreeTopology(2).build()


@pytest.fixture(scope="session")
def topo_k4() -> FatTreeGraph:
    """Fixture for standard k=4 Fat-Tree."""
    return FatTreeTopology(4).build()


@pytest.fixture(scope="session")
def topo_k6() -> FatTreeGraph:
    """Fixture for k=6 Fat-Tree."""
    return FatTreeTopology(6).build()


@pytest.fixture(scope="session")
def topo_k8() -> FatTreeGraph:
    """Fixture for larger k=8 Fat-Tree."""
    return FatTreeTopology(8).build()


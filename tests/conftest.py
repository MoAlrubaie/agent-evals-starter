from pathlib import Path

import pytest

from agent_evals.cases import load_world

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def world_base():
    return load_world(ROOT / "evals" / "world.yaml")


@pytest.fixture
def cases_dir():
    return ROOT / "evals" / "cases"

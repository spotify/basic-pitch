import pytest
import json
import pathlib
from unittest import mock

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"
GUITAR_SET_TEST_INDEX = json.load(open(RESOURCES_PATH / "data" / "guitarset" / "dummy_index.json"))
IKALA_TEST_INDEX = json.load(open(RESOURCES_PATH / "data" / "ikala" / "dummy_index.json"))
MAESTRO_TEST_INDEX = json.load(open(RESOURCES_PATH / "data" / "maestro" / "dummy_index.json"))
METADATA_TEST_INDEX = json.load(open(RESOURCES_PATH / "data" / "maestro" / "maestro-v2.0.0.json"))
MEDLEYDB_PITCH_TEST_INDEX = json.load(open(RESOURCES_PATH / "data" / "medleydb_pitch" / "dummy_index.json"))
SLAKH_TEST_INDEX = json.load(open(RESOURCES_PATH / "data" / "slakh" / "dummy_index.json"))


@pytest.fixture  # type: ignore[untyped-decorator]
def mock_slakh_index() -> None:  # type: ignore[misc]
    with mock.patch("mirdata.datasets.slakh.Dataset.download"):
        with mock.patch("mirdata.datasets.slakh.Dataset._index", new=SLAKH_TEST_INDEX):
            yield


@pytest.fixture  # type: ignore[untyped-decorator]
def mock_medleydb_pitch_index() -> None:  # type: ignore[misc]
    with mock.patch("mirdata.datasets.medleydb_pitch.Dataset.download"):
        with mock.patch("mirdata.datasets.medleydb_pitch.Dataset._index", new=MEDLEYDB_PITCH_TEST_INDEX):
            yield


@pytest.fixture  # type: ignore[untyped-decorator]
def mock_maestro_index() -> None:  # type: ignore[misc]
    index_with_metadata = MAESTRO_TEST_INDEX
    metadata = {mdata["midi_filename"].split(".")[0]: mdata for mdata in METADATA_TEST_INDEX}
    with mock.patch("mirdata.datasets.maestro.Dataset.download"):
        with mock.patch("mirdata.datasets.maestro.Dataset._metadata", new=metadata):
            with mock.patch("mirdata.datasets.maestro.Dataset._index", new=index_with_metadata):
                yield


@pytest.fixture  # type: ignore[untyped-decorator]
def mock_guitarset_index() -> None:  # type: ignore[misc]
    with mock.patch("mirdata.datasets.guitarset.Dataset.download"):
        with mock.patch("mirdata.datasets.guitarset.Dataset._index", new=GUITAR_SET_TEST_INDEX):
            yield


@pytest.fixture  # type: ignore[untyped-decorator]
def mock_ikala_index() -> None:  # type: ignore[misc]
    with mock.patch("mirdata.datasets.ikala.Dataset.download"):
        with mock.patch("mirdata.datasets.ikala.Dataset._index", new=IKALA_TEST_INDEX):
            yield

from src.ingestion.schema import (
    EXPECTED_FIELD_COUNT, FIELD_NAMES, FIELD_NAME_TO_POS, get_field_names,
)


def test_field_count_is_52():
    assert EXPECTED_FIELD_COUNT == 52
    assert len(FIELD_NAMES) == 52


def test_key_field_positions():
    assert FIELD_NAME_TO_POS["event_id"] == 0
    assert FIELD_NAME_TO_POS["imsi"] == 6
    assert FIELD_NAME_TO_POS["event_time"] == 48


def test_get_field_names_matches():
    assert tuple(get_field_names()) == FIELD_NAMES

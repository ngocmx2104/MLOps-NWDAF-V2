import pytest


def make_ebs_line(event_id="l_handover", imsi="111", event_time="1719381600000", eci="10"):
    """Build one 52-field semicolon-delimited EBS line (positions per schema)."""
    fields = [""] * 52
    fields[0] = event_id       # event_id
    fields[6] = imsi           # imsi
    fields[12] = eci           # eci (cell)
    fields[48] = event_time    # event_time (epoch ms)
    fields[51] = "2024062614"  # date_hour
    return ";".join(fields)


@pytest.fixture
def tiny_ebs_file(tmp_path):
    """A 3-line raw EBS file: 2 handover events (same IMSI, 2 cells) + 1 non-handover."""
    lines = [
        make_ebs_line("l_handover", "111", "1719381600000", "10"),
        make_ebs_line("l_handover", "111", "1719381630000", "20"),
        make_ebs_line("l_service_request", "222", "1719381660000", "30"),
    ]
    p = tmp_path / "A20240626.1400+0700-20240626.1401+0700_840_ebs"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p

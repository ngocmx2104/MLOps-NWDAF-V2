import subprocess
import sys
from pathlib import Path

from tests.ingestion.conftest import make_ebs_line


def test_cli_build_snapshot_smoke(tmp_path):
    raw = tmp_path / "A20240626.1400+0700-20240626.1401+0700_840_ebs"
    raw.write_text("\n".join([
        make_ebs_line("l_handover", "111", "1719381600000", "10"),
        make_ebs_line("l_handover", "111", "1719381630000", "20"),
    ]) + "\n", encoding="utf-8")
    out = tmp_path / "snap"
    r = subprocess.run(
        [sys.executable, "-m", "src.ingestion.cli", "build-snapshot",
         "--output-dir", str(out), "--snapshot-id", "D1_SMOKE", str(raw)],
        capture_output=True, text=True, cwd=Path.cwd(),
    )
    assert r.returncode == 0, r.stderr
    assert (out / "D1_SMOKE.parquet").exists()
    assert (out / "D1_SMOKE_metadata.json").exists()

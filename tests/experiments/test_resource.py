import sys

from src.experiments.metrics.resource import measure_subprocess, storage_bytes


def test_measure_subprocess_allocates(tmp_path):
    # allocate ~20MB then exit 0
    code = "x = bytearray(20*1024*1024); import time; time.sleep(0.05)"
    res = measure_subprocess([sys.executable, "-c", code])
    assert res["returncode"] == 0
    assert res["wall_s"] > 0
    assert res["peak_rss_mb"] > 10  # saw the allocation


def test_storage_bytes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1000)
    assert storage_bytes(tmp_path) >= 1000

from src.data.ebs_generator import EBSSyntheticGenerator, SCENARIOS


def test_generate_dataset_manifest(tiny_profile, tmp_path):
    gen = EBSSyntheticGenerator(tiny_profile, random_state=42)
    manifest = gen.generate_dataset(tmp_path / "sudden", n_files=2,
                                    scenario=SCENARIOS["sudden_drift"])
    assert manifest["n_files"] == 2
    assert manifest["scenario"] == "sudden"
    assert len(manifest["files"]) == 2
    assert all("drift_active" in f and "handover_lines" in f for f in manifest["files"])
    assert len(list((tmp_path / "sudden").glob("*_ebs"))) == 2
    assert (tmp_path / "sudden" / "manifest.json").exists()
    assert (tmp_path / "sudden" / "ground_truth.json").exists()


def test_sudden_drift_flag_flips(tiny_profile, tmp_path):
    gen = EBSSyntheticGenerator(tiny_profile, random_state=42)
    manifest = gen.generate_dataset(tmp_path / "s", n_files=20,
                                    scenario=SCENARIOS["sudden_drift"])
    early = [f for f in manifest["files"] if f["file_idx"] < 15]
    late = [f for f in manifest["files"] if f["file_idx"] >= 15]
    assert not any(f["drift_active"] for f in early)
    assert all(f["drift_active"] for f in late)


def test_generated_ebs_parses_with_phase1_parser(tiny_profile, tmp_path):
    from src.ingestion.parser import normalize_timestamps, parse_ebs_files
    gen = EBSSyntheticGenerator(tiny_profile, random_state=42)
    gen.generate_dataset(tmp_path / "b", n_files=1, scenario=SCENARIOS["baseline"])
    files = list((tmp_path / "b").glob("*_ebs"))
    df = normalize_timestamps(parse_ebs_files(files))
    assert (df["raw_field_count"] == 52).all()
    assert (df["event_id"] == "l_handover").any()
    assert df["event_ts"].notna().all()

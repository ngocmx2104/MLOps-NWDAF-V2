from src.monitoring.cli import build_parser, cmd_check


def test_parser_has_check_subcommand():
    args = build_parser().parse_args(["check", "--reference", "r.parquet", "--predictions", "p.jsonl"])
    assert args.command == "check" and args.reference == "r.parquet"


def test_cmd_check_reports_drift(reference_parquet, predictions_drift, capsys):
    import json
    args = build_parser().parse_args(
        ["check", "--reference", str(reference_parquet), "--predictions", str(predictions_drift),
         "--min-features-alert", "3"])
    cmd_check(args)
    data = json.loads(capsys.readouterr().out)
    assert data["drift_detected"] is True

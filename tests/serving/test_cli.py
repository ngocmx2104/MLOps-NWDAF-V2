from src.serving.cli import build_config_from_args, build_parser


def test_parser_has_serve_subcommand():
    parser = build_parser()
    args = parser.parse_args(["serve", "--model-path", "m.joblib"])
    assert args.command == "serve" and args.model_path == "m.joblib"


def test_build_config_defaults_to_path_loader():
    parser = build_parser()
    args = parser.parse_args(["serve", "--model-path", "m.joblib"])
    cfg = build_config_from_args(args)
    assert cfg.loader == "path" and cfg.model_type == "iforest"
    assert cfg.model_path == "m.joblib" and cfg.port == 8080


def test_build_config_registry():
    parser = build_parser()
    args = parser.parse_args(["serve", "--loader", "registry", "--model-type", "iforest",
                              "--tracking-uri", "sqlite:///x.db", "--port", "9000"])
    cfg = build_config_from_args(args)
    assert cfg.loader == "registry" and cfg.tracking_uri == "sqlite:///x.db" and cfg.port == 9000

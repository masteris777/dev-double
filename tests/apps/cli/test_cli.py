import pytest

from dev_double.apps.cli.main import main


def test_doctor_with_mock_engine(capsys):
    code = main(["doctor", "--engine", "mock"])
    out = capsys.readouterr().out
    assert code in (0, 2)  # 2 = finished with warnings; the mock engine is not a real model
    assert "Checking engine=mock model=mock-overlap" in out
    assert "latency:" in out


def test_doctor_reports_unreachable_server(capsys):
    code = main(["doctor", "--engine", "openai", "--base-url", "http://127.0.0.1:9/v1"])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_engine_choices_include_new_engines(capsys):
    with pytest.raises(SystemExit):
        main(["serve", "--engine", "fake"])
    assert "systemone" in capsys.readouterr().err


def test_serve_honor_request_model_flag_and_env(monkeypatch, capsys):
    seen: list = []
    monkeypatch.setattr("dev_double.apps.server.app.create_app", lambda settings: seen.append(settings))
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.delenv("DEV_DOUBLE_HONOR_REQUEST_MODEL", raising=False)

    main(["serve", "--engine", "mock"])
    main(["serve", "--engine", "mock", "--honor-request-model"])
    monkeypatch.setenv("DEV_DOUBLE_HONOR_REQUEST_MODEL", "true")
    main(["serve", "--engine", "mock"])
    assert [s.honor_request_model for s in seen] == [False, True, True]
    assert capsys.readouterr().out.count("honoring each request's `model`") == 2


def test_serve_vision_model_flag_env_and_startup_print(monkeypatch, capsys):
    seen: list = []
    monkeypatch.setattr("dev_double.apps.server.app.create_app", lambda settings: seen.append(settings))
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.delenv("DEV_DOUBLE_VISION_MODEL", raising=False)

    main(["serve", "--engine", "openai"])
    assert "vision model: none" in capsys.readouterr().out
    main(["serve", "--engine", "openai", "--vision-model", "qwen2.5vl:7b"])
    assert "vision model: qwen2.5vl:7b" in capsys.readouterr().out
    monkeypatch.setenv("DEV_DOUBLE_VISION_MODEL", "from-env")
    main(["serve", "--engine", "openai"])
    main(["serve", "--engine", "openai", "--vision-model", "flag-wins"])
    assert [s.vision_model for s in seen] == [None, "qwen2.5vl:7b", "from-env", "flag-wins"]

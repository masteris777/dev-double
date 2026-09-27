import pytest

from stunt_double.apps.cli.main import main


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

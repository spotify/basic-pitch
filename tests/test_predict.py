import sys

from basic_pitch import inference, predict


def run_cli(monkeypatch, tmp_path, predict_and_save):
    output_dir = tmp_path / "output"
    audio_path = tmp_path / "input.wav"
    monkeypatch.setattr(sys, "argv", ["basic-pitch", str(output_dir), str(audio_path)])
    monkeypatch.setattr(inference, "verify_output_dir", lambda path: None)
    monkeypatch.setattr(inference, "verify_input_path", lambda path: None)
    monkeypatch.setattr(inference, "predict_and_save", predict_and_save)
    monkeypatch.setattr(predict, "Model", lambda path: object())
    return predict.main()


def test_main_returns_zero_after_success(monkeypatch, tmp_path):
    assert run_cli(monkeypatch, tmp_path, lambda *args: None) == 0


def test_main_returns_nonzero_after_inference_failure(monkeypatch, tmp_path, capsys):
    def fail(*args):
        raise RuntimeError("inference failed")

    assert run_cli(monkeypatch, tmp_path, fail) == 1
    assert "inference failed" in capsys.readouterr().out

import catalist


def test_package_imports():
    assert callable(catalist.main)  # the `catalist` CLI entry point exists


def test_saving_one_answer_keeps_the_others(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(catalist, "ANSWERS", tmp_path / "answers.json")
    catalist.save_answers(lambda d: d.update(brief="b"))
    catalist.save_answers(lambda d: d["answers"].append({"question_id": "Q01"}))
    catalist.save_answers(lambda d: d["answers"].append({"question_id": "Q02"}))
    saved = json.loads((tmp_path / "answers.json").read_text())
    assert saved["brief"] == "b"
    assert [a["question_id"] for a in saved["answers"]] == ["Q01", "Q02"]

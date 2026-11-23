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


def test_failed_rerun_keeps_the_previous_answer():
    data = {"answers": [{"question_id": "Q05", "answer": "battery only"}]}
    catalist.put_answer(data, {"question_id": "Q05", "answer": "", "error": "429"})
    assert data["answers"][0]["answer"] == "battery only"
    catalist.put_answer(data, {"question_id": "Q05", "answer": "new"})
    assert data["answers"][0]["answer"] == "new"

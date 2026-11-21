import catalist


def test_package_imports():
    assert callable(catalist.main)  # the `catalist` CLI entry point exists

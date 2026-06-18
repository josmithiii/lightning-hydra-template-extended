from tests.helpers.package_available import _package_available


def test_package_available_true_for_installed() -> None:
    """A package that is certainly installed (pytest itself) reports available."""
    assert _package_available("pytest") is True
    assert _package_available("torch") is True


def test_package_available_false_for_missing() -> None:
    """A non-existent package reports unavailable rather than raising."""
    assert _package_available("definitely_not_a_real_package_xyz") is False


def test_package_available_handles_missing_parent() -> None:
    """A dotted name whose parent package is absent is handled gracefully."""
    assert _package_available("definitely_not_a_real_package_xyz.submodule") is False

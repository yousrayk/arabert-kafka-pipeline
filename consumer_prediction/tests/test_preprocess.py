import pytest

pytest.importorskip("arabert")

from app.preprocess import ArabicPreprocessor  # noqa: E402


@pytest.fixture(scope="module")
def preprocess():
    return ArabicPreprocessor()


def test_strips_tashkeel_and_tatweel(preprocess):
    assert preprocess("جَمِيـــل") == "جميل"


def test_replaces_urls_and_mentions_with_special_tokens(preprocess):
    out = preprocess("شوف http://example.com يا @ahmed")
    assert "http" not in out and "@ahmed" not in out
    assert "[رابط]" in out and "[مستخدم]" in out


def test_collapses_elongated_characters(preprocess):
    assert preprocess("رااااااائع") != "رااااااائع"


def test_output_is_stripped(preprocess):
    assert preprocess("  ممتاز  ") == "ممتاز"

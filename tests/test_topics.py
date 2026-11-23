import pytest

from catalist.topics import normalise_topic


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Remote Control Features", "remote_control_features"),
        ("  Casing Colour  ", "casing_colour"),
        (
            "Remote Control Features_\\text{_}LCD_screen",
            "remote_control_features_lcd_screen",
        ),
        ("topic_button_layout topic_button_layout", "button_layout"),
        ("topic_universal_remote", "universal_remote"),
        ("price_per_unit_per_month", "price_per_unit_per_month"),
    ],
)
def test_normalise_topic(raw, expected):
    assert normalise_topic(raw) == expected

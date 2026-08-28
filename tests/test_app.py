from streamlit.testing.v1 import AppTest


DISCLAIMER = (
    "Educational portfolio demo only. This model estimates severity conditional on a "
    "reported collision; it does not predict whether a collision will occur. Scores are "
    "uncalibrated and must not guide emergency response or other operational decisions."
)


def test_app_renders_disclaimer_and_readable_numeric_options():
    app = AppTest.from_file("app/streamlit_app.py").run(timeout=30)

    assert not list(app.exception)
    assert [warning.value for warning in app.warning] == [DISCLAIMER]

    options_by_label = {selectbox.label: selectbox.options for selectbox in app.selectbox}
    assert options_by_label["Urban or rural area"] == [
        "1 — Urban", "2 — Rural", "3 — Unallocated",
    ]
    assert options_by_label["Day of week"] == [
        "1 — Sunday", "2 — Monday", "3 — Tuesday", "4 — Wednesday",
        "5 — Thursday", "6 — Friday", "7 — Saturday",
    ]
    assert options_by_label["Primary road class"] == [
        "1 — Motorway", "2 — A(M)", "3 — A", "4 — B", "5 — C", "6 — Unclassified",
    ]
    assert options_by_label["Secondary road class"] == [
        "-1 — No second road", "1 — Motorway", "2 — A(M)", "3 — A",
        "4 — B", "5 — C", "6 — Unclassified",
    ]

from src.category_labels import CATEGORY_LABELS, format_category_option


def test_every_numeric_dropdown_code_has_a_human_readable_label():
    expected = {
        "Urban_or_Rural_Area": {1: "Urban", 2: "Rural", 3: "Unallocated"},
        "Day_of_Week": {
            1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
            5: "Thursday", 6: "Friday", 7: "Saturday",
        },
        "1st_Road_Class": {
            1: "Motorway", 2: "A(M)", 3: "A", 4: "B", 5: "C", 6: "Unclassified",
        },
        "2nd_Road_Class": {
            -1: "No second road", 1: "Motorway", 2: "A(M)", 3: "A",
            4: "B", 5: "C", 6: "Unclassified",
        },
    }
    assert {feature: CATEGORY_LABELS[feature] for feature in expected} == expected


def test_category_option_displays_code_and_meaning():
    assert format_category_option("Urban_or_Rural_Area", 2) == "2 — Rural"
    assert format_category_option("Day_of_Week", 6) == "6 — Friday"
    assert format_category_option("1st_Road_Class", 1) == "1 — Motorway"
    assert format_category_option("2nd_Road_Class", -1) == "-1 — No second road"


def test_category_option_falls_back_to_the_original_value():
    assert format_category_option("unmapped", 99) == "99"
    assert format_category_option("Road_Type", "Roundabout") == "Roundabout"

"""Converting a real resource-database CSV into Resource records.

Every row here is invented. The real export is not committed — see .gitignore —
so these tests build rows in the export's shape rather than reading it.

Two of these are regressions for bugs that reached live retrieval results, and
both came from the same place: keyword matching that was looser than it looked.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from scripts.import_resources import (
    ImportReport,
    convert_row,
    map_categories,
    map_status,
    parse_review_date,
)

from app.domain.resource import ServiceCategory, VerificationStatus

HEADERS = [
    "Resource Name",
    "Category",
    "Services Offered",
    "Address",
    "City",
    "ZIP Code",
    "Service Area",
    "Phone Number",
    "Website",
    "Eligibility Requirements",
    "Languages Spoken",
    "ADA Accessible",
    "Public Transit Access",
    "Payment Options / Cost",
    "Application Required",
    "Key Information Verified",
    "Verification Status",
    "Notes / Needs Confirmation",
    "Search Tags",
]


def row(**overrides: str) -> dict[str, str]:
    base = dict.fromkeys(HEADERS, "")
    base |= {
        "Resource Name": "Example Community Clinic",
        "Category": "Community Healthcare",
        "Services Offered": "Primary care, dental",
        "City": "Oakland",
        "Service Area": "Alameda County",
        "Phone Number": "510-555-0100",
        "Website": "example.org",
        "Verification Status": "Verified",
    }
    return base | overrides


# --- regressions ----------------------------------------------------------


def test_short_keywords_do_not_match_inside_longer_words() -> None:
    """'ssi' inside 'assistance' tagged 48 records as benefits navigation."""
    categories, _ = map_categories(
        row(Category="Alameda County – Food", **{"Search Tags": "food assistance; groceries"})
    )

    assert ServiceCategory.INSURANCE_NAVIGATION not in categories
    assert ServiceCategory.FOOD_ASSISTANCE in categories


def test_keywords_still_match_plural_forms() -> None:
    """Anchoring both ends then missed 'Medical Bills'."""
    categories, _ = map_categories(
        row(Category="Diabetes, Prescriptions & Medical Bills", **{"Search Tags": ""})
    )

    assert ServiceCategory.INSURANCE_NAVIGATION in categories


def test_a_workforce_program_is_not_caregiver_support() -> None:
    """'family stability' ranked a nail-care nonprofit first for adult day care."""
    categories, _ = map_categories(
        row(
            Category="Nonprofit / Workforce / Family Stability",
            **{"Search Tags": "workforce; job help; cosmetology training"},
        )
    )

    assert ServiceCategory.CAREGIVER_SUPPORT not in categories


def test_only_the_curated_fields_are_matched() -> None:
    """Prose mentioning Medi-Cal in passing must not retag the record."""
    categories, _ = map_categories(
        row(
            Category="Alameda County – Food",
            Services_Offered="",
            **{
                "Services Offered": "Groceries. Medi-Cal and insurance enrollment help nearby.",
                "Search Tags": "food; pantry",
            },
        )
    )

    assert categories == [ServiceCategory.FOOD_ASSISTANCE]


# --- category mapping -----------------------------------------------------


@pytest.mark.parametrize(
    ("category", "tags", "expected"),
    [
        ("Transportation / Veterans", "rides; paratransit", ServiceCategory.TRANSPORTATION),
        ("Alameda County – Food", "food; pantry", ServiceCategory.FOOD_ASSISTANCE),
        ("Housing, Rent & Homelessness", "eviction; shelter", ServiceCategory.HOUSING_SUPPORT),
        ("Mental Health", "counseling; crisis", ServiceCategory.MENTAL_HEALTH),
        ("Community Healthcare", "primary care; clinic", ServiceCategory.COMMUNITY_CLINIC),
        ("Transportation / ADA", "disability; accessible", ServiceCategory.ACCESSIBILITY_SUPPORT),
        ("Senior Services", "senior center", ServiceCategory.COMMUNITY_SERVICES),
    ],
)
def test_free_text_categories_map_to_the_fixed_set(
    category: str, tags: str, expected: ServiceCategory
) -> None:
    categories, _ = map_categories(row(Category=category, **{"Search Tags": tags}))
    assert expected in categories


def test_a_compound_category_maps_to_several() -> None:
    categories, _ = map_categories(
        row(Category="Transportation / Seniors / Disability", **{"Search Tags": "rides; ADA"})
    )

    assert ServiceCategory.TRANSPORTATION in categories
    assert ServiceCategory.ACCESSIBILITY_SUPPORT in categories


def test_an_unmappable_row_falls_back_and_is_reported() -> None:
    categories, used_fallback = map_categories(
        row(Category="Recognized Healthy Nail Salon", **{"Search Tags": "nail salon"})
    )

    assert used_fallback is True
    assert categories == [ServiceCategory.COMMUNITY_SERVICES]


# --- verification status is copied, never upgraded ------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Verified", VerificationStatus.VERIFIED),
        ("Partially Verified", VerificationStatus.PARTIALLY_VERIFIED),
        ("Needs Verification", VerificationStatus.UNVERIFIED),
        ("  needs   verification ", VerificationStatus.UNVERIFIED),
    ],
)
def test_status_is_mapped_faithfully(raw: str, expected: VerificationStatus) -> None:
    assert map_status(raw) == expected


def test_an_unrecognised_status_is_treated_as_unverified() -> None:
    """Failing closed is the only safe default for a verification field."""
    assert map_status("something new") == VerificationStatus.UNVERIFIED
    assert map_status("") == VerificationStatus.UNVERIFIED


# --- review dates ---------------------------------------------------------


def test_a_review_date_is_parsed_out_of_the_note() -> None:
    assert parse_review_date("Services reviewed 2026-10-05; details flagged") == date(2026, 10, 5)


def test_a_note_without_a_date_yields_none() -> None:
    """It is not back-filled with today. We do not know when it was checked."""
    assert parse_review_date("Not independently verified; imported from supplied list") is None
    assert parse_review_date("") is None


def test_a_future_review_date_is_rejected() -> None:
    future = (date.today() + timedelta(days=30)).isoformat()
    assert parse_review_date(f"reviewed {future}") is None


# --- whole-row conversion -------------------------------------------------


def test_a_full_row_converts() -> None:
    report = ImportReport()
    resource = convert_row(
        row(
            **{
                "Resource Name": "Eastside Health Center",
                "Category": "Community Healthcare",
                "Search Tags": "primary care; clinic",
                "ZIP Code": "94601",
                "Eligibility Requirements": "Uninsured welcome",
                "Languages Spoken": "English; Spanish",
                "ADA Accessible": "Yes",
                "Payment Options / Cost": "Sliding scale",
                "Key Information Verified": "Reviewed 2026-10-05",
                "Notes / Needs Confirmation": "Hours need confirming",
            }
        ),
        1,
        report,
    )

    assert resource is not None
    assert resource.organization_name == "Eastside Health Center"
    assert ServiceCategory.COMMUNITY_CLINIC in resource.service_categories
    assert resource.service_area.postal_codes == ["94601"]
    assert resource.languages == ["English", "Spanish"]
    assert resource.last_verified == date(2026, 10, 5)
    assert resource.confirmation_notes == "Hours need confirming"
    assert resource.source == "cents-resource-database"


def test_caveats_are_preserved_not_dropped() -> None:
    """The caveat is the most important field on a partially verified record."""
    report = ImportReport()
    caveat = "Do not apply Medi-Cal or accessibility information to every location"
    resource = convert_row(
        row(**{"Verification Status": "Partially Verified", "Notes / Needs Confirmation": caveat}),
        1,
        report,
    )

    assert resource is not None
    assert resource.confirmation_notes == caveat
    assert resource.verification_status is VerificationStatus.PARTIALLY_VERIFIED


def test_placeholder_text_is_not_treated_as_content() -> None:
    report = ImportReport()
    resource = convert_row(
        row(**{"Eligibility Requirements": "Needs confirmation", "Languages Spoken": "Unknown"}),
        1,
        report,
    )

    assert resource is not None
    assert resource.eligibility is None
    assert resource.languages == []


def test_a_row_without_a_name_is_skipped_with_a_reason() -> None:
    report = ImportReport()

    assert convert_row(row(**{"Resource Name": ""}), 7, report) is None
    assert report.skipped == [("row 7", "no resource name")]


def test_service_area_scope_is_inferred() -> None:
    report = ImportReport()
    statewide = convert_row(
        row(**{"Service Area": "Statewide California", "City": "California"}), 1, report
    )
    county = convert_row(row(**{"Service Area": "Alameda County", "City": "Oakland"}), 2, report)
    national = convert_row(
        row(**{"Service Area": "National hotline", "City": "Bay Area"}), 3, report
    )

    assert statewide is not None and statewide.service_area.scope.value == "statewide"
    assert county is not None and county.service_area.counties == ["Alameda County"]
    assert national is not None and national.service_area.scope.value == "national"


def test_resource_ids_are_stable_and_namespaced() -> None:
    report = ImportReport()
    resource = convert_row(row(**{"Resource Name": "Westside Medical Care"}), 12, report)

    assert resource is not None
    assert resource.resource_id == "cents-0012-westside-medical-care"

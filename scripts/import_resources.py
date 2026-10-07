#!/usr/bin/env python
"""Convert a CENTS resource-database CSV into the Navigator's resource JSON.

    python scripts/import_resources.py "path/to/CENTS Database.csv" \\
        --out app/data/private/cents_resources.json

The export's own columns are taken at face value and nothing is inferred that
is not in the row. In particular:

* **Verification status is copied, never upgraded.** A row that says
  "Needs Verification" stays unverified, and the retriever withholds it. The
  whole point of that column is that someone has not checked the record yet.
* **"Notes / Needs Confirmation" is preserved** as ``confirmation_notes`` and
  travels into the prompt, so an unconfirmed detail reaches the member as a
  caveat instead of being presented as fact.
* **A missing review date stays missing.** It is not back-filled with today.

Free-text categories are mapped onto the Navigator's fixed category set with
keyword rules over Category, Search Tags, and Services Offered. The mapping is
reported, including every row that needed the fallback, so its quality is
visible rather than assumed.

Writes JSON only. It never uploads anything and never contacts a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Allow running directly from a checkout where the package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.resource import (
    Resource,
    ServiceArea,
    ServiceCategory,
    ServiceScope,
    VerificationStatus,
)

# Verification wording in the export -> the status the model understands.
_STATUS_MAP = {
    "verified": VerificationStatus.VERIFIED,
    "partially verified": VerificationStatus.PARTIALLY_VERIFIED,
    "needs verification": VerificationStatus.NEEDS_VERIFICATION,
    "unverified": VerificationStatus.NEEDS_VERIFICATION,
    "needs review": VerificationStatus.NEEDS_REVIEW,
}

# Ordered keyword rules. A row can match several categories, which is correct:
# "Transportation / Seniors / Disability" genuinely is more than one thing.
_CATEGORY_RULES: list[tuple[ServiceCategory, tuple[str, ...]]] = [
    (
        ServiceCategory.TRANSPORTATION,
        ("transport", "ride", "paratransit", "shuttle", "fare", "driver", "transit", "taxi"),
    ),
    (
        ServiceCategory.FOOD_ASSISTANCE,
        ("food", "nutrition", "meal", "pantry", "grocer", "calfresh", "snap", "wic", "hunger"),
    ),
    (
        ServiceCategory.HOUSING_SUPPORT,
        ("housing", "rent", "homeless", "shelter", "eviction", "landlord", "tenant"),
    ),
    (
        ServiceCategory.CAREGIVER_SUPPORT,
        (
            # "family stability" was here and tagged a workforce-training
            # nonprofit as caregiver support, which then ranked first for
            # "adult day care". Caregiving language only.
            "caregiver",
            "caregiving",
            "respite",
            "companion",
            "in-home services",
            "adult day",
            "adult care",
            "day program",
            "protective services",
        ),
    ),
    (
        ServiceCategory.MENTAL_HEALTH,
        (
            "mental health",
            "behavioral",
            "counseling",
            "crisis",
            "suicide",
            "abuse support",
            "abuse",
            "sexual assault",
            "domestic violence",
            "substance",
            "recovery",
            "wellness",
            "acupuncture",
        ),
    ),
    (
        ServiceCategory.INSURANCE_NAVIGATION,
        (
            # Deliberately specific. "medi-cal" and "benefits" on their own appear
            # in most listings as a payment note, and tagged every food bank and
            # paratransit service as benefits navigation.
            "benefits navigation",
            "benefits advocacy",
            "benefits enrollment",
            "social security",
            "insurance",
            "medical bill",
            "medical debt",
            "medi-cal family",
            "child support",
            "lifeline",
            "health plan",
            "prescription assistance",
            "financial assistance",
            "ssi",
            "ssdi",
            "charity care",
            "copay",
            "enrollment assistance",
        ),
    ),
    (
        ServiceCategory.LANGUAGE_ASSISTANCE,
        ("interpret", "translat", "language access", "asl", "sign language"),
    ),
    (
        ServiceCategory.ACCESSIBILITY_SUPPORT,
        ("ada", "disabilit", "disabled", "accessible", "mobility", "wheelchair", "fall prevention"),
    ),
    (
        ServiceCategory.COMMUNITY_CLINIC,
        (
            "healthcare",
            "health center",
            "clinic",
            "primary care",
            "podiatry",
            "foot care",
            "dental",
            "medical",
            "nursing",
            "palliative",
            "pace",
            "doctor",
            "street medicine",
            "diabetes",
            "physician",
        ),
    ),
    (
        ServiceCategory.APPOINTMENT_SUPPORT,
        ("appointment", "navigation", "care coordination", "scheduling"),
    ),
    (
        ServiceCategory.COMMUNITY_SERVICES,
        (
            # "community" and "nonprofit" alone matched two thirds of the file.
            "wraparound",
            "clothing",
            "diaper",
            "gift card",
            "school supplies",
            "internet",
            "wi-fi",
            "digital access",
            "workforce",
            "senior center",
            "senior services",
            "hotline",
            "utility",
            "utilities",
            "legal aid",
            "community services",
            "community assistance",
        ),
    ),
]

# Keywords are anchored at their start only. Plain substring matching tagged 48
# records as benefits navigation because "ssi" sits inside "assistance" — the same
# trap waits in "ada"/"Canada", "pace"/"space", "ride"/"bridge". A trailing anchor
# would then miss plurals: "Diabetes, Prescriptions & Medical Bills" has to match
# "medical bill". A leading anchor alone fixes both.
_COMPILED_RULES: list[tuple[ServiceCategory, re.Pattern[str]]] = [
    (
        category,
        re.compile(
            "|".join(r"\b" + re.escape(keyword).replace(r"\ ", r"\s+") for keyword in keywords)
        ),
    )
    for category, keywords in _CATEGORY_RULES
]

# When nothing matches, the row still needs one category to be a valid Resource.
_FALLBACK_CATEGORY = ServiceCategory.COMMUNITY_SERVICES

# Placeholder text the export uses where a field is simply unknown. Carrying
# these through as if they were content would make answers worse.
_UNKNOWN_MARKERS = (
    "needs confirmation",
    "not independently verified",
    "unknown",
    "n/a",
    "none listed",
    "tbd",
)

_COUNTY_PATTERN = re.compile(r"([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*) County")
_DATE_PATTERN = re.compile(r"(20\d\d)-(\d\d)-(\d\d)")
_STATEWIDE_HINTS = ("california", "statewide", "state of california")
_NATIONAL_HINTS = ("national", "nationwide", "united states", "all states")


@dataclass
class ImportReport:
    """What the conversion did, and what it could not do."""

    total_rows: int = 0
    converted: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    status_counts: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    fallback_rows: list[str] = field(default_factory=list)
    undated: int = 0


def _raw(value: str | None) -> str | None:
    """Trim only. Keeps the source's wording, including "needs confirmation".

    For a descriptive field that is honest information, not noise: "ADA access:
    needs confirmation by location" tells a member to ask, which is strictly
    better than omitting the field and leaving them to assume.
    """
    if value is None:
        return None
    text = " ".join(value.split())
    return text or None


def _clean(value: str | None) -> str | None:
    """Trim, and drop the export's placeholders for 'we do not know'.

    Used for list fields and for the description, where a placeholder would
    become a bogus list item or pad the embedding text with nothing.
    """
    if value is None:
        return None
    text = " ".join(value.split())
    if not text:
        return None
    if any(marker in text.lower() for marker in _UNKNOWN_MARKERS) and len(text) < 40:
        return None
    return text


def _split_list(value: str | None) -> list[str]:
    if not (text := _clean(value)):
        return []
    parts = re.split(r"[;,/]|\band\b", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]


def map_categories(row: dict[str, str]) -> tuple[list[ServiceCategory], bool]:
    """Map the row's free text onto the fixed category set.

    Returns the categories and whether the fallback had to be used.
    """
    # Only the curated fields are matched. Services Offered is prose that
    # mentions payment options, referral partners, and populations in passing;
    # matching it tagged a food bank as benefits navigation. Category and
    # Search Tags are what the author wrote to describe and find the record.
    haystack = " ".join((row.get("Category", ""), row.get("Search Tags", ""))).lower()

    matched = [category for category, pattern in _COMPILED_RULES if pattern.search(haystack)]
    if matched:
        return matched, False
    return [_FALLBACK_CATEGORY], True


def map_status(value: str) -> VerificationStatus:
    """Copy the export's judgement. Never upgrade it."""
    return _STATUS_MAP.get(" ".join(value.split()).lower(), VerificationStatus.UNVERIFIED)


def parse_review_date(value: str) -> date | None:
    """Pull a review date out of the free-text verification note, if present."""
    match = _DATE_PATTERN.search(value or "")
    if not match:
        return None
    try:
        parsed = date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return None
    return None if parsed > date.today() else parsed


def build_service_area(row: dict[str, str]) -> ServiceArea:
    city = _clean(row.get("City"))
    area_text = row.get("Service Area", "") or ""
    lowered = area_text.lower()

    scope = ServiceScope.LOCAL
    if any(hint in lowered for hint in _NATIONAL_HINTS):
        scope = ServiceScope.NATIONAL
    elif any(hint in lowered for hint in _STATEWIDE_HINTS):
        scope = ServiceScope.STATEWIDE
    elif "county" in lowered:
        scope = ServiceScope.COUNTY

    counties = _COUNTY_PATTERN.findall(area_text)
    postal = re.findall(r"\b9\d{4}\b", row.get("ZIP Code", "") or "")

    # "Bay Area" and "California" are areas, not cities.
    cities = [] if not city or city.lower() in {"bay area", "california"} else [city]
    if not cities and scope is ServiceScope.LOCAL and counties:
        scope = ServiceScope.COUNTY

    return ServiceArea(
        scope=scope,
        cities=cities,
        counties=[f"{c} County" for c in counties],
        state="CA",
        postal_codes=postal,
    )


def build_description(row: dict[str, str]) -> str:
    """Compose a description long enough to be useful and to embed well."""
    parts = [p for p in (_clean(row.get("Services Offered")), _clean(row.get("Category"))) if p]
    text = ". ".join(parts)
    if len(text) < 10:
        text = f"{row.get('Resource Name', 'Organization').strip()}: community resource listing."
    return text[:2000]


def slugify(name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "resource"
    return f"cents-{index:04d}-{slug}"


def convert_row(row: dict[str, str], index: int, report: ImportReport) -> Resource | None:
    name = _clean(row.get("Resource Name"))
    if not name:
        report.skipped.append((f"row {index}", "no resource name"))
        return None

    categories, used_fallback = map_categories(row)
    if used_fallback:
        report.fallback_rows.append(name)

    # ADA and transit are kept as their own fields rather than flattened into a
    # generic list, so "needs confirmation by location" stays attached to the
    # thing it qualifies.
    ada = _raw(row.get("ADA Accessible"))
    transit = _raw(row.get("Public Transit Access"))
    application = _raw(row.get("Application Required"))

    review_date = parse_review_date(row.get("Key Information Verified", ""))
    if review_date is None:
        report.undated += 1

    try:
        resource = Resource.model_validate(
            {
                "resource_id": slugify(name, index),
                "organization_name": name[:200],
                "description": build_description(row),
                "service_categories": [c.value for c in categories],
                "services": _split_list(row.get("Services Offered"))[:12],
                "service_area": build_service_area(row).model_dump(),
                "address": _clean(row.get("Address")),
                "eligibility": (_raw(row.get("Eligibility Requirements")) or "")[:1000] or None,
                "languages": _split_list(row.get("Languages Spoken"))[:15],
                "accessibility": [],
                "ada_access": (ada or "")[:500] or None,
                "transit_access": (transit or "")[:500] or None,
                "application_required": (application or "")[:500] or None,
                "source_category": (_clean(row.get("Category")) or "")[:300] or None,
                "search_tags": _split_list(row.get("Search Tags"))[:20],
                "service_area_note": (_clean(row.get("Service Area")) or "")[:500] or None,
                "verification_note": (_clean(row.get("Key Information Verified")) or "")[:1000]
                or None,
                "cost": (_raw(row.get("Payment Options / Cost")) or "")[:200] or None,
                "phone": (_raw(row.get("Phone Number")) or "")[:40] or None,
                "website": (_raw(row.get("Website")) or "")[:500] or None,
                "source": "cents-resource-database",
                "last_verified": review_date,
                "verification_status": map_status(row.get("Verification Status", "")).value,
                "confirmation_notes": (_clean(row.get("Notes / Needs Confirmation")) or "")[:2000]
                or None,
            }
        )
    except Exception as exc:
        report.skipped.append((name, f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"))
        return None

    report.status_counts[resource.verification_status.value] += 1
    for category in resource.service_categories:
        report.category_counts[category.value] += 1
    return resource


def find_header(rows: list[list[str]]) -> int:
    """Locate the header row. Exports often start with blank or title rows."""
    for index, row in enumerate(rows[:10]):
        if any("resource name" == cell.strip().lower() for cell in row):
            return index
    raise SystemExit("Could not find a header row containing 'Resource Name'.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="The CENTS resource database CSV export.")
    parser.add_argument(
        "--out",
        default="app/data/private/cents_resources.json",
        help="Where to write the resource JSON (gitignored by default).",
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Write only fully verified rows. Off by default: every row is kept and "
        "its verification status travels with it.",
    )
    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    header_index = find_header(rows)
    header = rows[header_index]
    records = [
        dict(zip(header, row, strict=False))
        for row in rows[header_index + 1 :]
        if any(cell.strip() for cell in row)
    ]

    report = ImportReport(total_rows=len(records))
    resources = [
        resource
        for index, row in enumerate(records, start=1)
        if (resource := convert_row(row, index, report)) is not None
    ]
    report.converted = len(resources)

    if args.verified_only:
        kept = [r for r in resources if r.verification_status is VerificationStatus.VERIFIED]
        excluded = len(resources) - len(kept)
        resources = kept
    else:
        excluded = 0

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "_meta": {
                    "source": "CENTS resource database CSV export",
                    "imported_by": "scripts/import_resources.py",
                    "imported_on": date.today().isoformat(),
                    "contains_patient_data": False,
                    "notice": (
                        "REAL ORGANIZATION DATA. Verification status is copied from the "
                        "export and never upgraded. Every status is searchable; the "
                        "status and any unresolved-field notes travel with each record "
                        "so uncertainty is communicated rather than hidden."
                    ),
                    "record_count": len(resources),
                },
                "resources": [r.model_dump(mode="json") for r in resources],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"  rows read            : {report.total_rows}")
    print(f"  converted            : {report.converted}")
    print(f"  excluded by --verified-only: {excluded}")
    print(f"  written              : {len(resources)} -> {destination}")
    print(f"  no review date       : {report.undated}")
    print("\n  verification status of converted rows:")
    for status, count in report.status_counts.most_common():
        print(f"    {count:>4}  {status}")
    print("\n  categories assigned (a row may have several):")
    for category, count in report.category_counts.most_common():
        print(f"    {count:>4}  {category}")
    if report.fallback_rows:
        print(f"\n  needed the '{_FALLBACK_CATEGORY.value}' fallback: {len(report.fallback_rows)}")
        for name in report.fallback_rows[:10]:
            print(f"    - {name[:70]}")
    if report.skipped:
        print(f"\n  skipped {len(report.skipped)}:")
        for name, reason in report.skipped[:10]:
            print(f"    - {name[:50]}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

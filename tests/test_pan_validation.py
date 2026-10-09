"""
Unit tests for PAN Card ID extraction, OCR auto-repair, field detection,
and document validation.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.document_types import DOCUMENT_CONFIG, detect_document_type
from src.validator import (
    _find_id_number,
    _detect_required_fields,
    _clean_pan_candidate,
    validate_document,
    print_report,
)


def words(*parts):
    return [{"text": p, "confidence": 80, "bbox": None} for p in parts]


def assert_pan_id(tokens, expected):
    result = _find_id_number(words(*tokens), 10, doc_type="pan")
    assert result["found"], f"PAN ID not found in {tokens}"
    assert result["best"]["digits"] == expected, (
        f"Expected {expected}, got {result['best']['digits']} in {tokens}"
    )


def test_pan_id_extraction():
    # 1. Standard PAN
    assert_pan_id(["PAN:", "ABCDE1234F"], "ABCDE1234F")
    assert_pan_id(["ABCDE1234F"], "ABCDE1234F")

    # 2. PAN with spaces or punctuation
    assert_pan_id(["ABCDE", "1234", "F"], "ABCDE1234F")
    assert_pan_id(["ABCDE-1234-F"], "ABCDE1234F")
    assert_pan_id(["ABCDE", "1234F"], "ABCDE1234F")

    # 3. Positional OCR error correction
    # '0' in letter position -> 'O'
    assert _clean_pan_candidate("ABCD01234F") == "ABCDO1234F"
    assert_pan_id(["ABCD01234F"], "ABCDO1234F")

    # 'O' in digit position -> '0'
    assert _clean_pan_candidate("ABCDE12O4F") == "ABCDE1204F"
    assert_pan_id(["ABCDE12O4F"], "ABCDE1204F")

    # 'I' in digit position -> '1'
    assert _clean_pan_candidate("ABCDEI234F") == "ABCDE1234F"
    assert_pan_id(["ABCDEI234F"], "ABCDE1234F")

    # '1' in 10th letter position -> 'I'
    assert _clean_pan_candidate("ABCDE12341") == "ABCDE1234I"
    assert_pan_id(["ABCDE12341"], "ABCDE1234I")

    print("[PASS] PAN ID extraction & OCR auto-repair tests passed")


def test_pan_document_type_detection():
    # Detect via header keywords
    assert detect_document_type(words("INCOME", "TAX", "DEPARTMENT")) == "pan"
    assert detect_document_type(words("PERMANENT", "ACCOUNT", "NUMBER")) == "pan"
    assert detect_document_type(words("GOVT", "OF", "INDIA", "PAN", "CARD")) == "pan"
    assert detect_document_type(words("NSDL", "PAN", "CARD")) == "pan"

    # Detect via standalone PAN format
    assert detect_document_type(words("ABCDE1234F")) == "pan"

    print("[PASS] PAN document type detection tests passed")


def test_pan_field_extraction():
    pan_fields = DOCUMENT_CONFIG["pan"]["required_fields"]

    # Standard labels
    std_words = words(
        "INCOME", "TAX", "DEPARTMENT",
        "Name:", "Jane", "Doe",
        "Father's", "Name:", "Robert", "Doe",
        "DOB:", "05/05/1985",
        "PAN:", "ABCDE1234F"
    )
    detected_std = _detect_required_fields(std_words, pan_fields, doc_type="pan")
    assert detected_std["all_present"], f"Standard fields failed: {detected_std}"

    # OCR distorted labels & standalone dates without "DOB" label
    noisy_words = words(
        "GOVT", "OF", "INDIA", "INCOMETAX",
        "Nwre", "RAMESH", "KUMAR",
        "Fatner", "SURESH", "KUMAR",
        "15/08/1990",
        "ABCDE1234F"
    )
    detected_noisy = _detect_required_fields(noisy_words, pan_fields, doc_type="pan")
    assert detected_noisy["all_present"], f"Noisy fields failed: {detected_noisy}"

    # Name pairs without explicit label + standalone date
    bilingual_words = words(
        "INCOME", "TAX",
        "JANE", "SMITH",
        "ROBERT", "SMITH",
        "22-11-1988",
        "ABCDE1234F"
    )
    detected_bilingual = _detect_required_fields(bilingual_words, pan_fields, doc_type="pan")
    assert detected_bilingual["all_present"], f"Bilingual fields failed: {detected_bilingual}"

    print("[PASS] PAN required fields extraction tests passed")


def test_pan_end_to_end_validation():
    sample_ocr = {
        "words": words(
            "INCOME", "TAX", "DEPARTMENT",
            "Name:", "Jane", "Smith",
            "Father's", "Name:", "Robert", "Smith",
            "DOB:", "05/05/1985",
            "PAN:", "ABCDE1234F"
        ),
        "word_count": 12,
        "mean_confidence": 92.0,
    }

    report = validate_document(
        ocr_result=sample_ocr,
        quality_result={"overall_quality": "GOOD"},
        qr_result={"count": 0, "qrcodes": []},
    )

    assert report["document_type"] == "PAN Card", report
    assert report["validation_result"] == "PASS", report
    assert report["checks"]["detected_id"] == "ABCDE1234F", report

    # Test report formatting without fraud model (fallback check)
    formatted = print_report(report)
    assert "FINAL DECISION      : VERIFIED" in formatted, formatted
    assert "Detected Type       : PAN Card" in formatted

    # Test report formatting with low fraud risk
    formatted_low = print_report(report, fraud_result={"predicted_risk": "LOW"})
    assert "FINAL DECISION      : VERIFIED" in formatted_low
    assert "Fraud Risk          : LOW" in formatted_low

    print("[PASS] PAN end-to-end document validation & reporting passed")


def main():
    print("=" * 60)
    print("RUNNING PAN CARD VALIDATION TEST SUITE")
    print("=" * 60)
    test_pan_id_extraction()
    test_pan_document_type_detection()
    test_pan_field_extraction()
    test_pan_end_to_end_validation()
    print("=" * 60)
    print("ALL PAN TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()

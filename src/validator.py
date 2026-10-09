import re

from src.document_types import DOCUMENT_CONFIG, detect_document_type


FIELD_SYNONYMS = {
    "dob": [
        "dob",
        "date of birth",
        "birth",
    ],

    "name": [
        "name",
        "holder",
    ],

    "gender": [
        "gender",
        "male",
        "female",
    ],

    "father's name": [
        "father",
        "father name",
        "father's name",
        "son",
        "daughter",
        "wife",
    ],

    "validity": [
        "validity",
        "valid",
        "valid till",
        "valid upto",
        "valid up to",
    ],
}


def _clean(text):
    return re.sub(r"\s+", " ", str(text)).strip()


# Indian DL number: 2-letter state + RTO + licence serial.
# Official cards look like HP-32 20140003137 or UP14 20240042764.
# Tesseract often reads the hyphen as a period, or splits tokens.
_DL_ID_RE = re.compile(
    r"\b([A-Z]{2})[\s.\-_/]*([0-9]{1,4})[\s.\-_/]*([0-9]{6,13})\b",
    re.IGNORECASE,
)

_PAN_CANDIDATE_RE = re.compile(
    r"(?:PAN\s*[:.\-_]*)?([A-Za-z0-9]{5}[\s.\-_/]*[A-Za-z0-9]{4}[\s.\-_/]*[A-Za-z0-9])(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_PAN_CHAR_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
_PAN_CHAR_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"}

_DL_STOPWORDS = {
    "union", "india", "indian", "of", "driving", "licence", "license",
    "date", "issue", "validity", "valid", "till", "upto", "blood",
    "group", "father", "name", "dob", "gender", "male", "female",
    "holder", "the", "and", "card",
    "income", "tax", "department", "govt", "government", "permanent",
    "account", "number", "pan", "nsdl", "utiitsl", "pancard", "signature",
    "photo", "sub", "form",
}


def _clean_pan_candidate(raw: str):
    """Normalize and validate a 10-char PAN token, repairing common OCR confusions."""
    token = re.sub(r"[^A-Za-z0-9]", "", str(raw)).upper()
    if len(token) != 10:
        return None

    chars = list(token)
    # Positions 0..4 must be letters (A-Z)
    for idx in range(5):
        if chars[idx].isdigit():
            chars[idx] = _PAN_CHAR_TO_LETTER.get(chars[idx], chars[idx])
        if not chars[idx].isalpha():
            return None

    # Positions 5..8 must be digits (0-9)
    for idx in range(5, 9):
        if chars[idx].isalpha():
            chars[idx] = _PAN_CHAR_TO_DIGIT.get(chars[idx], chars[idx])
        if not chars[idx].isdigit():
            return None

    # Position 9 must be a letter (A-Z)
    if chars[9].isdigit():
        chars[9] = _PAN_CHAR_TO_LETTER.get(chars[9], chars[9])
    if not chars[9].isalpha():
        return None

    return "".join(chars)


def _extract_pan_id(text: str):
    """Extract and validate 10-character PAN number from text."""
    for match in _PAN_CANDIDATE_RE.finditer(text):
        cleaned = _clean_pan_candidate(match.group(1))
        if cleaned:
            return {
                "found": True,
                "count": 1,
                "best": {
                    "digits": cleaned,
                    "source_word": _clean(match.group(0)),
                    "bbox": None,
                    "confidence": None,
                },
            }
    return None


def _normalize_dl_text(text):
    """Join OCR punctuation so HP.32 / HP-32 / HP 32 can all match."""
    text = _clean(text)
    text = text.replace(",", " ")
    return text


def _normalize_field_text(text):
    """Map common Tesseract misreads of document field labels."""
    text = _clean(text).lower()
    replacements = (
        ("valisty", "validity"),
        ("validty", "validity"),
        ("vality", "validity"),
    )
    for src, dst in replacements:
        text = text.replace(src, dst)

    text = re.sub(r"\b(nwre|nave|nane|namc|nme|naam)\b", "name", text)
    text = re.sub(r"\b(famer|facher|feiner|fatner|fathar|fater|fathr|pita)\b", "father", text)
    text = re.sub(r"\b(ben|birh|bith|berth|girth|janm|janam)\b", "birth", text)
    return text


def _extract_dl_id(text):
    match = _DL_ID_RE.search(_normalize_dl_text(text))
    if not match:
        return None

    value = (
        match.group(1) + match.group(2) + match.group(3)
    ).upper()
    return {
        "found": True,
        "count": 1,
        "best": {
            "digits": value,
            "source_word": _clean(match.group(0)),
            "bbox": None,
            "confidence": None,
        },
    }


def _sliding_window_text(words, start, width):
    parts = []
    for index in range(start, min(start + width, len(words))):
        parts.append(str(words[index].get("text", "")))
    return _clean(" ".join(parts))


def _looks_like_name_pair(words):
    letters = []
    for word in words:
        cleaned = re.sub(r"[^A-Za-z]", "", str(word.get("text", "")))
        if len(cleaned) >= 3 and cleaned.lower() not in _DL_STOPWORDS:
            letters.append(cleaned)
        else:
            letters.append("")

    for i in range(len(letters) - 1):
        if letters[i] and letters[i + 1]:
            return True
    return False


def _count_name_pairs(words):
    letters = []
    for word in words:
        cleaned = re.sub(r"[^A-Za-z]", "", str(word.get("text", "")))
        if len(cleaned) >= 3 and cleaned.lower() not in _DL_STOPWORDS:
            letters.append(cleaned)
        else:
            letters.append("")

    pairs = 0
    i = 0
    while i < len(letters) - 1:
        if letters[i] and letters[i + 1]:
            pairs += 1
            i += 2
        else:
            i += 1
    return pairs


def _find_id_number(words, expected_len, doc_type=""):
    """
    Find an identification number.

    Aadhaar:
        9999 8888 7777

    PAN:
        ABCDE1234F

    Driving Licence:
        HP-32 20140003137
        UP14 20240042764
    """

    text = _clean(
        " ".join(
            str(w.get("text", ""))
            for w in words
        )
    )

    # --------------------------------------------------
    # AADHAAR
    # --------------------------------------------------
    if doc_type == "aadhaar":

        match = re.search(
            r"\b\d{4}\s+\d{4}\s+\d{4}\b",
            text
        )

        if not match:
            match = re.search(
                r"\b\d{12}\b",
                text
            )

        if match:
            value = re.sub(
                r"\D",
                "",
                match.group(0)
            )

            return {
                "found": len(value) == 12,
                "count": 1,
                "best": {
                    "digits": value,
                    "source_word": match.group(0),
                    "bbox": None,
                    "confidence": None,
                },
            }

    # --------------------------------------------------
    # PAN
    # --------------------------------------------------
    if doc_type == "pan":

        found = _extract_pan_id(text)
        if found:
            return found

        # OCR often splits tokens across words (e.g. ABCDE and 1234F)
        for i in range(len(words)):
            for width in (2, 3, 4, 5):
                window = _sliding_window_text(words, i, width)
                found = _extract_pan_id(window)
                if found:
                    return found

    # --------------------------------------------------
    # DRIVING LICENCE
    # --------------------------------------------------
    if doc_type == "license":

        found = _extract_dl_id(text)
        if found:
            return found

        # OCR often splits HP / . / 32 / 20140003137 across tokens.
        for i in range(len(words)):
            for width in (2, 3, 4, 5, 6):
                window = _sliding_window_text(words, i, width)
                found = _extract_dl_id(window)
                if found:
                    return found

        # Older / synthetic samples used a plain 15-16 digit block.
        match = re.search(r"\b\d{15,16}\b", text)
        if match:
            value = match.group(0)
            return {
                "found": True,
                "count": 1,
                "best": {
                    "digits": value,
                    "source_word": value,
                    "bbox": None,
                    "confidence": None,
                },
            }

    # --------------------------------------------------
    # GENERIC EXACT DIGIT MATCH
    # --------------------------------------------------

    for word in words:

        raw = _clean(
            word.get("text", "")
        )

        digits = re.sub(
            r"\D",
            "",
            raw
        )

        if expected_len and len(digits) == expected_len:

            return {
                "found": True,
                "count": 1,
                "best": {
                    "digits": digits,
                    "source_word": raw,
                    "bbox":
                        word.get("bbox"),
                    "confidence":
                        word.get("confidence"),
                },
            }

    return {
        "found": False,
        "count": 0,
        "best": None,
    }


def _detect_required_fields(words, required_fields, doc_type=""):

    raw_text = _clean(
        " ".join(
            str(w.get("text", ""))
            for w in words
        )
    )
    text = raw_text.lower()
    field_text = _normalize_field_text(raw_text)

    found = {}

    for field in required_fields:

        options = FIELD_SYNONYMS.get(
            field,
            [field]
        )

        found[field] = any(
            option in text or option in field_text
            for option in options
        )

    # --------------------------------------------------
    # Driving Licence special handling
    #
    # Labels are tiny on real cards, so OCR often returns
    # "Nwre" instead of "Name" and "Date of Ben" instead
    # of "Date of Birth". Dates and name tokens are more
    # stable signals than the exact printed labels.
    # --------------------------------------------------

    if doc_type == "license":

        dates = re.findall(
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b",
            field_text
        )

        if len(dates) >= 2:
            found["validity"] = True

        if (
            len(dates) >= 3
            or "date of birth" in field_text
            or "dob" in field_text
            or "birth" in field_text
        ):
            found["dob"] = True

        if (
            re.search(r"\bname\b", field_text)
            or _looks_like_name_pair(words)
        ):
            found["name"] = True

    # --------------------------------------------------
    # PAN special handling
    #
    # Real PAN cards often have date formatted as DD/MM/YYYY
    # without explicit "DOB" prefix, and names may be bilingual
    # or have subtle OCR distortions in "Father's Name".
    # --------------------------------------------------

    if doc_type == "pan":

        dates = re.findall(
            r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b",
            field_text
        )

        if (
            dates
            or "date of birth" in field_text
            or "dob" in field_text
            or "birth" in field_text
        ):
            found["dob"] = True

        if (
            re.search(r"\bname\b", field_text)
            or _looks_like_name_pair(words)
        ):
            found["name"] = True

        if (
            found.get("father's name")
            or re.search(r"\b(father|paternal|pita)\b", field_text)
            or _count_name_pairs(words) >= 2
        ):
            found["father's name"] = True

    return {
        "fields": found,
        "all_present": all(found.values()) if found else False,
    }


def validate_document(
    ocr_result,
    quality_result,
    qr_result,
    image_path=""
):

    words = ocr_result.get(
        "words",
        []
    )

    # --------------------------------------------------
    # DOCUMENT DISPATCHER
    # --------------------------------------------------

    doc_type = detect_document_type(
        words
    )

    full_text = _clean(
        " ".join(
            str(w.get("text", ""))
            for w in words
        )
    ).lower()

    # --------------------------------------------------
    # FALLBACK DETECTION
    # --------------------------------------------------

    if doc_type == "unknown":

        if any(
            x in full_text
            for x in [
                "aadhaar",
                "aadhar",
                "uidai",
            ]
        ):
            doc_type = "aadhaar"

        elif (
            "income tax" in full_text
            or "permanent account" in full_text
            or re.search(
                r"\b[a-z]{5}\d{4}[a-z]\b",
                full_text,
                re.IGNORECASE
            )
        ):
            doc_type = "pan"

        elif (
            "driving licence" in full_text
            or "driving license" in full_text
            or "driving" in full_text
        ):
            doc_type = "license"

    # --------------------------------------------------
    # BASIC CHECKS
    # --------------------------------------------------

    # Quality is advisory.
    # We don't reject a document merely because the
    # photograph has blur/lighting issues.
    image_ok = True

    ocr_ok = (
        ocr_result.get(
            "word_count",
            0
        ) > 0
    )

    qr_detected = (
        qr_result.get(
            "count",
            0
        ) > 0
    )

    # --------------------------------------------------
    # UNKNOWN DOCUMENT
    # --------------------------------------------------

    if doc_type == "unknown":

        return {
            "document_type": "Unknown",
            "validation_result": "FAIL",
            "checks": {
                "image_quality_acceptable":
                    image_ok,

                "ocr_successful":
                    ocr_ok,

                "number_region_detected":
                    False,

                "expected_id_format":
                    "Unknown",

                "detected_id_length":
                    0,

                "required_fields_present":
                    False,

                "qr_code_detected":
                    qr_detected,
            },
        }

    config = DOCUMENT_CONFIG[
        doc_type
    ]

    # --------------------------------------------------
    # ID
    # --------------------------------------------------

    number_check = _find_id_number(
        words,
        config["expected_digits"],
        doc_type=doc_type
    )

    # --------------------------------------------------
    # REQUIRED FIELDS
    # --------------------------------------------------

    fields = _detect_required_fields(
        words,
        config["required_fields"],
        doc_type=doc_type
    )

    detected_value = ""

    if number_check["best"]:

        detected_value = (
            number_check["best"]["digits"]
        )

    # --------------------------------------------------
    # EXPECTED FORMAT
    # --------------------------------------------------

    if doc_type == "aadhaar":

        expected_format = "12 digits"

    elif doc_type == "pan":

        expected_format = (
            "5 letters + 4 digits + 1 letter"
        )

    else:

        expected_format = (
            "State/RTO prefix + licence number"
        )

    # --------------------------------------------------
    # CHECKS
    # --------------------------------------------------

    checks = {

        "image_quality_acceptable":
            image_ok,

        "ocr_successful":
            ocr_ok,

        "number_region_detected":
            number_check["found"],

        "detected_id":
            detected_value or "",

        "expected_id_format":
            expected_format,

        "detected_id_length":
            len(detected_value),

        "required_fields_present":
            fields["all_present"],

        "qr_code_detected":
            qr_detected,
    }

    # --------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------

    validation_result = (

        "PASS"

        if (
            image_ok
            and ocr_ok
            and number_check["found"]
            and fields["all_present"]
        )

        else "FAIL"
    )

    return {
        "document_type":
            config["name"],

        "validation_result":
            validation_result,

        "checks":
            checks,
    }


def print_report(
    report,
    fraud_result=None
):

    if "checks" not in report:

        return "Invalid Report"

    c = report["checks"]

    final_decision = "FAILED"

    if report.get(
        "validation_result"
    ) == "PASS":

        risk = (
            fraud_result.get(
                "predicted_risk"
            )
            if fraud_result
            else None
        )

        if risk == "LOW":

            final_decision = "VERIFIED"

        elif risk == "MEDIUM":

            final_decision = (
                "MANUAL REVIEW REQUIRED"
            )

        elif risk == "HIGH":

            final_decision = (
                "FRAUD SUSPECTED"
            )

        else:

            final_decision = "VERIFIED"

    lines = [

        "=" * 60,

        "DOCUMENT VERIFICATION REPORT",

        "=" * 60,

        f"Detected Type       : "
        f"{report.get('document_type', 'Unknown')}",

        f"OCR Successful      : "
        f"{'YES' if c.get('ocr_successful') else 'NO'}",

        f"Image Quality OK    : "
        f"{'YES' if c.get('image_quality_acceptable') else 'NO'}",

        f"ID Number Detected  : "
        f"{'YES' if c.get('number_region_detected') else 'NO'}",

        f"Detected ID         : "
        f"{c.get('detected_id') or '-'}",

        f"Expected ID Format  : "
        f"{c.get('expected_id_format', 'Unknown')}",

        f"Detected ID Length  : "
        f"{c.get('detected_id_length', 0)}",

        f"Required Fields OK  : "
        f"{'YES' if c.get('required_fields_present') else 'NO'}",

        f"QR Code Detected    : "
        f"{'YES' if c.get('qr_code_detected') else 'NO'}",

        "-" * 60,

        f"Validation Result   : "
        f"{report.get('validation_result', 'FAIL')}",
    ]

    if fraud_result:

        lines.append(
            f"Fraud Risk          : "
            f"{fraud_result.get('predicted_risk', 'N/A')}"
        )

    lines.extend([

        "=" * 60,

        f"FINAL DECISION      : "
        f"{final_decision}",

        "=" * 60,
    ])

    return "\n".join(lines)
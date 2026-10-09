"""
Field extraction module for documents (PAN, Aadhaar, Driving Licence).
Extracts key-value pairs (Name, Father's Name, DOB, Document Number, etc.)
from OCR results to display in the UI.
"""

import re
from typing import Dict, Any


def group_words_into_lines(words: list) -> list:
    """Group OCR word bounding boxes into distinct horizontal lines."""
    lines = []
    valid_words = [w for w in words if w.get("bbox")]
    sorted_words = sorted(
        valid_words,
        key=lambda x: (x["bbox"]["y"], x["bbox"]["x"])
    )
    for w in sorted_words:
        placed = False
        for line in lines:
            avg_y = sum(item["bbox"]["y"] for item in line) / len(line)
            avg_h = sum(item["bbox"]["h"] for item in line) / len(line)
            if abs(w["bbox"]["y"] - avg_y) <= max(16, avg_h * 0.5):
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    line_strings = [
        " ".join(w["text"] for w in sorted(line, key=lambda x: x["bbox"]["x"])).strip()
        for line in lines
    ]
    return line_strings


def extract_document_fields(ocr_result: dict, doc_type: str, detected_id: str = "") -> Dict[str, str]:
    """
    Extract structured fields for display.
    Returns an ordered dict of {field_label: field_value}.
    """
    words = ocr_result.get("words", [])
    full_text = ocr_result.get("text", "")
    lines = group_words_into_lines(words)

    dtype = str(doc_type).lower()
    fields = {}

    if "pan" in dtype:
        # 1. Name & Father's Name
        name_val = ""
        father_val = ""

        # Strategy A: Explicit labels
        name_m = re.search(r"(?:Name|Holder)\s*[:\-]?\s*([A-Za-z\s\.\']+?)(?=\s*Father|\s*DOB|\s*Birth|\s*PAN|\s*$)", full_text, re.IGNORECASE)
        father_m = re.search(r"Father(?:'s)?(?:\s*Name)?\s*[:\-]?\s*([A-Za-z\s\.\']+?)(?=\s*DOB|\s*Birth|\s*PAN|\s*$)", full_text, re.IGNORECASE)

        if name_m and father_m:
            name_val = name_m.group(1).strip()
            father_val = father_m.group(1).strip()
        else:
            # Strategy B: Common PAN layout (Header lines -> Name -> Father's Name -> DOB)
            stopwords = {
                "income", "tax", "department", "govt", "government", "india",
                "permanent", "account", "number", "pan", "card", "signature",
                "sample", "immihelp.com", "immihelp", "sitaeee", "teats", "form"
            }
            candidate_names = []
            for line in lines:
                clean_l = re.sub(r"[^A-Za-z\s]", " ", line).strip()
                tokens = clean_l.lower().split()
                if not tokens or len(tokens) < 1:
                    continue
                if any(t in stopwords for t in tokens) or "income tax" in clean_l.lower():
                    continue
                if re.search(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b", line):
                    break
                # A line with only uppercase letters and spaces that looks like a name
                if re.match(r"^[A-Za-z\s\.\']+$", line) and len(clean_l) >= 3 and not re.search(r"\d", line):
                    candidate_names.append(line.strip())

            if len(candidate_names) >= 2:
                name_val = candidate_names[0]
                father_val = candidate_names[1]
            elif len(candidate_names) == 1:
                name_val = candidate_names[0]

        # 2. Date of birth
        dob_val = ""
        dob_m = re.search(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})\b", full_text)
        if dob_m:
            dob_val = dob_m.group(1)

        fields["Name"] = name_val if name_val else "Not detected"
        fields["Father's name"] = father_val if father_val else "Not detected"
        fields["Date of birth"] = dob_val if dob_val else "Not detected"
        fields["PAN"] = detected_id if detected_id else "Not detected"

    elif "aadhaar" in dtype or "aadhar" in dtype:
        name_val = ""
        name_m = re.search(r"Name\s*[:\-]?\s*([A-Za-z\s\.\']+?)(?=\s*Number|\s*DOB|\s*Birth|\s*Gender|\s*YOB|\s*$)", full_text, re.IGNORECASE)
        if name_m:
            name_val = name_m.group(1).strip()
        else:
            for line in lines:
                clean_l = line.strip()
                if "aadhaar" in clean_l.lower() or "uidai" in clean_l.lower() or "government" in clean_l.lower():
                    continue
                if re.match(r"^[A-Za-z\s]+$", clean_l) and len(clean_l.split()) >= 2:
                    name_val = clean_l
                    break

        dob_val = ""
        dob_m = re.search(r"(?:DOB|Birth|YOB)\s*[:\-]?\s*(\S+)", full_text, re.IGNORECASE) or re.search(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})\b", full_text)
        if dob_m:
            dob_val = dob_m.group(1)

        gender_val = ""
        gen_m = re.search(r"Gender\s*[:\-]?\s*([A-Za-z]+)", full_text, re.IGNORECASE) or re.search(r"\b(Male|Female|Transgender)\b", full_text, re.IGNORECASE)
        if gen_m:
            gender_val = gen_m.group(1).capitalize()

        fields["Name"] = name_val if name_val else "Not detected"
        if gender_val:
            fields["Gender"] = gender_val
        fields["Date of birth"] = dob_val if dob_val else "Not detected"
        fields["Aadhaar No"] = detected_id if detected_id else "Not detected"

    elif "license" in dtype or "driving" in dtype:
        name_val = ""
        name_m = re.search(r"Name\s*[:\-]?\s*([A-Za-z\s\.\']+?)(?=\s*DL|\s*Licence|\s*License|\s*DOB|\s*Validity|\s*$)", full_text, re.IGNORECASE)
        if name_m:
            name_val = name_m.group(1).strip()

        dob_val = ""
        dob_m = re.search(r"(?:DOB|Birth)\s*[:\-]?\s*(\S+)", full_text, re.IGNORECASE) or re.search(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})\b", full_text)
        if dob_m:
            dob_val = dob_m.group(1)

        val_m = re.search(r"Validity\s*[:\-]?\s*([A-Za-z0-9/\-]+)", full_text, re.IGNORECASE)
        val_str = val_m.group(1).strip() if val_m else ""

        fields["Name"] = name_val if name_val else "Not detected"
        fields["Date of birth"] = dob_val if dob_val else "Not detected"
        if val_str:
            fields["Validity"] = val_str
        fields["Licence No"] = detected_id if detected_id else "Not detected"

    else:
        # Generic document
        fields["Detected ID"] = detected_id if detected_id else "Not detected"

    return fields

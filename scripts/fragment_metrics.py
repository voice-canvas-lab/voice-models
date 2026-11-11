"""Manual fragment annotations and occurrence-aware lexical retention scoring."""

import json

import jiwer

from evaluate import normalize

CATEGORIES = ("name", "number", "negation")
MATCH_RULES = (
    "Use the same normalization as WER/CER. Spans are zero-based half-open token "
    "indices in the normalized reference. Project that specific occurrence through "
    "the global minimum-edit word alignment, then compare the entire projected "
    "token sequence with the annotated reference or explicit variants. No substring "
    "search, numeric conversion, stemming or semantic interpretation. Interior "
    "insertions are included; outer-boundary insertions are excluded unless they "
    "complete an explicitly allowed variant without consuming neighboring reference tokens. "
    "A partial unequal-length substitution is ambiguous and counted as not preserved. "
    "Repeated identical text may have ambiguous alignments; the deterministic jiwer "
    "alignment is used. Counts are fragment-by-audio checks, not independent commands."
)


def load_annotations(path, samples):
    if path is None:
        return {}
    result = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"sample_id", "fragments"}:
            raise ValueError(f"Annotation line {line_no}: expected sample_id and fragments")
        sid = row["sample_id"]
        if not isinstance(sid, str) or sid not in samples or sid in result:
            raise ValueError(f"Annotation line {line_no}: unknown or duplicate sample_id {sid!r}")
        if not isinstance(row["fragments"], list):
            raise ValueError("fragments must be an array")
        reference = normalize(samples[sid]["text"]).split()
        ids = set()
        for fragment in row["fragments"]:
            required = {"fragment_id", "category", "start_token", "end_token", "reference"}
            if not isinstance(fragment, dict) or not required <= fragment.keys() or fragment.keys() - required - {"variants"}:
                raise ValueError(f"{sid}: invalid fragment fields")
            fid = fragment["fragment_id"]
            if not isinstance(fid, str) or not fid.strip() or fid in ids:
                raise ValueError(f"{sid}: invalid or duplicate fragment_id")
            ids.add(fid)
            if fragment["category"] not in CATEGORIES:
                raise ValueError(f"{sid}/{fid}: invalid category")
            start, end = fragment["start_token"], fragment["end_token"]
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)) or not 0 <= start < end <= len(reference):
                raise ValueError(f"{sid}/{fid}: invalid token span")
            if not isinstance(fragment["reference"], str) or normalize(fragment["reference"]).split() != reference[start:end]:
                raise ValueError(f"{sid}/{fid}: reference fragment does not match its annotated occurrence")
            variants = fragment.get("variants", [])
            if not isinstance(variants, list) or any(not isinstance(v, str) or not normalize(v) for v in variants):
                raise ValueError(f"{sid}/{fid}: variants must be nonempty strings")
        result[sid] = row["fragments"]
    return result


def project_span(chunks, start, end):
    ranges = []
    for chunk in chunks:
        a, b = chunk.ref_start_idx, chunk.ref_end_idx
        c, d = chunk.hyp_start_idx, chunk.hyp_end_idx
        if chunk.type == "insert":
            if start < a < end:
                ranges.append((c, d))
            continue
        left, right = max(start, a), min(end, b)
        if left >= right:
            continue
        if chunk.type == "equal" or b - a == d - c:
            ranges.append((c + left - a, c + right - a))
        elif chunk.type == "delete":
            ranges.append((c, c))
        elif start <= a and b <= end:
            ranges.append((c, d))
        else:
            return None
    return (min(a for a, b in ranges), max(b for a, b in ranges)) if ranges else None


def score_fragments(reference, prediction, fragments):
    ref, hyp = normalize(reference), normalize(prediction)
    alignment = jiwer.process_words(ref, hyp).alignments[0]
    tokens = hyp.split()
    details = []
    for fragment in fragments:
        span = project_span(alignment, fragment["start_token"], fragment["end_token"])
        observed = " ".join(tokens[span[0]:span[1]]) if span is not None else None
        variants = list(map(normalize, fragment.get("variants", [])))
        accepted = [normalize(fragment["reference"]), *variants]
        preserved = observed in accepted
        # One reference token may become several hypothesis tokens (21 -> twenty one).
        # Only zero-reference insertion chunks at the exact span boundaries may
        # extend a match, and the resulting text must equal an explicit variant.
        if not preserved and variants and span is not None and span[0] < span[1]:
            left, right = span
            for chunk in alignment:
                if chunk.type == "insert":
                    if chunk.ref_start_idx == fragment["start_token"] and chunk.hyp_end_idx == left:
                        left = chunk.hyp_start_idx
                    if chunk.ref_start_idx == fragment["end_token"] and chunk.hyp_start_idx == right:
                        right = chunk.hyp_end_idx
            for candidate in ((left, span[1]), (span[0], right), (left, right)):
                text = " ".join(tokens[candidate[0]:candidate[1]])
                if text in variants:
                    span, observed, preserved = candidate, text, True
                    break
        details.append({**fragment, "normalized_reference": normalize(fragment["reference"]),
                        "accepted_normalized": accepted, "hypothesis_token_span": list(span) if span is not None else None,
                        "observed_normalized": observed, "preserved": preserved,
                        "reason": "matched" if preserved else ("ambiguous_alignment" if span is None else "deleted" if not observed else "mismatch")})
    return details


def summarize(records):
    categories = {}
    for category in CATEGORIES:
        fragments = [f for row in records for f in row["fragments"] if f["category"] == category]
        count = len(fragments)
        saved = sum(f["preserved"] for f in fragments)
        categories[category] = {"status": "evaluated" if count else "not_evaluated",
                                "checked": count, "preserved": saved,
                                "retention": saved / count if count else None}
    return {"categories": categories, "audio_records": len(records),
            "annotated_audio_records": sum(bool(row["fragments"]) for row in records),
            "matching_rules": MATCH_RULES}


def report_lines(summary):
    lines = ["| Category | Checked fragments | Preserved | Retention |", "|---|---:|---:|---|"]
    for category, value in summary["categories"].items():
        rate = "не оценено" if value["retention"] is None else f"{value['retention']:.2%}"
        lines.append(f"| {category} | {value['checked']} | {value['preserved']} | {rate} |")
    return lines

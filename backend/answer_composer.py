"""Statement-level answer boundary for simple evidence answers."""


def _normal(value):
    return "".join(str(value or "").split()).replace("。", "").replace("，", "")


def validate_statement_plan(draft, allowed):
    known_ids = {evidence_id for bucket in ("allowed_answer_facts", "allowed_possibilities") for item in allowed.get(bucket, []) for evidence_id in item.get("evidence_ids", [])}
    failures = []
    for statement in draft.get("statements", []):
        evidence_ids = set(statement.get("evidence_ids", []))
        if not evidence_ids.issubset(known_ids):
            failures.append({"text": statement.get("text"), "reason": "unknown_evidence_id"})
            continue
        if statement.get("condition_keys"):
            allowed_conditions = {item.get("condition_key") for bucket in ("allowed_answer_facts", "allowed_possibilities") for item in allowed.get(bucket, []) if evidence_ids.intersection(item.get("evidence_ids", []))}
            if not set(statement["condition_keys"]).issubset(allowed_conditions):
                failures.append({"text": statement.get("text"), "reason": "semantic_boundary"})
        else:
            supported_texts = [_normal(item.get("text")) for bucket in ("allowed_answer_facts", "allowed_possibilities") for item in allowed.get(bucket, []) if evidence_ids.intersection(item.get("evidence_ids", []))]
            unknown_texts = [_normal(item.get("text")) for item in allowed.get("required_unknowns", []) if not item.get("evidence_ids")]
            statement_text = _normal(statement.get("text"))
            if statement.get("status") == "unknown" and any(text and (text in statement_text or statement_text in text) for text in unknown_texts):
                pass
            elif not any(text and (text in statement_text or statement_text in text) for text in supported_texts):
                failures.append({"text": statement.get("text"), "reason": "semantic_boundary"})
        if statement.get("status") == "matched" and not any(item.get("status") == "matched" and evidence_ids.intersection(item.get("evidence_ids", [])) for item in allowed.get("allowed_answer_facts", [])):
            failures.append({"text": statement.get("text"), "reason": "overstated_status"})
    return type("StatementValidation", (), {"valid": not failures, "failures": failures})()


def compose_answer(draft, allowed):
    checked = validate_statement_plan(draft, allowed)
    if checked.valid:
        return {"answer": str(draft.get("answer") or ""), "statements": draft.get("statements", []), "valid": True}
    facts = [item["text"] for item in allowed.get("allowed_answer_facts", []) if item.get("status") == "matched"]
    unknowns = [item["text"] for item in allowed.get("required_unknowns", [])]
    parts = facts + unknowns
    answer = "；".join(parts)
    answer = (answer + "；" if answer else "") + "目前无法确认这部分是否成立。"
    return {"answer": answer, "statements": [], "valid": False, "validation_failures": checked.failures}

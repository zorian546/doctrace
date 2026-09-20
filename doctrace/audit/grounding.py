"""
Claim-level grounding score.

An answer is graded in plain, inspectable steps:
1. The answer is broken into atomic factual claims that can each be checked alone.
2. Every claim is tested against the retrieved passages with a strict true/false prompt.
3. The score is supported_claims / total_claims, computed in Python and not by the model.
"""

import re

from doctrace.answering.local_model import complete as local_complete

ABSTAIN_MARKER = "don't have enough information in the provided context"


def is_abstention(answer: str) -> bool:
    """True when the answer is the model declining to answer.

    Looks for a short stable phrase rather than demanding the whole sentence, so
    small wording drift does not hide a refusal.
    """
    # models often emit a typographic apostrophe, which would defeat the match
    normalized = answer.lower().replace("’", chr(39)).replace("‘", chr(39))
    return ABSTAIN_MARKER in normalized


def split_into_claims(answer: str) -> list[str]:
    """Turn an answer into atomic, individually checkable statements.

    Only call this on answers that already passed the is_abstention() check.
    """
    prompt = f"""Break the following answer into atomic, independently checkable \
factual claims, one claim per line, with no numbering or bullets. Each line \
must be a complete, grammatically standalone sentence -- do not split a \
single sentence across multiple lines.

Example (for format only, not the real topic below):
Answer: "Bread is made by mixing flour and water, then baking it in an oven."
Output:
Bread is made by mixing flour and water.
Bread is baked in an oven.

Now do the same for this actual answer:

Answer:
\"\"\"{answer}\"\"\"

Output (one complete claim per line):
"""
    raw_text = local_complete(None, prompt, max_new_tokens=500)
    return _clean_claim_lines(raw_text)


def _clean_claim_lines(raw_text: str) -> list[str]:
    """Parse the model's one-claim-per-line output.

    Splits on newlines, removes bullets and numbering, drops blanks, and glues
    consecutive lines together until one ends in terminal punctuation (. ! ?) or a
    closing quote/backtick. That guards against the model wrapping one sentence
    across two output lines.
    """
    text = raw_text.strip()
    if not text:
        return []

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if line:
            lines.append(line)

    claims = []
    pending = ""
    for line in lines:
        pending = f"{pending} {line}".strip() if pending else line
        if pending.endswith((".", "!", "?")) or pending.rstrip().endswith(("`", '"')):
            claims.append(pending)
            pending = ""
    if pending:
        claims.append(pending)

    return claims


def verify_claims(claims: list[str], evidence: list[dict]) -> list[dict]:
    """Check each claim against the evidence. One model call per claim, never
    batched."""
    if not claims:
        return []

    context_text = "\n\n---\n\n".join(p["text"] for p in evidence)

    verdicts = []
    for claim in claims:
        prompt = f"""Context:
\"\"\"{context_text}\"\"\"

Claim: "{claim}"

Is this claim directly supported by the context above? Be strict: the \
claim must actually follow from the context, not just be plausible or \
generally true.

Respond with ONLY the single word true or false, no other text.
"""
        reply = local_complete(None, prompt, max_new_tokens=10).strip().lower()
        if "true" in reply and "false" not in reply:
            supported = True
        elif "false" in reply:
            supported = False
        else:
            raise ValueError(
                f"Could not parse true/false from model response for claim "
                f"{claim!r}. Raw response: {reply!r}"
            )
        verdicts.append({"claim": claim, "supported": supported})

    return verdicts


def grounding_score(answer: str, evidence: list[dict]) -> dict:
    """Whole procedure: rule out a refusal with a plain string match first (no model
    call needed), then split into claims, verify each, and divide in code."""
    if is_abstention(answer):
        return {"score": None, "claims": [], "abstained": True}

    claims = split_into_claims(answer)
    if not claims:
        return {"score": None, "claims": [], "abstained": True}

    verdicts = verify_claims(claims, evidence)
    supported = sum(1 for v in verdicts if v["supported"])
    return {"score": supported / len(verdicts), "claims": verdicts, "abstained": False}

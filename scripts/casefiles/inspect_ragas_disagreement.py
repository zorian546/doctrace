"""
Case file: why RAGAS scored some answers lower than the claim-level grounding score.

Steps through RAGAS's two internal stages (statement extraction, then an NLI verdict
per statement) on chosen disagreement cases, so the gap can be explained instead of
written off as a black box.

Run with: python -m scripts.casefiles.inspect_ragas_disagreement
"""

import asyncio
import json

from doctrace.search.semantic import semantic_search
from doctrace.audit.goldset import load_gold_pairs
from doctrace.audit.ragas_bridge import LocalRagasJudge
from ragas.metrics import faithfulness

TOP_K = 5
CASE_IDS = ["sh_014", "sh_019", "mh_019"]


async def inspect_case(pid: str, question: str, answer: str, context_texts: list[str]) -> None:
    print(f"\n{'=' * 70}\n{pid}\nQ: {question}\n{'=' * 70}")
    print(f"\nAnswer:\n  {answer}")

    row = {"question": question, "answer": answer, "contexts": context_texts}

    p_value = faithfulness._create_statements_prompt(row)
    statements_result = await faithfulness.llm.generate(p_value)
    from ragas.metrics._faithfulness import _statements_output_parser
    parsed_statements = await _statements_output_parser.aparse(
        statements_result.generations[0][0].text, p_value, faithfulness.llm, faithfulness.max_retries
    )
    if parsed_statements is None:
        print("\n  RAGAS statement extraction returned None (parse failure)")
        return
    statements = [s for item in parsed_statements.dicts() for s in item["simpler_statements"]]

    print(f"\nRAGAS-extracted statements ({len(statements)}):")
    for s in statements:
        print(f"  - {s}")

    p_value2 = faithfulness._create_nli_prompt(row, statements)
    nli_result = await faithfulness.llm.generate(p_value2)
    from ragas.metrics._faithfulness import _faithfulness_output_parser
    faith_result = await _faithfulness_output_parser.aparse(
        nli_result.generations[0][0].text, p_value2, faithfulness.llm, faithfulness.max_retries
    )
    if faith_result is None:
        print("\n  RAGAS NLI verdict step returned None (parse failure)")
        return

    print("\nRAGAS NLI verdicts:")
    for item in faith_result.dicts():
        mark = "TRUE " if item["verdict"] == 1 else "FALSE"
        print(f"  [{mark}] {item['statement']}")
        print(f"         reason: {item.get('reason', '(none given)')}")


async def main() -> None:
    faithfulness.llm = LocalRagasJudge()

    generation_results = {
        r["id"]: r for r in json.loads(open("data/processed/generation_report.json", encoding="utf-8").read())
    }
    gold_pairs = {p["id"]: p for p in load_gold_pairs()}

    for pid in CASE_IDS:
        question = gold_pairs[pid]["question"]
        answer = generation_results[pid]["answer"]
        retrieved = semantic_search(question, top_k=TOP_K)
        context_texts = [c["text"] for c in retrieved]
        await inspect_case(pid, question, answer, context_texts)


if __name__ == "__main__":
    asyncio.run(main())
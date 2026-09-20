"""
CLI for the injection probe: planted-document and prompt-injection benchmark.

Run with:
    python -m scripts.bench_injection
"""

import json
from pathlib import Path
from doctrace.audit.injection_probe import run_injection_suite

REPORT_PATH = Path("data/processed/injection_report.json")


def main() -> None:
    print("=" * 80)
    print("INJECTION PROBE: PLANTED-DOCUMENT SCENARIOS")
    print("=" * 80)

    results = run_injection_suite()

    print(f"\nScenarios run:                 {results['total_scenarios']}")
    print(f"Planted passage reached top-k:  {results['topk_reach_rate'] * 100:.1f}%")
    print(f"Attack landed rate:             {results['attack_landed_rate'] * 100:.1f}%")

    print("\n" + "-" * 80)
    for sc in results["scenarios"]:
        print(f"\nScenario [{sc['scenario_id']}] ({sc['attack_type']})")
        print(f"  Target Query:       {sc['target_query']}")
        print(f"  Reached top-k:      {'YES' if sc['tainted_reached_topk'] else 'NO'}")
        print(f"  Marker in answer:   {'YES (LANDED)' if sc['marker_in_answer'] else 'NO (HELD)'}")
        print(f"  Attack landed:      {'YES [LANDED]' if sc['attack_landed'] else 'NO [HELD]'}")
        print(f"  Generated Answer:   {sc['generated_answer'][:150]}...")
    print("\n" + "=" * 80)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote injection report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

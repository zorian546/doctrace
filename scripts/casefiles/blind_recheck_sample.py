"""
Draws a random sample of gold pairs for a blind second look.

After all 80 pairs were written, roughly 10 are re-read after a break to catch
labelling mistakes before they harden into "ground truth". One such mistake (sh_014)
had already surfaced by accident during unrelated debugging.

Run with: python -m scripts.casefiles.blind_recheck_sample
"""

import random

from doctrace.audit.goldset import load_gold_pairs

SAMPLE_SIZE = 10
SEED = 42  # fixed seed so the sample is reproducible, not re-rolled if run again


def main() -> None:
    gold_pairs = load_gold_pairs()
    rng = random.Random(SEED)
    sample = rng.sample(gold_pairs, SAMPLE_SIZE)

    print(f"Blind re-check sample ({SAMPLE_SIZE} pairs, seed={SEED}):\n")
    print("Answer each of these COLD, using only the actual FastAPI docs "
          "(not the stored answer below) -- then compare afterward.\n")

    for i, pair in enumerate(sample, start=1):
        print(f"{i}. [{pair['id']}] [{pair['category']}]")
        print(f"   Q: {pair['question']}\n")

    print("=" * 70)
    print("STORED ANSWERS (do not look until you've answered all above):")
    print("=" * 70)
    for i, pair in enumerate(sample, start=1):
        print(f"\n{i}. [{pair['id']}]")
        print(f"   A: {pair['answer']}")
        print(f"   Source: {pair['source_chunks']}")


if __name__ == "__main__":
    main()
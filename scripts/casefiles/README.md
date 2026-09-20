# Case files

One-off investigation scripts, separate from the regular indexing, benchmark and smoke-test scripts in `scripts/`. Each answers a specific question that came up while working on retrieval and answering, and the finding it produced is written up in `ENGINEERING_NOTES.md`.

They are kept so the code behind a claim in the notes can still be read and run. They are not maintained as tools, and nothing in the package, the other scripts or the tests imports them.

| Script | Question it answered |
|---|---|
| `diagnose_fusion_regression.py` | Why the fused pipeline trailed semantic-only search; the cross-encoder was demoting canonical passages |
| `inspect_reranker_demotion.py` | The cross-encoder's heading-vocabulary bias (the sh_009 and mh_012 cases) |
| `inspect_multihop_failures.py` | Where multi-hop grounding scores lost points |
| `inspect_ragas_disagreement.py` | The largest gaps between RAGAS and the claim-level grounding score |
| `check_mh015_retrieval.py` | Whether mh_015's source passages were in its retrieved context |
| `inspect_mh015_refusal.py` | Root cause of the mh_015 and mh_025 over-refusals |
| `inspect_local_model_regressions.py` | Where Qwen2.5-3B regressed against the Claude run |
| `rescore_with_fixed_refusal_check.py` | Re-scoring after the refusal detector was fixed |
| `blind_recheck_sample.py` | The blind re-check of 10 of the 80 gold pairs (seed 42) |

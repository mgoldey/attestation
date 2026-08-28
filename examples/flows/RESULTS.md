# Example flows: results measured 2026-08-28

## Persona eval (mode=live, chat=gemma4:e2b-it-q4_K_M, embed=embeddinggemma, items=40)

Agreement with `corpus/labels.json`; evidence about the flow, not a model benchmark.

| persona | precision | recall | AUC (signed confidence) | tp/fp/fn/tn | unsure | rank AUC | classifier AUC | s/reaction |
|---|---|---|---|---|---|---|---|---|
| bench-chemist | 0.846 | 0.917 | 0.921 | 11/2/1/26 | 0 | 1.000 | 0.854 (40 clicks) | 3.40 |
| ml-engineer | 0.923 | 1.000 | 1.000 | 12/1/0/27 | 0 | 1.000 | 0.938 (40 clicks) | 2.76 |

Confidence histograms: bench-chemist {'4': 1, '5': 39}; ml-engineer {'3': 1, '4': 1, '5': 38}

## MCP end to end (mode=live, chat=gemma4:e2b-it-q4_K_M)

127 calls over stdio across feed / provenance / knowledge / symbolic / full; 0 failed.

| surface | tool | result |
|---|---|---|
| feed | feed.personas | ok |
| feed | feed.persona_create | ok |
| feed | feed.persona_update | ok |
| feed | feed.persona_suggest_interests | ok |
| feed | feed.persona_status | ok |
| feed | feed.list | ok |
| feed | feed.list | ok |
| feed | feed.search | ok |
| feed | feed.read | ok |
| feed | feed.explain | ok |
| feed | feed.rate | ok |
| feed | feed.rate | ok |
| feed | feed.harvest_engagement | ok |
| feed | feed.simulate_ratings | ok |
| feed | feed.digest | ok |
| feed | feed.ask | ok |
| feed | feed.ask | ok |
| feed | feed.ask | refused |
| feed | feed.persona_reset | ok |
| feed | feed.persona_delete | ok |
| feed | feed.persona_delete | refused |
| feed | feed.sources | ok |
| feed | feed.source_preview | ok |
| feed | feed.source_add | ok |
| feed | feed.source_suggest | ok |
| feed | feed.source_remove | ok |
| feed | feed.tools | ok |
| feed | <list_tools> | ok |
| provenance | runs.scan | ok |
| provenance | runs.list | ok |
| provenance | runs.list | ok |
| provenance | runs.compare | ok |
| provenance | runs.detail | ok |
| provenance | runs.claims_coverage | ok |
| provenance | runs.claims_check | ok |
| provenance | runs.ask | ok |
| provenance | runs.ask | ok |
| provenance | runs.ask | refused |
| provenance | cite.check | ok |
| provenance | runs.tools | ok |
| provenance | <list_tools> | ok |
| knowledge | feed.search | ok |
| knowledge | kg.concepts | ok |
| knowledge | kg.central | ok |
| knowledge | kg.communities | ok |
| knowledge | kg.neighbors | ok |
| knowledge | kg.path | ok |
| knowledge | kg.ask | ok |
| knowledge | kg.ask | refused |
| knowledge | cite.sources | ok |
| knowledge | cite.lookup | refused |
| knowledge | cite.search | ok |
| knowledge | cite.check | ok |
| knowledge | kg.tools | ok |
| knowledge | <list_tools> | ok |
| symbolic | sym.simplify | ok |
| symbolic | sym.solve | ok |
| symbolic | sym.solve | refused |
| symbolic | sym.differentiate | ok |
| symbolic | sym.integrate | ok |
| symbolic | sym.derivation | ok |
| symbolic | sym.verify | ok |
| symbolic | sym.evaluate | ok |
| symbolic | sym.simplify | refused |
| symbolic | sym.ask | ok |
| symbolic | sym.ask | refused |
| symbolic | sym.tools | ok |
| symbolic | <list_tools> | ok |
| full | feed.personas | ok |
| full | feed.persona_create | ok |
| full | feed.persona_update | ok |
| full | feed.persona_suggest_interests | ok |
| full | feed.persona_status | ok |
| full | feed.list | ok |
| full | feed.list | ok |
| full | feed.search | ok |
| full | feed.read | ok |
| full | feed.explain | ok |
| full | feed.rate | ok |
| full | feed.rate | ok |
| full | feed.harvest_engagement | ok |
| full | feed.simulate_ratings | ok |
| full | feed.digest | ok |
| full | feed.ask | ok |
| full | feed.ask | ok |
| full | feed.ask | refused |
| full | feed.persona_reset | ok |
| full | feed.persona_delete | ok |
| full | feed.persona_delete | refused |
| full | feed.sources | ok |
| full | feed.source_preview | ok |
| full | feed.source_add | ok |
| full | feed.source_suggest | ok |
| full | feed.source_remove | ok |
| full | runs.scan | ok |
| full | runs.list | ok |
| full | runs.list | ok |
| full | runs.compare | ok |
| full | runs.detail | ok |
| full | runs.claims_coverage | ok |
| full | runs.claims_check | ok |
| full | runs.ask | ok |
| full | runs.ask | ok |
| full | runs.ask | refused |
| full | kg.concepts | ok |
| full | kg.central | ok |
| full | kg.communities | ok |
| full | kg.neighbors | ok |
| full | kg.path | ok |
| full | kg.ask | ok |
| full | kg.ask | refused |
| full | cite.sources | ok |
| full | cite.lookup | refused |
| full | cite.search | ok |
| full | cite.check | ok |
| full | sym.simplify | ok |
| full | sym.solve | ok |
| full | sym.solve | refused |
| full | sym.differentiate | ok |
| full | sym.integrate | ok |
| full | sym.derivation | ok |
| full | sym.verify | ok |
| full | sym.evaluate | ok |
| full | sym.simplify | refused |
| full | sym.ask | ok |
| full | sym.ask | refused |
| full | <list_tools> | ok |

## Training family `c_sweep` (mlflow-skinny, 1.9 s for 4 arms)

| C | accuracy | precision | recall | AUC |
|---|---|---|---|---|
| 0.01 | 0.9561 | 0.9467 | 0.9861 | 0.9864 |
| 0.1 | 0.9649 | 0.9474 | 1.0000 | 0.9940 |
| 1.0 | 0.9825 | 0.9730 | 1.0000 | 0.9957 |
| 10.0 | 0.9737 | 0.9726 | 0.9861 | 0.9960 |

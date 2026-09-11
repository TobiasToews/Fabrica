
## Assembly Generalist Policy (AG) — env_generalist_18-18-12-03 (2000 epochs)

┌──────────────────┬───────────────────────────────┬────────────────┬───────────────────┬────────────┬───────┐
│     Assembly     │              Run              │ SDF Resolution │ Insertion Success │ Paper (AG) │  Gap  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ car              │ _18-18-12-03                  │ 420            │ 96.48%            │ 60.55%     │ +35.9 │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ beam             │ _18-18-12-03                  │ 420            │ 92.97%            │ 98.83%     │ −5.9  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ duct             │ _18-18-12-03                  │ 420            │ 92.58%            │ 89.84%     │ +2.7  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ gamepad          │ _18-18-12-03                  │ 420            │ 73.93%            │ 71.48%     │ +2.4  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ plumbers_block   │ _18-18-12-03                  │ 420            │ 71.09%            │ 81.64%     │ −10.5 │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ cooling_manifold │ _18-18-12-03                  │ 420            │ 66.02%            │ 89.06%     │ −23.0 │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ stool_circular   │ _18-18-12-03                  │ 420            │ 61.72%            │ 58.59%     │ +3.1  │
└──────────────────┴───────────────────────────────┴────────────────┴───────────────────┴────────────┴───────┘

Paper (AG) values are Table 3's "Assembly Generalist Policy (AG)" row from the Fabrica paper
(https://fabrica.csail.mit.edu/static/pdf/paper.pdf) — % of successful steps without
intervention in simulation, across 1024 random trials, for a policy trained jointly on all
parts from all benchmark assemblies. All 7 rows use the same checkpoint (single generalist
policy evaluated per-assembly), unlike the AS table above where each assembly has its own
specialist checkpoint.


## Generalization onto own assembiles: 
- loaded exp_socket_set and exp_human_normal before evaluting on the exp_generalist checkpoint
-> kind of checking as the precomputed grasps from exp_socket_set and exp_human_normal where used -> hence half of the work is already done? -> however this is also done for the original fabrica parts 

┌──────────────────┬───────────────────────────────┬────────────────┬───────────────────┬────────────┬───────┐
│     Assembly     │              Run              │ SDF Resolution │ Insertion Success │ Paper (AG) │  Gap  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ socket_set       │ _18-18-12-03                  │ 420            │ 78.91%            │ N/A        │  N/A  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ human_normal     │ _18-18-12-03                  │ 420            │ 82.10%            │ N/A        │  N/A  │
└──────────────────┴───────────────────────────────┴────────────────┴───────────────────┴────────────┴───────┘

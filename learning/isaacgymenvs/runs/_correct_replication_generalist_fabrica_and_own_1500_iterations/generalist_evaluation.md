
## Assembly Generalist Policy (AG) — env_generalist_05-11-48-09 (1500 epochs)

┌──────────────────┬───────────────────────────────┬────────────────┬───────────────────┬────────────┬───────┐
│     Assembly     │              Run              │ SDF Resolution │ Insertion Success │ Paper (AG) │  Gap  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ car              │ _05-11-48-09                  │ 420            │ 95.7%             │ 60.6%      │ +35.1 │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ beam             │ _05-11-48-09                  │ 420            │ 94.7%             │ 98.8%      │ −4.1  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ duct             │ _05-11-48-09                  │ 420            │ 89.2%             │ 89.8%      │ −0.6  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ gamepad          │ _05-11-48-09                  │ 420            │ 73.7% (old: 89.2%)│ 71.5%      │ -2.2  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ plumbers_block   │ _05-11-48-09                  │ 420            │ 85.7%             │ 81.6%      │ +4.2  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ cooling_manifold │ _05-11-48-09                  │ 420            │ 69.8%             │ 89.1%      │ −19.3 │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ stool_circular   │ _05-11-48-09                  │ 420            │ 55.1%             │ 58.6%      │ −3.5  │
└──────────────────┴───────────────────────────────┴────────────────┴───────────────────┴────────────┴───────┘


## Generalization onto own assembiles: 
- loaded exp_socket_set and exp_human_normal before evaluting on the exp_generalist checkpoint
-> kind of checking as the precomputed grasps from exp_socket_set and exp_human_normal where used -> hence half of the work is already done? -> however this is also done for the original fabrica parts 
-> you would need to insert assembly pairs into the fabrica_pairs.yaml file -> hence copy them from generalist own -> not so sure if this is the right appoach

┌──────────────────┬───────────────────────────────┬────────────────┬───────────────────┬────────────┬───────┐
│     Assembly     │              Run              │ SDF Resolution │ Insertion Success │ Paper (AG) │  Gap  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ socket_set       │ _05-11-48-09                  │ 420            │ %            │ N/A        │  N/A  │
├──────────────────┼───────────────────────────────┼────────────────┼───────────────────┼────────────┼───────┤
│ human_normal     │ _05-11-48-09                  │ 420            │ %            │ N/A        │  N/A  │
└──────────────────┴───────────────────────────────┴────────────────┴───────────────────┴────────────┴───────┘


python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["socket_set"] task.env.numEnvs=1024 max_iterations=1 headless=True test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_correct_replication_generalist_fabrica_and_own_1500_iterations/env_generalist_05-11-48-09/nn/env_generalist.pth"

python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["human_normal"] task.env.numEnvs=1024 max_iterations=1 headless=True test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_correct_replication_generalist_fabrica_and_own_1500_iterations/env_generalist_05-11-48-09/nn/env_generalist.pth"


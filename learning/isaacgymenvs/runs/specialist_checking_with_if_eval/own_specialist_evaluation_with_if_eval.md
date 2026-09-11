Specialist Policies Own
┌────────────────┬────────────────┬──────────────────┬─────────────────────┬──────────────┬───────┐
│    Assembly    │      Run       │  SDF Resolution  │  Insertion Success  │  Paper (AS)  │  Gap  │
├────────────────┼────────────────┼──────────────────┼─────────────────────┼──────────────┼───────┤
│   socket_set   │  _15-15-21-07  │       512        │        96.5%        │     N/A      │  N/A  │
├────────────────┼────────────────┼──────────────────┼─────────────────────┼──────────────┼───────┤
│  human_normal  │  _16-22-20-04  │       512        │        81.9%        │     N/A      │  N/A  │
└────────────────┴────────────────┴──────────────────┴─────────────────────┴──────────────┴───────┘

Specialist_
python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["socket_set"] task.env.numEnvs=1024 max_iterations=1 headless=True test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_own_assemblies/exp_socket_15-15-21-07/nn/exp_socket.pth"

python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["human_normal"] task.env.numEnvs=1024 max_iterations=1 headless=True test=True task.env.if_eval=True task.env.franka_friction=1 checkpoint="runs/_own_assemblies/exp_human_normal_16-22-20-04/nn/exp_human_normal.pth"
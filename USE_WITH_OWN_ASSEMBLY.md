# How to include, train, and test your own assembly

> Status: this workflow is based on the current loading and planning code paths, but it has not been end-to-end verified with a brand-new custom assembly in this session.

The steps below reflect the actual pipeline used by the repository code, including the asset-loading and SDF-generation flow.

## 1. Prepare the asset format

Create a new folder at:

```bash
assets/fabrica/<my_assembly>/
```

Put one OBJ file per part in that folder, for example:

```text
assets/fabrica/<my_assembly>/0.obj
assets/fabrica/<my_assembly>/1.obj
assets/fabrica/<my_assembly>/2.obj
```

A few important details:

- Part IDs are derived from the file names, so they do not need to be sequential.
- Each mesh must already be positioned and oriented in a shared world coordinate frame.
- Mesh units should be in meters.
- If all OBJ files are loaded together with no extra transform, they should already appear as a correctly assembled object. This is the key assumption used by the loader.

Optional: add a config file named `config.json` with:

```json
{"contact_eps": 0.2}
```

This is only needed for special cases such as the car and duct assemblies. If omitted, the pipeline uses a sensible default.

## 2. Generate SDF files

Run:

```bash
conda activate fabrica
cd ~/Fabrica
python assets/make_sdf.py --dir assets/fabrica/<my_assembly>
```

This step voxelizes each OBJ into a `.sdf` collision file, matching the format already used by the built-in assemblies.

## 3. Run planning

```bash
bash ./planning/run_planning.sh <EXP_NAME> <my_assembly>
```

The planning pipeline is geometry-driven, so this should work the same way as it does for the built-in assemblies such as beam, car, or duct.

## 4. Prepare Isaac Gym assets

```bash
conda activate isaacgym
bash ./learning/preprocessing/prepare_isaac.sh <EXP_NAME>
```

This script automatically picks up every subdirectory under `assets/fabrica/`, so your new assembly should be included alongside the existing built-in ones without extra configuration.

## 5. Train

```bash
cd ~/Fabrica/learning/isaacgymenvs
python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["<my_assembly>"] task.env.numEnvs=1024 max_iterations=1500 headless=True experiment=my_exp
```

Passing `task.env.assemblies` overrides the default built-in assembly list, so a genuinely new assembly name is supported.

## 6. Test or run inference

```bash
python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["<my_assembly>"] task.env.numEnvs=1 headless=True test=True checkpoint=runs/my_exp/nn/my_exp.pth
```

## Notes

The guidance above is based on reading the repository implementation rather than on a completed end-to-end run with a brand-new custom assembly. If you already have a CAD model available, the first two steps are the most important ones to validate carefully before spending time on training.
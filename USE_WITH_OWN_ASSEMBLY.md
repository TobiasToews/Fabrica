# How to include, train, and test your own assembly

> Status: this workflow is based on the repository’s current loading and planning code paths. It has not been end-to-end verified with a brand-new custom assembly in this session.

## Quick summary

To add a new assembly, you need to:

1. Place one OBJ file per part in a new folder under assets/fabrica/.
2. Generate SDF collision files from those meshes.
3. Run planning, prepare Isaac Gym assets, and then train or evaluate with your assembly name.

## 1. Prepare the asset folder

Create a new directory at:

```bash
assets/fabrica/<my_assembly>/
```

Put one OBJ file per part into that folder, for example:

```text
assets/fabrica/<my_assembly>/0.obj
assets/fabrica/<my_assembly>/1.obj
assets/fabrica/<my_assembly>/2.obj
```

Important requirements:

- Part IDs are derived from the file names, so they do not need to be sequential.
- Each mesh must already be positioned and oriented in a shared world coordinate frame.
- Mesh units should be in meters.
- If all OBJ files are loaded together with no extra transform, they should already look like a correctly assembled object. This is the key assumption used by the loader.

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

The planning pipeline is geometry-driven, so it should work the same way as it does for the built-in assemblies such as beam, car, or duct.

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

## Useful external datasets

If you want to source example assemblies rather than preparing your own CAD files, these are the most relevant options:

- Assemble-Them-All: the closest fit to Fabrica’s lineage and format. It provides assembled OBJ meshes and is a strong starting point for insertion-style or translational assembly tasks.
- Fusion 360 Gallery Assembly Dataset: larger and more diverse, but it typically requires more conversion work to fit Fabrica’s expected folder layout.
- NIST Assembly Task Boards: very relevant for peg-in-hole and insertion-only scenarios, though they are less plug-and-play as a bulk dataset.

For a first pass, Assemble-Them-All is the most practical choice because it is closest to the expected mesh format and requires the least adaptation.
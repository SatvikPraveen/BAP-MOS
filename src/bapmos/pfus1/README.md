# PFUS1 modules live under `bapmos.preprocess.bladder`

This package is a **compatibility shim**. Prefer the preprocess paths in new code:

```bash
python -m bapmos.preprocess.bladder --help
python -m bapmos.preprocess.bladder.convert_json_polygons_to_masks --help
python -m bapmos.preprocess.bladder.create_splits --help
```

`from bapmos.pfus1 import …` still works for older imports.

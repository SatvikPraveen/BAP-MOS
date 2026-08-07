"""Shim — prefer ``bapmos.preprocess.bladder.convert_json_polygons_to_masks``."""
from bapmos.preprocess.bladder.convert_json_polygons_to_masks import *  # noqa: F401,F403
from bapmos.preprocess.bladder.convert_json_polygons_to_masks import main

if __name__ == "__main__":
    raise SystemExit(main())

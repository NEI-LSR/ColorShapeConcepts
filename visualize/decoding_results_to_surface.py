from bin.functional_to_surface import functional2surface
import os


CONTENT_ROOT = "../data"
SUBJECTS = ["jeeves"]
MODEL = "../results/models/jeeves_FLS"

_set_types = ["color", "shape"]

if not os.path.exists(MODEL):
    raise FileExistsError("Model directory not found. Run searchlight first.")

# iterate through subjects
for subject in SUBJECTS:
    subj_dir = os.path.join(MODEL, subject)

    for inset in _set_types:
        set_dir = os.path.join(subj_dir, inset)
        if not os.path.exists(set_dir):
            continue
        out_dir = os.path.join(set_dir, "surface_overlays")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        files = os.listdir(set_dir)
        for f in files:
            if ".nii.gz" in f and "reg" not in f:
                fname = os.path.join(set_dir, f)
                functional2surface(subject, fname, proj_root="../data", output_dir=out_dir)
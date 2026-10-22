python3 - << 'PY'
import glob
import numpy as np
paths = sorted(glob.glob("data/rollout_*.npz"))
for path in paths:
    episode = np.load(path)
    action = episode["action"]
    smooth = np.abs(np.diff(action, axis=0)).mean() if len(action) > 1 else 0.0
    print(f"{path}  {episode['reason']}  {len(action) * 0.1:.1f}s  smoothness {smooth:.3f}")
PY
import numpy as np
import jaxtyping as jt

def position_rotation2pose(
    positions: jt.Float[np.ndarray, "n 3"], rotations: jt.Float[np.ndarray, "n 3 3"]
) -> jt.Float[np.ndarray, "n 4 4"]:
    batched_input = len(positions.shape) == 2

    if not batched_input:
        positions = positions[None]
        rotations = rotations[None]

    n_poses = positions.shape[0]

    poses = np.zeros((n_poses, 4, 4), dtype=float)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = positions
    poses[:, 3, 3] = 1

    if not batched_input:
        poses = poses[0]

    return poses

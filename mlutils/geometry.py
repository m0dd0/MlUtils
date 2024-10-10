from scipy.spatial.transform import Rotation
import numpy as np

from mlutils.array_typing import NpArray


def position_rotation2pose(
    positions: NpArray["n,3", float], rotations: NpArray["n,3,3", float]
) -> NpArray["n,4,4", float]:
    batched_input = len(positions.shape) == 2

    if not batched_input:
        position = position[None]
        rotations = rotations[None]

    n_poses = position.shape[0]

    poses = np.zeros((n_poses, 4, 4), dtype=float)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = position
    poses[:, 3, 3] = 1

    if not batched_input:
        poses = poses[0]

    return poses

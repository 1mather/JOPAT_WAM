"""Rotation conversion helpers used by the DROID data converter."""

import torch


def euler_angles_to_rot_6d(
    euler_angles: torch.Tensor,
    convention: str = "XYZ",
) -> torch.Tensor:
    """Convert Euler angles in radians to a 6D rotation representation."""
    matrix = euler_angles_to_matrix(euler_angles, convention)
    return matrix_to_rotation_6d(matrix)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the first-two-rows 6D representation."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation matrices with shape (..., 3, 3), got {matrix.shape}."
        )
    batch_shape = matrix.shape[:-2]
    return matrix[..., :2, :].clone().reshape(batch_shape + (6,))


def euler_angles_to_matrix(
    euler_angles: torch.Tensor,
    convention: str,
) -> torch.Tensor:
    """Convert Euler angles in radians to rotation matrices."""
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Euler angles must have shape (..., 3).")
    if len(convention) != 3:
        raise ValueError("The Euler convention must contain three letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid Euler convention: {convention}.")
    if any(letter not in "XYZ" for letter in convention):
        raise ValueError(f"Invalid Euler convention: {convention}.")

    matrices = [
        _axis_angle_rotation(axis, angle)
        for axis, angle in zip(
            convention,
            torch.unbind(euler_angles, dim=-1),
        )
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        flattened = (
            one,
            zero,
            zero,
            zero,
            cosine,
            -sine,
            zero,
            sine,
            cosine,
        )
    elif axis == "Y":
        flattened = (
            cosine,
            zero,
            sine,
            zero,
            one,
            zero,
            -sine,
            zero,
            cosine,
        )
    elif axis == "Z":
        flattened = (
            cosine,
            -sine,
            zero,
            sine,
            cosine,
            zero,
            zero,
            zero,
            one,
        )
    else:
        raise ValueError(f"Invalid rotation axis: {axis}.")
    return torch.stack(flattened, dim=-1).reshape(angle.shape + (3, 3))



import nibabel as nib
import numpy as np

# Load the tumor mask and its voxel-to-RAS transformation
nii = nib.load("tumor_mask.nii.gz")
mask = np.asarray(nii.dataobj) > 0
affine = nii.affine

print("=== NIFTI MASK ===")
print("Shape (voxels):", mask.shape)
print("Voxel spacing (mm):", nii.header.get_zooms()[:3])
print("Affine (voxel -> RAS):")
print(affine)

indices = np.argwhere(mask)

if len(indices) == 0:
    raise ValueError("Tumor mask is empty!")

idx_min = indices.min(axis=0)
idx_max = indices.max(axis=0)

print("Tumor voxel count:", len(indices))
print("Voxel index min:", idx_min)
print("Voxel index max:", idx_max)

# Transform the eight corners of the occupied voxel-index bounds.
# These are voxel-center bounds, not outer surface bounds.
corners = np.array([
    [i, j, k, 1]
    for i in [idx_min[0], idx_max[0]]
    for j in [idx_min[1], idx_max[1]]
    for k in [idx_min[2], idx_max[2]]
])

ras = (affine @ corners.T).T[:, :3]

print("Mask center-bound RAS min:", ras.min(axis=0))
print("Mask center-bound RAS max:", ras.max(axis=0))

# Read the downloaded OBJ vertex coordinates
vertices = []
with open("tumor_mesh.obj", encoding="utf-8") as f:
    for line in f:
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])

vertices = np.asarray(vertices)

print("\n=== OBJ MESH ===")
print("Vertex count:", len(vertices))
print("Mesh RAS min:", vertices.min(axis=0))
print("Mesh RAS max:", vertices.max(axis=0))
print("Mesh extent:", vertices.max(axis=0) - vertices.min(axis=0))

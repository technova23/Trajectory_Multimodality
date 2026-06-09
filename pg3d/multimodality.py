import zarr
import numpy as np
from scipy.spatial.distance import pdist
from collections import defaultdict

DATASET = "debug_multimodal_test.zarr"

root = zarr.open(DATASET, mode="r")

# --------------------------------------------------
# Load data
# --------------------------------------------------

eef_pos = root["data"]["eef_pos"][:]  # (N,3)
family_ids = root["data"]["trajectory_family_id"][:]

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)

print("Samples:", len(eef_pos))
print("Families:", np.unique(family_ids))

# --------------------------------------------------
# FAMILY BALANCE
# --------------------------------------------------

print("\nFAMILY DISTRIBUTION")

unique, counts = np.unique(family_ids, return_counts=True)

for fam, cnt in zip(unique, counts):
    print(f"Family {fam:2d}: {cnt:8d}")

balance_ratio = counts.max() / counts.min()

print(f"\nFamily imbalance ratio: {balance_ratio:.2f}")

# --------------------------------------------------
# XYZ COVERAGE
# --------------------------------------------------

mins = eef_pos.min(axis=0)
maxs = eef_pos.max(axis=0)

print("\nXYZ COVERAGE")

print("X:", mins[0], "->", maxs[0], "range =", maxs[0] - mins[0])
print("Y:", mins[1], "->", maxs[1], "range =", maxs[1] - mins[1])
print("Z:", mins[2], "->", maxs[2], "range =", maxs[2] - mins[2])

# --------------------------------------------------
# VOXEL COVERAGE
# --------------------------------------------------

voxel_size = 0.02

voxels = np.floor(eef_pos / voxel_size).astype(int)

unique_voxels = np.unique(voxels, axis=0)

occupied_volume = len(unique_voxels) * voxel_size**3

print("\nVOXEL COVERAGE")

print("Occupied voxels:", len(unique_voxels))
print("Approx occupied volume:", occupied_volume, "m^3")

# --------------------------------------------------
# TRAJECTORY DISTANCES
# --------------------------------------------------

print("\nPAIRWISE DISTANCE")

max_points = 3000

if len(eef_pos) > max_points:
    idx = np.random.choice(len(eef_pos), max_points, replace=False)
    sample = eef_pos[idx]
else:
    sample = eef_pos

distances = pdist(sample)

print("Mean pairwise distance:", distances.mean())
print("Std pairwise distance:", distances.std())
print("Min pairwise distance:", distances.min())
print("Max pairwise distance:", distances.max())

# --------------------------------------------------
# FAMILY SEPARATION
# --------------------------------------------------

print("\nFAMILY SEPARATION")

family_points = defaultdict(list)

for p, f in zip(eef_pos, family_ids):
    family_points[int(f)].append(p)

family_centroids = {}

for f, pts in family_points.items():
    pts = np.asarray(pts)
    family_centroids[f] = pts.mean(axis=0)

centroids = np.stack(list(family_centroids.values()))

family_separation = pdist(centroids)

print("Mean family centroid distance:",
      family_separation.mean())

print("Min family centroid distance:",
      family_separation.min())

print("Max family centroid distance:",
      family_separation.max())

# --------------------------------------------------
# DIVERSITY SCORE
# --------------------------------------------------

xyz_volume = np.prod(maxs - mins)

coverage_score = min(
    occupied_volume / max(xyz_volume, 1e-9),
    1.0
)

distance_score = distances.mean()

separation_score = family_separation.mean()

print("\n" + "=" * 80)
print("DIVERSITY REPORT")
print("=" * 80)

print(f"Coverage score       : {coverage_score:.4f}")
print(f"Trajectory spread    : {distance_score:.4f}")
print(f"Family separation    : {separation_score:.4f}")

overall = (
    0.4 * coverage_score +
    0.3 * distance_score +
    0.3 * separation_score
)

print(f"\nOverall diversity score: {overall:.4f}")

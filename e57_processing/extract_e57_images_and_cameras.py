#!/usr/bin/env python3
"""
Extract images and cameras from E57, apply corrections, and generate trajectory.
Directly outputs cameras.json (with corrections applied) and traj_w_c.txt.

Usage:
    python extract_e57_images_and_cameras.py <e57_path> [--out_dir <dir>]
"""

import os
import json
import numpy as np
import pye57
import copy
import argparse


# ============================================================
# Configuration for camera corrections
# ============================================================

# Number of cameras to process
NUM_CHECK = 408
GROUP = 6

# Reordering within each group
REMAP = [5, 3, 4, 1, 2, 0]

# Cameras needing 180° rotation (group-local indices)
ROTATE_180_IDX = {1, 2, 3, 4}

# 180° rotation matrix around camera Z axis
A_pi = np.array([
    [-1.0,  0.0,  0.0, 0.0],
    [ 0.0, -1.0,  0.0, 0.0],
    [ 0.0,  0.0,  1.0, 0.0],
    [ 0.0,  0.0,  0.0, 1.0],
], dtype=np.float64)


# ============================================================
# Helper functions for E57 extraction
# ============================================================

def node_to_float(x):
    """Convert E57 node to float"""
    try:
        return float(x)
    except TypeError:
        pass

    v = getattr(x, "value", None)
    if callable(v):
        return float(v())
    elif v is not None:
        return float(v)

    raise TypeError(f"Cannot convert node {x} (type {type(x)}) to float")


def quat_to_R(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix"""
    n = (qw**2 + qx**2 + qy**2 + qz**2) ** 0.5
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ], dtype=float)


def pinhole_to_K(pin):
    """Extract camera intrinsic matrix from pinhole representation"""
    f  = node_to_float(pin["focalLength"])
    px = node_to_float(pin["pixelWidth"])
    py = node_to_float(pin["pixelHeight"])
    cx = node_to_float(pin["principalPointX"])
    cy = node_to_float(pin["principalPointY"])

    fx = f / px
    fy = f / py

    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=float)


def save_blob_as_file(blob_node, out_path):
    """Save E57 blob to file"""
    byte_count = blob_node.byteCount()
    buf = np.zeros(byte_count, dtype=np.uint8)
    blob_node.read(buf, 0, byte_count)
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())


def find_image_blob_and_type(image2d):
    """Find image blob and projection type in E57 Image2D"""
    rep_candidates = [
        ("pinholeRepresentation", "pinhole"),
        ("sphericalRepresentation", "spherical"),
        ("cylindricalRepresentation", "cylindrical"),
        ("visualReferenceRepresentation", "visual"),
    ]

    for rep_key, proj_label in rep_candidates:
        try:
            rep = image2d[rep_key]
        except Exception:
            continue

        for field in ("jpegImage", "pngImage"):
            try:
                blob = rep[field]
            except Exception:
                continue
            return blob, proj_label, rep_key

    for field in ("jpegImage", "pngImage"):
        try:
            blob = image2d[field]
            return blob, "unknown", "image2D"
        except Exception:
            continue

    return None, None, None


# ============================================================
# Main extraction and processing
# ============================================================

def extract_images_and_cameras(e57_path, out_dir=None):
    """
    Extract images and cameras from E57, apply corrections, and generate outputs.
    
    Outputs:
        - rgb/rgb_*.png: Extracted images
        - cameras.json: Camera parameters (with corrections applied)
        - traj_w_c.txt: Trajectory file
    """
    if out_dir is None:
        base = os.path.splitext(os.path.basename(e57_path))[0]
        out_dir = os.path.join(os.path.dirname(e57_path), base + "_images")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Create rgb subdirectory
    rgb_dir = os.path.join(out_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    print("="*60)
    print("E57 EXTRACTION AND PROCESSING")
    print("="*60)
    print(f"Input:  {e57_path}")
    print(f"Output: {out_dir}\n")

    # ========================================
    # Step 1: Extract from E57
    # ========================================
    print("[1/3] Extracting images and cameras from E57...")
    
    e57 = pye57.E57(e57_path)
    root = e57.image_file.root()

    images2d = root["images2D"]
    print(f"      Found {len(images2d)} Image2D entries")

    raw_cameras = []
    cam_id = 0

    for e57_idx in range(len(images2d)):
        image2d = images2d[e57_idx]
        
        # Progress indicator
        if (e57_idx + 1) % 10 == 0 or e57_idx == 0 or e57_idx == len(images2d) - 1:
            print(f"      Processing Image2D {e57_idx + 1}/{len(images2d)} ({(e57_idx + 1)*100//len(images2d)}%)")

        pose = image2d["pose"]
        rot = pose["rotation"]
        trans = pose["translation"]

        qw = node_to_float(rot["w"])
        qx = node_to_float(rot["x"])
        qy = node_to_float(rot["y"])
        qz = node_to_float(rot["z"])

        tx = node_to_float(trans["x"])
        ty = node_to_float(trans["y"])
        tz = node_to_float(trans["z"])

        R = quat_to_R(qw, qx, qy, qz)
        t = np.array([[tx], [ty], [tz]])

        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, 3:4] = t
        T_wc = np.linalg.inv(T_cw)

        K = None
        projection = None
        try:
            pin = image2d["pinholeRepresentation"]
            K = pinhole_to_K(pin)
            projection = "pinhole"
        except Exception:
            projection = None

        blob, proj_from_blob, rep_key = find_image_blob_and_type(image2d)
        if blob is None:
            print(f"      WARNING: Image2D {e57_idx} has no blob, skipping")
            continue

        if projection is None:
            projection = proj_from_blob or "unknown"

        # Save image as rgb_{cam_id}.png
        img_name = f"rgb_{cam_id}.png"
        img_path = os.path.join(rgb_dir, img_name)
        save_blob_as_file(blob, img_path)

        cam_info = {
            "index": int(cam_id),
            "e57_index": int(e57_idx),
            "image_path": img_path,
            "projection": projection,
            "rep_key": rep_key,
            "T_cw": T_cw.tolist(),
            "T_wc": T_wc.tolist(),
        }
        if K is not None:
            cam_info["K"] = K.tolist()

        raw_cameras.append(cam_info)
        cam_id += 1

    print(f"      Extracted {len(raw_cameras)} cameras\n")

    # ========================================
    # Step 2: Apply corrections
    # ========================================
    print("[2/3] Applying camera corrections...")
    
    # Sort by index to ensure stable ordering
    cams = sorted(raw_cameras, key=lambda x: x["index"])
    
    corrected_cameras = []
    
    # Process only first NUM_CHECK cameras
    num_to_process = min(NUM_CHECK, len(cams))
    num_groups = num_to_process // GROUP
    print(f"      Processing {num_to_process} cameras ({num_groups} groups of {GROUP})")
    
    for block_start in range(0, num_to_process, GROUP):
        block = cams[block_start:block_start + GROUP]
        
        if len(block) != GROUP:
            print(f"      WARNING: Incomplete group at {block_start}, skipping remaining")
            break
        
        # Reorder within group according to REMAP
        for new_local_idx, old_local_idx in enumerate(REMAP):
            cam = copy.deepcopy(block[old_local_idx])
            
            # Renumber index (0, 1, 2, ..., 407)
            cam["index"] = block_start + new_local_idx
            
            # Apply 180° rotation correction if needed
            if old_local_idx in ROTATE_180_IDX:
                if "T_cw" in cam:
                    T_cw = np.array(cam["T_cw"], dtype=np.float64)
                    T_cw = T_cw @ A_pi  # Apply 180° rotation around Z
                    cam["T_cw"] = T_cw.tolist()
                    cam["T_wc"] = np.linalg.inv(T_cw).tolist()
            
            corrected_cameras.append(cam)
    
    print(f"      Corrected {len(corrected_cameras)} cameras\n")

    # ========================================
    # Step 3: Save outputs
    # ========================================
    print("[3/3] Saving outputs...")
    
    # Save cameras.json (corrected version)
    cameras_json_path = os.path.join(out_dir, "cameras.json")
    with open(cameras_json_path, "w", encoding="utf-8") as f:
        json.dump({"cameras": corrected_cameras}, f, indent=2)
    print(f"      ✓ cameras.json")
    
    # Generate and save traj_w_c.txt
    traj_path = os.path.join(out_dir, "traj_w_c.txt")
    with open(traj_path, "w") as f:
        for cam in sorted(corrected_cameras, key=lambda x: x["index"]):
            T_cw = np.array(cam["T_cw"], dtype=np.float64)
            assert T_cw.shape == (4, 4), f"Bad T_cw shape for camera {cam['index']}"
            
            # Row-major flattening
            vals = T_cw.reshape(-1)
            line = " ".join(f"{v:.18e}" for v in vals)
            f.write(line + "\n")
    print(f"      ✓ traj_w_c.txt")

    # Summary
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Output files:")
    print(f"  • {os.path.join(out_dir, 'rgb/')}           ({len(corrected_cameras)} images)")
    print(f"  • {cameras_json_path}     ({len(corrected_cameras)} cameras)")
    print(f"  • {traj_path}      ({len(corrected_cameras)} poses)")
    print("="*60 + "\n")

    return corrected_cameras, cameras_json_path, traj_path


# ============================================================
# Command line interface
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and process cameras from E57 file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("e57_path", 
                       help="Path to E57 file")
    parser.add_argument("--out_dir", default=None,
                       help="Output directory (default: <e57_basename>_images)")
    
    args = parser.parse_args()
    
    extract_images_and_cameras(args.e57_path, args.out_dir)

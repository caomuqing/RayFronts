# Calibration Files

RayFronts needs calibration that is not stored in this bag.

The actual calibration files for this bag are stored here:

- `intrinsics.txt`
- `extrinsics.conf`

The code expects:

- `intrinsics_file`: any JSON-readable file with camera intrinsics.
- `extrinsics_conf_path`: a ModalAI `extrinsics.conf`.

`rgb_intrinsics.json` should contain:

```json
{
  "fx": 0.0,
  "fy": 0.0,
  "cx": 0.0,
  "cy": 0.0,
  "w": 0,
  "h": 0,
  "dist_coeffs": [0.0, 0.0, 0.0, 0.0]
}
```

`dist_coeffs` is optional if images are already rectified.

`extrinsics.conf` should be the ModalAI extrinsics file with entries for:

- `child: "hires_front"`
- `child: "tof"`

Both entries must be connected to `parent: "body"` directly or through a chain.

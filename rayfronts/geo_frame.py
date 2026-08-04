"""Geographic frame utilities for the RayFronts frontier-mapping frame.

Adapted from compass_planner_gps.py. Establishes the transform between the
drone's local odometry frame and a mission-level *frontier-mapping frame*
defined by a global origin (lat/lon/alt) and a heading.

Frame conventions (identical to compass_planner_gps.py):
  - Local / odometry frame: ENU at "home" (x=East, y=North, z=Up). This is the
    frame of ``/robot_1/odometry_conversion/odometry`` and, after the
    RDF->FLU remap in FrontierBehavior, the frame the ``global_plan`` is
    published in.
  - Frontier-mapping frame: ENU at (origin_lat, origin_lon, origin_alt) rotated
    counter-clockwise about Up by ``heading_deg`` (heading=0 -> frame == ENU).

The local frame origin ("home") is unknown in global coordinates, so it is
estimated online from GPS + odometry (see ``update_home``) exactly as the
compass planner does, then locked once successive estimates agree.
"""

import numpy as np

# WGS84 ellipsoid constants
_WGS84_A = 6378137.0          # semi-major axis (m)
_WGS84_E2 = 0.00669437999014  # first eccentricity squared


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
  """WGS84 geodetic coordinates -> ECEF [X, Y, Z] (metres)."""
  lat = np.radians(lat_deg)
  lon = np.radians(lon_deg)
  n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
  x = (n + alt_m) * np.cos(lat) * np.cos(lon)
  y = (n + alt_m) * np.cos(lat) * np.sin(lon)
  z = (n * (1.0 - _WGS84_E2) + alt_m) * np.sin(lat)
  return np.array([x, y, z])


def ecef_to_enu(ecef, ref_lat_deg, ref_lon_deg, ref_alt_m):
  """ECEF -> ENU relative to a reference geodetic point [e, n, u] (metres)."""
  ref_lat = np.radians(ref_lat_deg)
  ref_lon = np.radians(ref_lon_deg)
  ref_ecef = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
  d = ecef - ref_ecef
  sl, cl = np.sin(ref_lat), np.cos(ref_lat)
  slo, clo = np.sin(ref_lon), np.cos(ref_lon)
  e = -slo * d[0] + clo * d[1]
  n = -sl * clo * d[0] - sl * slo * d[1] + cl * d[2]
  u = cl * clo * d[0] + cl * slo * d[1] + sl * d[2]
  return np.array([e, n, u])


def enu_to_ecef(enu, ref_lat_deg, ref_lon_deg, ref_alt_m):
  """ENU (relative to reference) -> ECEF [X, Y, Z] (metres)."""
  ref_lat = np.radians(ref_lat_deg)
  ref_lon = np.radians(ref_lon_deg)
  sl, cl = np.sin(ref_lat), np.cos(ref_lat)
  slo, clo = np.sin(ref_lon), np.cos(ref_lon)
  ref_ecef = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
  e, n, u = enu[0], enu[1], enu[2]
  dx = -slo * e - sl * clo * n + cl * clo * u
  dy = clo * e - sl * slo * n + cl * slo * u
  dz = cl * n + sl * u
  return ref_ecef + np.array([dx, dy, dz])


def ecef_to_geodetic(ecef):
  """ECEF [X, Y, Z] -> WGS84 geodetic (lat_deg, lon_deg, alt_m).

  Uses Bowring's iterative method (converges in a few iterations).
  """
  x, y, z = ecef
  lon_rad = np.arctan2(y, x)
  p = np.hypot(x, y)
  lat_rad = np.arctan2(z, p * (1.0 - _WGS84_E2))
  for _ in range(5):
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat_rad) ** 2)
    lat_rad = np.arctan2(z + _WGS84_E2 * n * np.sin(lat_rad), p)
  n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat_rad) ** 2)
  cl = np.cos(lat_rad)
  alt_m = (p / cl - n) if abs(cl) > 1e-10 else (
      abs(z) / np.sin(lat_rad) - n * (1.0 - _WGS84_E2))
  return np.degrees(lat_rad), np.degrees(lon_rad), alt_m


def points_in_polygon(pts_xy, poly_xy):
  """Even-odd ray-casting point-in-polygon test, vectorized over points.

  Args:
    pts_xy: (N, 2) array of query points.
    poly_xy: (M, 2) array of polygon vertices in order (open or closed).
  Returns:
    (N,) bool array, True where the point lies inside the polygon.
  """
  pts_xy = np.asarray(pts_xy, dtype=np.float64)
  poly_xy = np.asarray(poly_xy, dtype=np.float64)
  x, y = pts_xy[:, 0], pts_xy[:, 1]
  inside = np.zeros(pts_xy.shape[0], dtype=bool)
  m = poly_xy.shape[0]
  for i in range(m):
    x1, y1 = poly_xy[i]
    x2, y2 = poly_xy[(i + 1) % m]
    if y1 == y2:
      continue
    crosses = ((y1 > y) != (y2 > y)) & \
              (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    inside ^= crosses
  return inside


class FrontierFrame:
  """Transform between the local odometry frame and a global frontier frame.

  Attributes:
    origin_lat/lon/alt: WGS84 origin of the frontier-mapping frame.
    heading_deg: Frame X-axis, degrees CCW from East.
    home_fixed: True once the home (local-frame origin) estimate is locked.
  """

  def __init__(self, origin_lat, origin_lon, origin_alt, heading_deg,
               home_lock_thresh=0.1):
    self.origin_lat = origin_lat
    self.origin_lon = origin_lon
    self.origin_alt = origin_alt
    self.heading_deg = heading_deg
    self._home_lock_thresh = home_lock_thresh

    h = np.radians(heading_deg)
    # ENU -> frame: CCW rotation about Up by h.
    self._R_enu_to_frame = np.array([
        [np.cos(h), np.sin(h), 0.0],
        [-np.sin(h), np.cos(h), 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Home (local-frame origin) state, estimated online.
    self._home_lat = None
    self._home_lon = None
    self._home_alt = None
    self.home_fixed = False

    # Cached affine: frame_xyz = _affine_R @ local_xyz + _affine_t
    self._affine_R = None
    self._affine_t = None

  def is_ready(self):
    """True once a home estimate exists (transform can be applied)."""
    return self._home_lat is not None

  def gps_to_frame(self, lat_deg, lon_deg, alt_m=None):
    """WGS84 coordinates -> frontier frame [x, y, z] (metres).

    Depends only on the frame definition, not on the home estimate, so it is
    usable at construction time (e.g. to convert surveyed keepout corners).
    """
    if alt_m is None:
      alt_m = self.origin_alt
    enu = ecef_to_enu(geodetic_to_ecef(lat_deg, lon_deg, alt_m),
                      self.origin_lat, self.origin_lon, self.origin_alt)
    return self._R_enu_to_frame @ enu

  def update_home(self, gps_lat, gps_lon, gps_alt, odom_pos_enu):
    """Refine the home estimate from a GPS fix + local (ENU) odometry position.

    The local frame is ENU at home, so:
      robot_ecef = home_ecef + R_enu(home) @ odom_pos
      home_ecef  ~= robot_ecef - R_enu(gps) @ odom_pos
    (using the current GPS as a proxy for home in the rotation, sub-mm for
    missions < 1 km). Locks once successive estimates agree within the
    threshold. No-op once locked. Returns True if home is now fixed.
    """
    if self.home_fixed:
      return True

    robot_ecef = geodetic_to_ecef(gps_lat, gps_lon, gps_alt)
    ox, oy, oz = float(odom_pos_enu[0]), float(odom_pos_enu[1]), float(odom_pos_enu[2])

    rl = np.radians(gps_lat)
    rlo = np.radians(gps_lon)
    sl, cl = np.sin(rl), np.cos(rl)
    slo, clo = np.sin(rlo), np.cos(rlo)
    dx = -slo * ox - sl * clo * oy + cl * clo * oz
    dy = clo * ox - sl * slo * oy + cl * slo * oz
    dz = cl * oy + sl * oz

    home_ecef = robot_ecef - np.array([dx, dy, dz])
    lat, lon, alt = ecef_to_geodetic(home_ecef)

    prev = None
    if self._home_lat is not None:
      prev = geodetic_to_ecef(self._home_lat, self._home_lon, self._home_alt)

    self._home_lat = lat
    self._home_lon = lon
    self._home_alt = alt
    # Sync the frame origin altitude to home altitude (as the compass planner
    # does) so the planar x/y transform is not skewed by an altitude offset.
    self.origin_alt = alt
    self._recompute_affine()

    if prev is not None:
      change = float(np.linalg.norm(home_ecef - prev))
      if change < self._home_lock_thresh:
        self.home_fixed = True
    return self.home_fixed

  def set_home(self, lat_deg, lon_deg, alt_m):
    """Fixes the home (local-frame origin) directly from surveyed values.

    Alternative to the online GPS estimation (update_home) for robots whose
    local-frame origin is known at takeoff (e.g. a surveyed launch point),
    or for bag-replay tests where no live GPS fix is available.
    """
    self._home_lat = float(lat_deg)
    self._home_lon = float(lon_deg)
    self._home_alt = float(alt_m)
    # Sync the frame origin altitude to home altitude (as update_home does)
    # so the planar x/y transform is not skewed by an altitude offset.
    self.origin_alt = float(alt_m)
    self._recompute_affine()
    self.home_fixed = True

  def _local_to_frame_exact(self, pos_local):
    """Exact single-point local(ENU@home) -> frontier-frame transform."""
    ecef = enu_to_ecef(pos_local, self._home_lat, self._home_lon, self._home_alt)
    enu_origin = ecef_to_enu(ecef, self.origin_lat, self.origin_lon,
                             self.origin_alt)
    return self._R_enu_to_frame @ enu_origin

  def _recompute_affine(self):
    """Cache the local->frame transform as an affine (rotation + translation).

    The full chain (ENU@home -> ECEF -> ENU@origin -> frame rotation) is affine
    in the local position, so we recover it once by transforming the origin and
    three unit basis vectors. This keeps per-frontier filtering a cheap
    matrix-multiply instead of repeated geodetic math.
    """
    if self._home_lat is None:
      return
    p0 = self._local_to_frame_exact(np.array([0.0, 0.0, 0.0]))
    px = self._local_to_frame_exact(np.array([1.0, 0.0, 0.0]))
    py = self._local_to_frame_exact(np.array([0.0, 1.0, 0.0]))
    pz = self._local_to_frame_exact(np.array([0.0, 0.0, 1.0]))
    self._affine_R = np.column_stack([px - p0, py - p0, pz - p0])
    self._affine_t = p0

  def local_to_frame(self, pos_local):
    """Transform local-frame position(s) into the frontier frame.

    Args:
      pos_local: (3,) or (N, 3) array of positions in the local/odom frame.
    Returns:
      Array of the same shape in the frontier frame, or None if home is not
      yet estimated.
    """
    if self._affine_R is None:
      return None
    pos_local = np.asarray(pos_local, dtype=np.float64)
    return pos_local @ self._affine_R.T + self._affine_t

  def frame_to_local(self, pos_frame):
    """Inverse of ``local_to_frame``: frontier-frame position(s) -> local frame.

    Args:
      pos_frame: (3,) or (N, 3) array of positions in the frontier frame.
    Returns:
      Array of the same shape in the local/odom frame, or None if home is not
      yet estimated.
    """
    if self._affine_R is None:
      return None
    pos_frame = np.asarray(pos_frame, dtype=np.float64)
    return (pos_frame - self._affine_t) @ np.linalg.inv(self._affine_R).T

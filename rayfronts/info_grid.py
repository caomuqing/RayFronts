"""2D class-specific information grid keyed in the shared frontier frame.

Port of the low-flying drone's exploration-planner info grid (see
reference_only/RayFronts_low/rayfronts/exploration_planner.py,
"2D information grid" section) for the high-flying drone. Differences:

  - Cells are keyed directly in the frontier-mapping frame via a
    FrontierFrame (whose home estimate arrives online from GPS + odometry)
    instead of a statically georeferenced NED config. Until the home
    transform is ready, updates are no-ops.
  - Runs in the mapping server's main loop (no separate planner thread), so
    no locking is needed around the mapper's snapshot buffers.

Cell semantics (identical to the low drone so grids fuse across robots):
dict {(ix, iz): [state, confidence, y, gnd_votes, obs_votes, last_seen]}
with cell centers at (ix * cell_size, iz * cell_size) in frontier-frame xy.
state 1 = non-obstacle (ground-like), 2 = obstacle; unknown cells are absent.
A cell is an obstacle iff it has at least one obstacle vote AND
obs/(obs+gnd) >= obstacle_min_frac; confidence is the winning fraction.
y is the world-RDF display height (visualization only).
"""

import logging
import math
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)


class InfoGrid2D:
  """Accumulates per-voxel ground/obstacle votes into 2D frontier-frame cells.

  Attributes:
    info_grid: The cell dict described in the module docstring.
    cell_size: Cell edge length in meters (must match the peer robots').
    bounds: (min_x, max_x, min_y, max_y) in frontier-frame xy; voxels outside
      do not vote and the published grid window covers exactly this box.
  """

  def __init__(self, vox_size, geo_frame, bounds,
               cell_size=0.5,
               obstacle_max_height_voxels=10,
               obstacle_min_frac=0.5):
    """
    Args:
      vox_size: The mapper's voxel size in meters (column tests run at this
        resolution).
      geo_frame: FrontierFrame providing local->frontier-frame transforms.
      bounds: (min_x, max_x, min_y, max_y) frontier-frame xy bounds.
      cell_size: Info-grid cell size in meters.
      obstacle_max_height_voxels: Other-class voxels within this many voxels
        above a chosen-class voxel of the same lateral column vote obstacle;
        higher ones (e.g. canopy over ground) are ignored.
      obstacle_min_frac: Minimum obs/(obs+gnd) vote fraction for a cell to be
        classified obstacle.
    """
    self.vox_size = float(vox_size)
    self.geo_frame = geo_frame
    self.bounds = bounds
    self.cell_size = float(cell_size)
    self.obstacle_max_height_voxels = int(obstacle_max_height_voxels)
    self.obstacle_min_frac = float(obstacle_min_frac)

    self.info_grid = {}
    self._last_seq = 0

  # ---------- hashing helpers (voxel-resolution set membership) ----------

  @staticmethod
  def _cell_hash(xyz, res):
    """Full 3D voxel-cell hash at resolution res."""
    c = torch.round(xyz / res).long()
    m = 2 ** 20
    return (c[:, 0] * m + c[:, 1]) * m + c[:, 2]

  @staticmethod
  def _col_hash(xyz, res):
    """Lateral (x, z) column hash at resolution res."""
    c = torch.round(xyz[:, [0, 2]] / res).long()
    return c[:, 0] * (2 ** 20) + c[:, 1]

  @staticmethod
  def _member(query_h, sorted_h):
    """Per-row membership of query hashes in a sorted hash tensor."""
    if sorted_h.shape[0] == 0:
      return torch.zeros(query_h.shape[0], dtype=torch.bool)
    p = torch.searchsorted(sorted_h, query_h).clamp(max=sorted_h.shape[0] - 1)
    return sorted_h[p] == query_h

  # ---------- frame handling ----------

  def _frame_xy(self, pts_rdf):
    """World-RDF points (Nx3 CPU tensor) -> frontier-frame xy (Nx2 tensor).

    The mapper's world frame is RDF; the local/odom ENU frame is its
    [z, -x, -y] remap (same convention as FrontierBehavior), from which
    geo_frame's affine maps into the frontier frame.
    """
    local = np.stack([pts_rdf[:, 2].numpy(),
                      -pts_rdf[:, 0].numpy(),
                      -pts_rdf[:, 1].numpy()], axis=1)
    xy = self.geo_frame.local_to_frame(local)[:, :2]
    return torch.from_numpy(np.ascontiguousarray(xy)).float()

  def _in_bounds_mask(self, pts_rdf):
    """Boolean mask of RDF points inside the frontier-frame bounds box."""
    if self.bounds is None:
      return torch.ones(pts_rdf.shape[0], dtype=torch.bool)
    xy = self._frame_xy(pts_rdf)
    mnx, mxx, mny, mxy = self.bounds
    return ((xy[:, 0] >= mnx) & (xy[:, 0] <= mxx) &
            (xy[:, 1] >= mny) & (xy[:, 1] <= mxy))

  # ---------- update ----------

  def update(self, mapper):
    """Updates the grid from new keyframe visible-voxel snapshots.

    Per keyframe: chosen-class visible voxels with no other-class voxel
    beneath them (same lateral voxel column, checked against the GLOBAL
    labeled map) mark their cell non-obstacle; other-class visible voxels
    within obstacle_max_height_voxels above a chosen-class voxel in their
    column mark their cell obstacle. Obstacle wins cell conflicts.
    """
    if self.geo_frame is None or not self.geo_frame.is_ready():
      return
    snaps = [(s, v.detach().cpu().clone())
             for s, v in mapper.keyframe_visible if s > self._last_seq]
    if not snaps:
      return
    part = mapper.get_class_partition()
    if part is None:
      return
    chosen = part[0].detach().cpu().clone()
    other = part[1].detach().cpu().clone()
    if chosen.shape[0] == 0:
      return

    vox = self.vox_size
    # Global per-column structures (up = -y in world RDF):
    # - highest chosen voxel per column (min y)
    # - deepest other voxel per column (max y): any other BELOW a voxel v in
    #   the column exists iff this deepest y > y_v.
    ch_col = self._col_hash(chosen, vox)
    ch_u, ch_inv = torch.unique(ch_col, return_inverse=True)
    ch_top = torch.full((ch_u.shape[0],), torch.inf)
    ch_top.scatter_reduce_(0, ch_inv, chosen[:, 1], reduce="amin",
                           include_self=False)
    ot_deep_u = torch.empty(0, dtype=torch.long)
    ot_deep = torch.empty(0)
    if other.shape[0] > 0:
      ot_col = self._col_hash(other, vox)
      ot_deep_u, ot_inv = torch.unique(ot_col, return_inverse=True)
      ot_deep = torch.full((ot_deep_u.shape[0],), -torch.inf)
      ot_deep.scatter_reduce_(0, ot_inv, other[:, 1], reduce="amax",
                              include_self=False)

    ch_set = torch.sort(self._cell_hash(chosen, vox)).values
    ot_set = torch.sort(self._cell_hash(other, vox)).values \
      if other.shape[0] > 0 else torch.empty(0, dtype=torch.long)

    n_h = self.obstacle_max_height_voxels * vox
    # Debug counters: where do snapshot voxels drop out of the vote pipeline?
    n_vox = n_inb = n_chosen = n_other = n_gnd = n_obs = 0
    for seq, vis in snaps:
      self._last_seq = max(self._last_seq, seq)
      n_vox += vis.shape[0]
      # Only voxels inside the mission bounds vote.
      vis = vis[self._in_bounds_mask(vis)]
      n_inb += vis.shape[0]
      if vis.shape[0] == 0:
        continue
      vis_h = self._cell_hash(vis, vox)
      is_chosen = self._member(vis_h, ch_set)
      is_other = self._member(vis_h, ot_set) & ~is_chosen
      n_chosen += int(is_chosen.sum())
      n_other += int(is_other.sum())

      # Non-obstacle: chosen visible voxels with NO other-class beneath.
      vc = vis[is_chosen]
      if vc.shape[0] > 0:
        col = self._col_hash(vc, vox)
        has_other_below = torch.zeros(vc.shape[0], dtype=torch.bool)
        if ot_deep_u.shape[0] > 0:
          p = torch.searchsorted(ot_deep_u, col).clamp(
            max=ot_deep_u.shape[0] - 1)
          found = ot_deep_u[p] == col
          has_other_below = found & (ot_deep[p] > vc[:, 1] + 0.5 * vox)
        n_gnd += int((~has_other_below).sum())
        self._mark_cells(vc[~has_other_below], 1)

      # Obstacle: other visible voxels within n voxels above a chosen voxel
      # of the same column (chosen top strictly below the voxel).
      vo = vis[is_other]
      if vo.shape[0] > 0:
        col = self._col_hash(vo, vox)
        p = torch.searchsorted(ch_u, col).clamp(max=ch_u.shape[0] - 1)
        found = ch_u[p] == col
        top = ch_top[p]
        near_ground = (found & (top > vo[:, 1] + 0.5 * vox) &
                       (top - vo[:, 1] <= n_h + 1e-6))
        # Obstacle cells store their column's ground height so the grid can
        # render as a flat map draped on the ground (y is display-only).
        n_obs += int(near_ground.sum())
        self._mark_cells(vo[near_ground], 2, y_vals=top[near_ground])

    logger.info(
      "info_grid: %d snaps, vox %d -> in_bounds %d -> chosen %d / other %d "
      "-> gnd_votes %d / obs_votes %d; %d cells known.",
      len(snaps), n_vox, n_inb, n_chosen, n_other, n_gnd, n_obs,
      len(self.info_grid))

  def _mark_cells(self, pts, state, y_vals=None):
    """Adds one vote per voxel to its cell and refreshes the cell state.

    Cells accumulate ground/obstacle vote counts across voxels and
    keyframes. State becomes obstacle iff there is at least one obstacle
    vote AND obs/(obs+gnd) >= obstacle_min_frac, else non-obstacle.
    Confidence is the winning state's vote fraction. y_vals optionally
    overrides the stored display height per point (display only).
    """
    if pts.shape[0] == 0:
      return
    now = time.time()
    xy = self._frame_xy(pts)
    ix = torch.round(xy[:, 0] / self.cell_size).long()
    iz = torch.round(xy[:, 1] / self.cell_size).long()
    ys = pts[:, 1] if y_vals is None else y_vals
    for k in range(pts.shape[0]):
      key = (int(ix[k]), int(iz[k]))
      cell = self.info_grid.get(key)
      if cell is None:
        # state, conf, y, gnd votes, obs votes, last-observed walltime
        cell = [0, 0.0, float(ys[k]), 0, 0, now]
        self.info_grid[key] = cell
      if state == 2:
        cell[4] += 1
      else:
        cell[3] += 1
      cell[2] = float(ys[k])
      cell[5] = now
      frac_obs = cell[4] / (cell[3] + cell[4])
      if cell[4] > 0 and frac_obs >= self.obstacle_min_frac:
        cell[0] = 2
        cell[1] = frac_obs
      else:
        cell[0] = 1
        cell[1] = 1.0 - frac_obs

  # ---------- publishing ----------

  def _region_grid(self):
    """Info-grid index window covering the bounds.

    Returns (ix0, iz0, nx, nz): the window covers cells whose centers lie
    inside the bounds box; array index (i, j) maps to cell key
    (i + ix0, j + iz0).
    """
    mnx, mxx, mny, mxy = self.bounds
    g = self.cell_size
    ix0 = int(math.ceil(mnx / g))
    iz0 = int(math.ceil(mny / g))
    nx = max(1, int(math.floor(mxx / g)) - ix0 + 1)
    nz = max(1, int(math.floor(mxy / g)) - iz0 + 1)
    return ix0, iz0, nx, nz

  def build_coverage_msg(self, msg, stamp):
    """Fills a nav_msgs/OccupancyGrid with the grid over the bounds window.

    Wire format shared with the low drone's MAIPP comms: frontier-frame
    coordinates, -1 unknown, 0 ground/non-obstacle, 100 obstacle; cell
    centers at origin + (index + 0.5) * resolution.

    Args:
      msg: A fresh nav_msgs.msg.OccupancyGrid to fill.
      stamp: builtin_interfaces Time message for the header.
    """
    ix0, iz0, nx, nz = self._region_grid()
    g = self.cell_size
    grid = np.full((nz, nx), -1, dtype=np.int8)
    for (ix, iz), cell in self.info_grid.items():
      i, j = ix - ix0, iz - iz0
      if 0 <= i < nx and 0 <= j < nz:
        grid[j, i] = 100 if cell[0] == 2 else 0
    msg.header.stamp = stamp
    msg.header.frame_id = "frontier_frame"
    msg.info.resolution = float(g)
    msg.info.width = nx
    msg.info.height = nz
    msg.info.origin.position.x = (ix0 - 0.5) * g
    msg.info.origin.position.y = (iz0 - 0.5) * g
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.reshape(-1).tolist()
    return msg

"""MAIPP-style region/track task selection for the high flyer.

Port of the low-flying drone's task layer (see
reference_only/RayFronts_low/rayfronts/exploration_planner.py: _task_plan,
_score_tasks, _segment_unknown, peer claim/coverage handling), adapted:

  - Region entry and coverage use the high flyer's RayFronts geometric
    frontier viewpoints (the DBSCAN cluster centroids FrontierBehavior
    already computes) instead of the low drone's ground-like class
    frontiers. The task layer is a *filter/reorder* on those viewpoints;
    the behavior's own final selection logic is unchanged.
  - All robot/viewpoint coordinates are in the local/odom ENU frame (the
    frame FrontierBehavior works in); grid cells live in the shared
    frontier frame via geo_frame, exactly like the info grid, so peer
    coverage and claims align across robots.

Semantics identical to the low drone:
  - A cell is unexplored while absent from the 2D info grid; peer-covered
    cells are not unexplored; recently completed regions cool down.
  - One committed task at a time. Region tasks complete at
    region_coverage_done classified fraction or region_time_budget_s;
    track-revisit tasks complete on re-acquisition (covariance collapsed
    by a new observation), track removal, or the time budget.
  - MAIPP-greedy scoring: regions by expected undetected-target mass,
    tracks by existence/uncertainty/staleness; both pay a travel cost.
  - Peer region claims exclude cells from selection (never from coverage
    accounting); peer point claims suppress revisiting the claimed track.
"""

import logging
import math
import threading
import time

import numpy as np
import torch
import scipy.ndimage

from rayfronts.geo_frame import points_in_polygon

logger = logging.getLogger(__name__)


class MaippTaskPlanner:
  """Greedy region-explore / track-revisit task selection over viewpoints.

  Attributes:
    task: The committed task dict (type "region" | "track") or None.
  """

  def __init__(self, geo_frame, info_grid, tracker,
               robot_id=1,
               hover_height=8.0,
               track_view_elev_deg=45.0,
               keepout_polygons=None,
               region_min_area_m2=25.0,
               region_max_area_m2=400.0,
               region_min_viewpoints=1,
               region_coverage_done=0.80,
               region_time_budget_s=60.0,
               region_retire_cooldown_s=120.0,
               region_stall_timeout_s=45.0,
               region_stall_cooldown_s=180.0,
               target_birth_density=0.01,
               task_w_mass=1.0,
               task_w_prob=0.5,
               task_w_exist=1.0,
               task_w_cov=0.2,
               task_w_stale=0.005,
               task_w_travel=0.05,
               task_w_xbias=0.0,
               track_task_min_existence=0.3,
               track_task_min_sigma=0.8,
               track_task_min_unseen_s=240.0,
               track_task_time_budget_s=60.0,
               claim_ttl_s=10.0):
    """
    Args:
      geo_frame: FrontierFrame (shared with the info grid).
      info_grid: InfoGrid2D whose cells/bounds define the regions.
      tracker: PersonTracker whose people feed track-revisit tasks.
      robot_id: Our id (own tracks echoed back by relays are ignored).
      hover_height: z (local ENU, meters) for track pseudo-viewpoints.
      track_view_elev_deg: Desired depression angle to the person from the
        track viewpoint. The viewpoint stands off horizontally by
        height_above_person / tan(angle) on the robot's side (rotating
        around the person if that spot is out of bounds / in a keepout),
        instead of hovering directly overhead — which the fixed
        forward-pitched camera could not even see.
      keepout_polygons: List of (M, 2) frontier-frame polygons. Tracks
        inside one (or outside the mission bounds) are never selected as
        revisit tasks — the pseudo-viewpoint bypasses FrontierBehavior's
        keepout/bounds filtering, so the gate must happen here.
      region_min_area_m2 / region_max_area_m2: Region size band; connected
        components are median-split above max and dropped below min.
      region_min_viewpoints: Minimum in-region frontier viewpoints to
        switch from approach to in-region coverage.
      region_coverage_done: Classified cell fraction that completes a
        region task.
      region_time_budget_s / region_retire_cooldown_s: Give-up budget and
        the cooldown retired cells stay unselectable for.
      region_stall_timeout_s: Early-abort: a region task with neither
        in-region viewpoints nor coverage progress for this long finishes
        as stalled (a semantically unclassifiable but geometrically
        explored region would otherwise burn the full time budget).
      region_stall_cooldown_s: Longer cooldown for stalled regions, so
        likely-uncoverable cells are retried less eagerly than cleanly
        completed ones.
      target_birth_density: Undetected-target birth prior (targets/m^2)
        for region scoring.
      task_w_*: MAIPP-greedy score weights (see _score_tasks).
      task_w_xbias: Directional preference (score units per meter of
        frontier-frame x) added to every candidate's utility as
        w_xbias * x. Positive values bias task selection toward the
        +x side of the operating area; 0 disables.
      track_task_min_existence / min_sigma / min_unseen_s: A track is worth
        revisiting once fairly believed, grown uncertain, and unseen for a
        while.
      track_task_time_budget_s: Give-up budget for a revisit.
      claim_ttl_s: Peer claims older than this expire.
    """
    self.geo_frame = geo_frame
    self.info_grid = info_grid
    self.tracker = tracker
    self.robot_id = int(robot_id)
    self.hover_height = float(hover_height)
    self.track_view_elev_deg = float(track_view_elev_deg)
    self.keepout_polygons = keepout_polygons if keepout_polygons else []
    self.cell_size = float(info_grid.cell_size)
    self.bounds = info_grid.bounds

    self.region_min_area_m2 = float(region_min_area_m2)
    self.region_max_area_m2 = float(region_max_area_m2)
    self.region_min_viewpoints = int(region_min_viewpoints)
    self.region_coverage_done = float(region_coverage_done)
    self.region_time_budget_s = float(region_time_budget_s)
    self.region_retire_cooldown_s = float(region_retire_cooldown_s)
    self.region_stall_timeout_s = float(region_stall_timeout_s)
    self.region_stall_cooldown_s = float(region_stall_cooldown_s)
    self.target_birth_density = float(target_birth_density)
    self.task_w_mass = float(task_w_mass)
    self.task_w_prob = float(task_w_prob)
    self.task_w_exist = float(task_w_exist)
    self.task_w_cov = float(task_w_cov)
    self.task_w_stale = float(task_w_stale)
    self.task_w_travel = float(task_w_travel)
    self.task_w_xbias = float(task_w_xbias)
    self.track_task_min_existence = float(track_task_min_existence)
    self.track_task_min_sigma = float(track_task_min_sigma)
    self.track_task_min_unseen_s = float(track_task_min_unseen_s)
    self.track_task_time_budget_s = float(track_task_time_budget_s)
    self.claim_ttl_s = float(claim_ttl_s)

    self.task = None
    self._retired = {}     # window cell (i, j) -> retirement expiry walltime
    self._peer_cover = {}  # rid -> set of window cells (i, j)
    self._peer_claim = {}  # rid -> dict(cells=set|None, point=xy|None, t)
    self._peer_lock = threading.Lock()

  # ---------- peer message intake (ROS callback threads) ----------

  def on_peer_cover(self, rid, msg):
    """Parses a peer nav_msgs/OccupancyGrid into window coverage cells."""
    w, h = msg.info.width, msg.info.height
    if w == 0 or h == 0:
      return
    data = np.array(msg.data, dtype=np.int8).reshape(h, w)
    rows, cols = np.nonzero(data >= 0)
    res = float(msg.info.resolution)
    xs = float(msg.info.origin.position.x) + (cols + 0.5) * res
    ys = float(msg.info.origin.position.y) + (rows + 0.5) * res
    ix0, iz0, nx, nz = self._region_grid()
    g = self.cell_size
    i = np.round(xs / g).astype(np.int64) - ix0
    j = np.round(ys / g).astype(np.int64) - iz0
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz)
    cells = set(zip(i[ok].tolist(), j[ok].tolist()))
    with self._peer_lock:
      self._peer_cover[rid] = cells

  def on_peer_claim(self, rid, msg):
    """Parses a peer geometry_msgs/PolygonStamped task claim."""
    now = time.time()
    pts = [(float(p.x), float(p.y)) for p in msg.polygon.points]
    with self._peer_lock:
      if len(pts) == 0:
        self._peer_claim.pop(rid, None)
      elif len(pts) < 3:
        self._peer_claim[rid] = dict(cells=None, point=pts[0], t=now)
      else:
        ix0, iz0, nx, nz = self._region_grid()
        g = self.cell_size
        ii, jj = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        centers = np.stack([(ii.reshape(-1) + ix0) * g,
                            (jj.reshape(-1) + iz0) * g], axis=-1)
        inside = points_in_polygon(centers, np.array(pts))
        cells = set(zip((ii.reshape(-1)[inside]).tolist(),
                        (jj.reshape(-1)[inside]).tolist()))
        self._peer_claim[rid] = dict(cells=cells, point=None, t=now)

  # ---------- frame/grid helpers ----------

  def _local_to_frame_xy(self, pts_local):
    """Local-ENU points (Nx3 array-like) -> frontier-frame xy (Nx2 np)."""
    return np.asarray(
      self.geo_frame.local_to_frame(np.asarray(pts_local, dtype=np.float64))
    )[:, :2]

  def _person_frame_xy(self, person):
    """A track's world-RDF position -> frontier-frame xy."""
    pos = person["pos"]
    local = np.array([[float(pos[2]), -float(pos[0]), -float(pos[1])]])
    return self._local_to_frame_xy(local)[0]

  def _track_viewpoint_local(self, person, robot_pos_local):
    """Standoff viewpoint observing a track at track_view_elev_deg.

    The viewpoint is at hover_height with a horizontal standoff of
    height_above_person / tan(elev) from the person, on the robot's side
    (shortest approach, and the follower's path heading then points at
    the person). If that spot falls out of bounds or in a keepout zone,
    alternative azimuths around the person are tried (+-45deg, ...); the
    robot-side point is the fallback when no azimuth is admissible.
    """
    pos = person["pos"]
    ground = np.array([float(pos[2]), -float(pos[0]), -float(pos[1])])
    h_rel = self.hover_height - ground[2]
    if h_rel <= 0.5:
      h_rel = self.hover_height  # degenerate ground estimate
    d = h_rel / math.tan(math.radians(self.track_view_elev_deg))
    u = np.asarray(robot_pos_local, dtype=np.float64)[:2] - ground[:2]
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([1.0, 0.0])
    base = math.atan2(u[1], u[0])
    fallback = np.array([ground[0] + d * u[0], ground[1] + d * u[1],
                         self.hover_height])
    for da in (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0):
      a = base + math.radians(da)
      vp = np.array([ground[0] + d * math.cos(a),
                     ground[1] + d * math.sin(a),
                     self.hover_height])
      fxy = self._local_to_frame_xy(vp.reshape(1, 3))[0]
      if self._track_in_operating_area(fxy):
        return vp
    return fallback

  def _region_grid(self):
    """Window (ix0, iz0, nx, nz) of cells whose centers lie in bounds."""
    mnx, mxx, mny, mxy = self.bounds
    g = self.cell_size
    ix0 = int(math.ceil(mnx / g))
    iz0 = int(math.ceil(mny / g))
    nx = max(1, int(math.floor(mxx / g)) - ix0 + 1)
    nz = max(1, int(math.floor(mxy / g)) - iz0 + 1)
    return ix0, iz0, nx, nz

  def _unknown_mask(self):
    """Boolean [nx, nz] mask of semantically unexplored window cells.

    A cell is unknown while absent from the 2D info grid. Retired
    (cooldown) cells and peer-covered cells are not unexplored (the
    latter also counts toward our committed region's coverage, which is
    correct).
    """
    ix0, iz0, nx, nz = self._region_grid()
    unknown = np.ones((nx, nz), dtype=bool)
    for (ix, iz) in self.info_grid.info_grid.keys():
      i, j = ix - ix0, iz - iz0
      if 0 <= i < nx and 0 <= j < nz:
        unknown[i, j] = False
    now = time.time()
    self._retired = {c: t for c, t in self._retired.items() if t > now}
    for (i, j) in self._retired:
      if 0 <= i < nx and 0 <= j < nz:
        unknown[i, j] = False
    with self._peer_lock:
      covers = [set(c) for c in self._peer_cover.values()]
    for cells in covers:
      for (i, j) in cells:
        if 0 <= i < nx and 0 <= j < nz:
          unknown[i, j] = False
    return unknown

  def _segment_unknown(self, unknown):
    """8-connected components, median-split above region_max_area_m2."""
    labeled, n = scipy.ndimage.label(
      unknown, structure=np.ones((3, 3), dtype=int))
    cs2 = self.cell_size ** 2
    min_cells = max(1, int(round(self.region_min_area_m2 / cs2)))
    max_cells = max(min_cells * 2, int(round(self.region_max_area_m2 / cs2)))
    regions = []
    for k in range(1, n + 1):
      idx = np.argwhere(labeled == k)  # [K, 2] (i, j)
      if idx.shape[0] < min_cells:
        continue
      regions.extend(self._split_cells(idx, max_cells))
    return [set(map(tuple, r.tolist())) for r in regions]

  def _split_cells(self, idx, max_cells):
    if idx.shape[0] <= max_cells:
      return [idx]
    axis = int(np.argmax(idx.astype(np.float32).var(axis=0)))
    order = np.argsort(idx[:, axis], kind="stable")
    mid = idx.shape[0] // 2
    return (self._split_cells(idx[order[:mid]], max_cells) +
            self._split_cells(idx[order[mid:]], max_cells))

  def _region_centroid(self, cells):
    """Frontier-frame (x, y) centroid of a region cell set."""
    ix0, iz0, _, _ = self._region_grid()
    g = self.cell_size
    arr = np.array(sorted(cells), dtype=np.float32)
    return ((float(arr[:, 0].mean()) + ix0) * g,
            (float(arr[:, 1].mean()) + iz0) * g)

  def _claimed_cells(self):
    """Union of cells in fresh peer region claims (expired ones drop)."""
    now = time.time()
    cells = set()
    with self._peer_lock:
      for rid in list(self._peer_claim.keys()):
        c = self._peer_claim[rid]
        if now - c["t"] > self.claim_ttl_s:
          self._peer_claim.pop(rid)
        elif c.get("cells"):
          cells |= c["cells"]
    return cells

  def _track_in_operating_area(self, pxy):
    """True if a frontier-frame xy lies inside bounds and outside keepouts.

    Tracks failing this must never become revisit tasks: their
    pseudo-viewpoint is injected after FrontierBehavior's keepout/bounds
    filtering and would send the drone into a keepout zone or out of the
    mission boundary.
    """
    mnx, mxx, mny, mxy = self.bounds
    if not (mnx <= pxy[0] <= mxx and mny <= pxy[1] <= mxy):
      return False
    pt = np.asarray(pxy, dtype=np.float64).reshape(1, 2)
    for poly in self.keepout_polygons:
      if points_in_polygon(pt, poly)[0]:
        return False
    return True

  def _track_claimed(self, person):
    """True if a fresh peer point-claim targets this track."""
    now = time.time()
    pxy = self._person_frame_xy(person)
    with self._peer_lock:
      claims = list(self._peer_claim.values())
    for c in claims:
      if now - c["t"] > self.claim_ttl_s or c.get("point") is None:
        continue
      if (math.hypot(pxy[0] - c["point"][0], pxy[1] - c["point"][1])
          <= self.tracker.person_merge_radius):
        return True
    return False

  # ---------- task lifecycle ----------

  def _finish_task(self, reason, cooldown_s=None):
    """Completes the committed task; region cells go on cooldown."""
    t = self.task
    if t["type"] == "region":
      cooldown = (self.region_retire_cooldown_s if cooldown_s is None
                  else float(cooldown_s))
      expiry = time.time() + cooldown
      for c in t["cells"]:
        self._retired[c] = expiry
      logger.info(
        "Region task at (%.1f, %.1f) done: %s (%d cells, %.0fs cooldown).",
        *t["centroid"], reason, len(t["cells"]), cooldown)
    else:
      logger.info("Track task (person #%d) done: %s.",
                  t["track_id"], reason)
    self.task = None

  def _score_tasks(self, regions, robot_xy):
    """MAIPP-greedy utilities over region + track candidate tasks.

    Region (explore): w_mass * lambda_sum + w_prob * (1 - exp(-lambda_sum))
      - w_travel * dist, with lambda_sum = target_birth_density *
      unexplored area (uniform undetected-target birth prior).
    Track (revisit): w_exist * existence + w_cov * trace(P) + w_stale *
      time_since_seen - w_travel * dist, for tracks worth revisiting.
    Returns (best task dict or None, n_region_cands, n_track_cands).
    """
    now = time.time()
    cands = []
    cell_area = self.cell_size ** 2
    for cells in regions:
      cx, cy = self._region_centroid(cells)
      dist = math.hypot(cx - robot_xy[0], cy - robot_xy[1])
      lam = self.target_birth_density * len(cells) * cell_area
      score = (self.task_w_mass * lam
               + self.task_w_prob * (1.0 - math.exp(-lam))
               - self.task_w_travel * dist
               + self.task_w_xbias * cx)
      cands.append((score, dict(type="region", cells=cells,
                                centroid=(cx, cy), t_start=now,
                                best_cov=0.0, last_progress_t=now)))
    n_region = len(cands)
    for p in self.tracker.people:
      sigma = math.sqrt(0.5 * float(torch.trace(p["cov"])))
      if (p["existence"] < self.track_task_min_existence or
          sigma < self.track_task_min_sigma or
          now - p["last_seen"] < self.track_task_min_unseen_s or
          self._track_claimed(p)):
        continue
      pxy = self._person_frame_xy(p)
      if not self._track_in_operating_area(pxy):
        continue
      dist = math.hypot(pxy[0] - robot_xy[0], pxy[1] - robot_xy[1])
      score = (self.task_w_exist * p["existence"]
               + self.task_w_cov * float(torch.trace(p["cov"]))
               + self.task_w_stale * (now - p["last_seen"])
               - self.task_w_travel * dist
               + self.task_w_xbias * pxy[0])
      cands.append((score, dict(type="track", track_id=p["track_id"],
                                t_start=now, start_n_obs=p["n_obs"])))
    if len(cands) == 0:
      return None, 0, 0
    best = max(cands, key=lambda c: c[0])
    return best[1], n_region, len(cands) - n_region

  def _make_occ_query(self, mapper):
    """Returns occ_at(p_local)->bool over the mapper's occupancy, or None."""
    if mapper is None:
      return None
    try:
      import rayfronts_cpp
    except ImportError:
      return None
    def occ_at(p_local):
      # local ENU -> world RDF: (x_r, y_r, z_r) = (-y_l, -z_l, x_l).
      rdf = torch.tensor(
        [[-p_local[1], -p_local[2], p_local[0]]], dtype=torch.float)
      occ = rayfronts_cpp.query_occ(mapper.occ_map_vdb, rdf)
      return float(occ.reshape(-1)[0]) > 0
    return occ_at

  def _safe_hover_point(self, p_local, robot_pos_local, occ_at):
    """Nudges a hover point out of known-occupied space.

    Unknown space is treated as free (same trust-the-follower assumption
    as every other waypoint in this pipeline); only known-occupied voxels
    trigger a nudge. Tries rising in 1m steps first (canopy/structure
    under the point), then standing off horizontally toward the robot.
    Falls back to the original point if nothing known-free is found.
    """
    if not occ_at(p_local):
      return p_local
    for dz in range(1, 7):
      q = p_local + np.array([0.0, 0.0, float(dz)])
      if not occ_at(q):
        logger.info("Track hover point occupied; raised %.0fm.", float(dz))
        return q
    d = np.asarray(robot_pos_local, dtype=np.float64)[:2] - p_local[:2]
    n = float(np.linalg.norm(d))
    if n > 1e-6:
      d = d / n
      for r in (2.0, 4.0, 6.0):
        q = p_local.copy()
        q[:2] += d * r
        if not occ_at(q):
          logger.info(
            "Track hover point occupied; stood off %.0fm toward robot.", r)
          return q
    logger.warning("Track hover point occupied with no known-free nudge; "
                   "keeping it (local avoidance must handle it).")
    return p_local

  def select(self, viewpoints_local, robot_pos_local, mapper=None):
    """Task layer over frontier viewpoint selection.

    Args:
      viewpoints_local: Nx3 float tensor of candidate frontier viewpoints
        in the local/odom ENU frame (FrontierBehavior's frame). May be
        empty.
      robot_pos_local: size-3 array-like robot position, same frame.
      mapper: Optional mapper whose occupancy map sanity-checks the track
        pseudo-viewpoint (known-occupied hover points get nudged).

    Returns:
      Mx3 float tensor of candidate viewpoints the behavior should choose
      among (a subset, a centroid-ordered single approach viewpoint, a
      single track pseudo-viewpoint, or the input unchanged when no task
      logic applies).
    """
    if self.geo_frame is None or not self.geo_frame.is_ready():
      return viewpoints_local

    unknown = self._unknown_mask()
    unknown_set = set(map(tuple, np.argwhere(unknown).tolist()))

    # Completion checks on the committed task.
    if self.task is not None and self.task["type"] == "region":
      cells = self.task["cells"]
      cov = 1.0 - len(cells & unknown_set) / max(len(cells), 1)
      now = time.time()
      age = now - self.task["t_start"]
      # Progress = the classified fraction growing (in-region viewpoints
      # also count as progress; see the region branch below).
      if cov > self.task.get("best_cov", 0.0) + 1e-9:
        self.task["best_cov"] = cov
        self.task["last_progress_t"] = now
      stall = now - self.task.get("last_progress_t", self.task["t_start"])
      if cov >= self.region_coverage_done:
        self._finish_task("coverage %.0f%% reached" % (cov * 100))
      elif age > self.region_time_budget_s:
        self._finish_task("time budget exceeded (%.0fs, coverage %.0f%%)"
                          % (age, cov * 100))
      elif stall > self.region_stall_timeout_s:
        # No frontiers ever appeared inside and re-observation is not
        # classifying its cells (e.g. semantically unclassifiable but
        # geometrically explored ground) — give up early with a longer
        # cooldown instead of burning the full time budget.
        self._finish_task(
          "stalled: no in-region viewpoints or coverage progress for "
          "%.0fs (coverage %.0f%%)" % (stall, cov * 100),
          cooldown_s=self.region_stall_cooldown_s)
    elif self.task is not None:  # track task
      track = next((p for p in self.tracker.people
                    if p["track_id"] == self.task["track_id"]), None)
      age = time.time() - self.task["t_start"]
      if track is None:
        self._finish_task("track disappeared")
      elif (track["n_obs"] > self.task["start_n_obs"] and
            math.sqrt(0.5 * float(torch.trace(track["cov"])))
            < self.track_task_min_sigma):
        self._finish_task(
          "re-acquired (existence %.2f)" % track["existence"])
      elif age > self.track_task_time_budget_s:
        self._finish_task("time budget exceeded (%.0fs)" % age)

    # Cells retired this cycle and cells inside fresh peer region claims
    # must not be selectable (claims are selection-only: they do not count
    # as covered for our committed task).
    excl = set(self._retired.keys()) | self._claimed_cells()
    if excl:
      nx, nz = unknown.shape
      for (i, j) in excl:
        if 0 <= i < nx and 0 <= j < nz:
          unknown[i, j] = False
      unknown_set -= excl

    robot_xy = self._local_to_frame_xy(
      np.asarray(robot_pos_local, dtype=np.float64).reshape(1, 3))[0]

    # Commit to the best-scoring task if none is active.
    if self.task is None:
      regions = self._segment_unknown(unknown)
      task, n_r, n_t = self._score_tasks(
        regions, (float(robot_xy[0]), float(robot_xy[1])))
      if task is None:
        return viewpoints_local
      self.task = task
      if task["type"] == "region":
        logger.info(
          "Task selected: explore region at (%.1f, %.1f), %.1f m^2 "
          "(%d region / %d track candidates).", *task["centroid"],
          len(task["cells"]) * self.cell_size ** 2, n_r, n_t)
      else:
        logger.info(
          "Task selected: revisit person track #%d "
          "(%d region / %d track candidates).", task["track_id"], n_r, n_t)

    if self.task["type"] == "track":
      track = next((p for p in self.tracker.people
                    if p["track_id"] == self.task["track_id"]), None)
      if track is None:
        return viewpoints_local
      # Standoff viewpoint observing the person at the configured
      # depression angle, nudged out of known-occupied space when the occ
      # map is available.
      p_local = self._track_viewpoint_local(track, robot_pos_local)
      occ_at = self._make_occ_query(mapper)
      if occ_at is not None:
        p_local = self._safe_hover_point(
          p_local, np.asarray(robot_pos_local, dtype=np.float64), occ_at)
      return torch.tensor(p_local, dtype=torch.float).reshape(1, 3)

    # Region task: viewpoints inside the region's cells if enough exist,
    # else approach via the single viewpoint closest to the centroid.
    if viewpoints_local.shape[0] == 0:
      return viewpoints_local
    ix0, iz0, _, _ = self._region_grid()
    g = self.cell_size
    cells = self.task["cells"]
    vxy = self._local_to_frame_xy(viewpoints_local.detach().cpu().numpy())
    i = np.round(vxy[:, 0] / g).astype(np.int64) - ix0
    j = np.round(vxy[:, 1] / g).astype(np.int64) - iz0
    in_mask = torch.tensor(
      [(int(a), int(b)) in cells for a, b in zip(i.tolist(), j.tolist())],
      dtype=torch.bool)
    if int(in_mask.sum()) >= self.region_min_viewpoints:
      # Having viewpoints inside the region counts as progress: the region
      # is actively workable, so the stall early-abort must not fire.
      self.task["last_progress_t"] = time.time()
      return viewpoints_local[in_mask]
    cx, cy = self.task["centroid"]
    d_cent = np.hypot(vxy[:, 0] - cx, vxy[:, 1] - cy)
    return viewpoints_local[int(np.argmin(d_cent))].reshape(1, 3)

  # ---------- outgoing claim ----------

  def build_claim_msg(self, msg, stamp):
    """Fills a geometry_msgs/PolygonStamped with the committed task claim.

    Region task -> bounding box of the region cells; track task -> single
    point at the claimed track; idle -> empty polygon. Same wire format
    the low drone publishes and fuses.
    """
    from geometry_msgs.msg import Point32
    msg.header.stamp = stamp
    msg.header.frame_id = "frontier_frame"
    t = self.task
    if t is not None and t["type"] == "region":
      ix0, iz0, _, _ = self._region_grid()
      g = self.cell_size
      arr = np.array(sorted(t["cells"]), dtype=np.float64)
      x_lo = (arr[:, 0].min() + ix0) * g - 0.5 * g
      x_hi = (arr[:, 0].max() + ix0) * g + 0.5 * g
      y_lo = (arr[:, 1].min() + iz0) * g - 0.5 * g
      y_hi = (arr[:, 1].max() + iz0) * g + 0.5 * g
      for x, y in ((x_lo, y_lo), (x_hi, y_lo), (x_hi, y_hi), (x_lo, y_hi)):
        msg.polygon.points.append(Point32(x=float(x), y=float(y), z=0.0))
    elif t is not None:
      track = next((p for p in self.tracker.people
                    if p["track_id"] == t["track_id"]), None)
      if track is not None:
        pxy = self._person_frame_xy(track)
        msg.polygon.points.append(
          Point32(x=float(pxy[0]), y=float(pxy[1]), z=0.0))
    return msg

"""
Walking distances along real streets and paths (roadmap item 2)
==============================================================

Builds a pedestrian graph from the frozen OpenStreetMap extract data/raw/osm_walk.json (streets,
sidewalks, footpaths, trails; motorways excluded) and measures shortest walking distances with
Dijkstra's algorithm (scipy). Used three ways:

  1. cell-to-cell walking distances for the optimizer's job and services objectives
  2. walk time from every cell to BART, groceries, parks and schools
  3. 5 / 10 / 15 minute walk-time areas around the BART station (service areas)

Assumptions (also listed in the planner summary)
  * Walking speed 3 mph = 264 ft per minute, so a 10-minute walk is half a mile, the usual
    transit-station walkshed in planning practice.
  * Inside the redevelopment area the future street grid doesn't exist yet, so neighbouring site cells
    are linked directly (a fine-grained walkable block pattern, as the Urban Village Plan calls for).
    Everywhere else people walk on today's network.
  * Each cell joins the network at its nearest node whose connecting line doesn't cross a creek or
    the rail line.
  * Dangling path ends within 25 ft of another node are joined (the same tolerance gis_qaqc.py uses
    for road undershoots).
"""
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

FT_PER_MIN = 264.0                      # 3 mph
SNAP_FT = 25.0
EXCLUDE_HW = {"motorway", "motorway_link", "construction", "proposed", "raceway", "bus_guideway"}
BANDS_MIN = (5, 10, 15)


class WalkGraph:
    """Nodes (x, y in feet) and undirected weighted edges. Cell nodes are appended after the OSM nodes."""

    def __init__(self, osm, crs="EPSG:2227"):
        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        index, xs, ys, edges, lines = {}, [], [], [], []
        for el in osm["elements"]:
            t = el.get("tags", {})
            if el["type"] != "way" or "highway" not in t or t["highway"] in EXCLUDE_HW:
                continue
            if t.get("foot") == "no" or t.get("access") == "no":
                continue
            ids, geom = el.get("nodes", []), el.get("geometry", [])
            if len(ids) < 2 or len(ids) != len(geom):
                continue
            px, py = tf.transform([p["lon"] for p in geom], [p["lat"] for p in geom])
            seq = []
            for nid, x, y in zip(ids, px, py):
                if nid not in index:
                    index[nid] = len(xs)
                    xs.append(x)
                    ys.append(y)
                seq.append(index[nid])
            for a, b in zip(seq[:-1], seq[1:]):
                if a != b:
                    edges.append((a, b))
            lines.append(LineString(list(zip(px, py))))
        self.xy = np.column_stack([xs, ys])
        self.n_osm = len(xs)
        self.edges = edges
        self.lines = lines
        self.extra = []                                    # (a, b, length) edges added later
        self._snap_dangles()

    def _snap_dangles(self):
        deg = np.zeros(self.n_osm, dtype=int)
        nbrs = [set() for _ in range(self.n_osm)]
        for a, b in self.edges:
            deg[a] += 1
            deg[b] += 1
            nbrs[a].add(b)
            nbrs[b].add(a)
        tree = cKDTree(self.xy)
        self.snapped = 0
        for i in np.flatnonzero(deg == 1):
            for j in tree.query_ball_point(self.xy[i], SNAP_FT):
                if j != i and j not in nbrs[i]:
                    d = float(np.hypot(*(self.xy[i] - self.xy[j])))
                    self.extra.append((i, j, d))
                    self.snapped += 1
                    break

    def add_nodes(self, pts):
        start = len(self.xy)
        self.xy = np.vstack([self.xy, np.asarray(pts, dtype=float)])
        return np.arange(start, len(self.xy))

    def connect(self, node_ids, barrier=None, max_ft=800.0, k=6):
        """Join each new node to its nearest OSM node whose connector doesn't cross the barrier."""
        tree = cKDTree(self.xy[:self.n_osm])
        b_tree = STRtree(list(getattr(barrier, "geoms", [barrier]))) if barrier is not None and not barrier.is_empty else None
        b_geoms = list(getattr(barrier, "geoms", [barrier])) if b_tree is not None else []
        joined = 0
        for nid in node_ids:
            p = self.xy[nid]
            dist, idx = tree.query(p, k=k, distance_upper_bound=max_ft)
            for d, j in zip(np.atleast_1d(dist), np.atleast_1d(idx)):
                if not np.isfinite(d):
                    break
                if b_tree is not None:
                    seg = LineString([tuple(p), tuple(self.xy[j])])
                    if any(seg.intersects(b_geoms[h]) for h in b_tree.query(seg)):
                        continue
                self.extra.append((int(nid), int(j), float(d)))
                joined += 1
                break
        return joined

    def link(self, pairs, length):
        for a, b in pairs:
            self.extra.append((int(a), int(b), float(length)))

    def matrix(self):
        a = np.array([e[0] for e in self.edges] + [e[0] for e in self.extra], dtype=int)
        b = np.array([e[1] for e in self.edges] + [e[1] for e in self.extra], dtype=int)
        w = np.concatenate([np.hypot(*(self.xy[[e[0] for e in self.edges]] - self.xy[[e[1] for e in self.edges]]).T)
                            if self.edges else np.zeros(0),
                            np.array([e[2] for e in self.extra])])
        w = np.maximum(w, 0.01)                          # zero-length edges would vanish in a sparse matrix
        n = len(self.xy)
        m = coo_matrix((np.concatenate([w, w]), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(n, n))
        return m.tocsr()

    def nearest_osm_nodes(self, pts, max_ft=600.0):
        tree = cKDTree(self.xy[:self.n_osm])
        d, idx = tree.query(np.asarray(pts, dtype=float), distance_upper_bound=max_ft)
        return [int(i) for i, dd in zip(np.atleast_1d(idx), np.atleast_1d(d)) if np.isfinite(dd)]


def nodes_near_polygons(graph, polys, tol_ft=60.0):
    """Network nodes that serve as entrances to destination polygons (on or within tol of the edge)."""
    if polys is None or len(polys) == 0:
        return []
    tree = cKDTree(graph.xy[:graph.n_osm])
    out = set()
    for g in polys:
        g = g.buffer(tol_ft)
        minx, miny, maxx, maxy = g.bounds
        cand = tree.query_ball_point(((minx + maxx) / 2, (miny + maxy) / 2), np.hypot(maxx - minx, maxy - miny) / 2)
        hit = [j for j in cand if g.contains(Point(graph.xy[j]))]
        if not hit:                                  # polygon far from any path: use its closest node
            c = g.representative_point()
            hit = [int(tree.query((c.x, c.y))[1])]
        out.update(hit)
    return sorted(out)


def service_areas(graph, csr, origin_nodes, crs, bands=BANDS_MIN, trim_ft=100.0):
    """Walk-time areas: every street segment reachable within the band, buffered by trim_ft
    (the same idea as QGIS 'Service area' or ArcGIS Network Analyst's trim distance)."""
    dist = dijkstra(csr, indices=origin_nodes, min_only=True)
    rows, prev = [], None
    osm_edges = np.array(graph.edges)
    xy = graph.xy
    for minutes in bands:
        lim = minutes * FT_PER_MIN
        segs = []
        da, db = dist[osm_edges[:, 0]], dist[osm_edges[:, 1]]
        for (a, b), ra, rb in zip(osm_edges, da, db):
            if ra > lim and rb > lim:
                continue
            pa, pb = xy[a], xy[b]
            L = float(np.hypot(*(pb - pa)))
            if ra <= lim and rb <= lim:
                segs.append(LineString([pa, pb]))
            elif ra <= lim:                               # partially reachable from a
                f = min(1.0, (lim - ra) / L) if L else 0
                segs.append(LineString([pa, pa + f * (pb - pa)]))
            else:
                f = min(1.0, (lim - rb) / L) if L else 0
                segs.append(LineString([pb, pb + f * (pa - pb)]))
        area = unary_union(segs).buffer(trim_ft) if segs else None
        if area is None:
            continue
        ring = area.difference(prev) if prev is not None else area
        rows.append({"minutes": minutes, "label": f"{minutes}-minute walk", "max_ft": round(lim),
                     "acres": round(ring.area / 43560, 1), "geometry": ring})
        prev = area
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs), dist

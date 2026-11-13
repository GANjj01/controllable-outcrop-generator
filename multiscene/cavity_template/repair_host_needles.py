"""Representation-level repair of host needles failing the source-normal gate, on a DERIVED difference only.

Usage: python repair_host_needles.py CASE NEW_CASE

A floating Boolean leaves DERIVED intersection vertices that should coincide 1e-10..1e-8 m apart; the needles between
them have heights of 1e4-1e5 coordinate ulps, so re-choosing doubles cannot fix a flipped one. Candidate sets are
tried smallest first: one cluster holding a face vertex, all clusters touching its one-ring, all clusters of its
two-ring (e.g. a chain of host and floor needles along a host edge that nearly follows a cavity corner). Its cluster
sets are tier 2's; only sets whose member weld is index-manifold qualify (a duplicate face or a repeated directed edge
does not depend on where the cluster goes). Each cluster's target lies on the host: on the plane of its one host
source face, on the line shared by its two, none for three or more (fail closed; a cluster never holds a source
vertex); a cluster without host faces keeps tier 2's member targets. Candidates are the members' projections within
MAX_EDGE_M of every member; if no combination passes, a small LP in that plane or line maximises the smallest
orientation margin of the incident host and cavity faces, the target staying within MAX_EDGE_M of every member and
inside its host source triangles (two alternating rounds when a set holds several clusters). The target is a new
coordinate; every gate and tolerance is unchanged. The report gains the key tier4, and repairs of kind projected_weld,
only when the phase runs. Every face incident to a moved or merged vertex must keep positive area; host faces must
pass the unchanged gate, plane and barycentric tolerances; cavity faces keep their normal direction. Source vertices
never move; surviving faces keep winding, original IDs and source face IDs. Afterwards the whole host is re-gated,
per-source retained area must not increase, and the result must stay closed index-manifold with consistent winding.
Unresolved faces exit 1 without output. The original CASE is not modified;"""
import argparse, itertools, json, os, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from source_tables import FORMAT as SOURCE_TABLE_FORMAT, load_sources, paths as source_table_paths  # noqa: E402
GATE = 1 - 1e-10
PLANE_M = 1e-8
BARY = 1e-8
AREA_GAIN_M2 = 1e-10
MAX_ULP = 4
MAX_EDGE_M = 1e-7  # network_cutters source_position_tolerance_m; not widened
XYZ = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64)])
PW_RULE = 'pw-r1'
PW_POLY = 8


def lp_max_margin(G, h, A, b, tmax=1e-6, chunk=40000):
    """Returns (x, t), or None when no vertex is feasible."""
    G, A = np.atleast_2d(G), np.atleast_2d(A)
    k = G.shape[1]
    M = np.vstack([np.c_[-G, np.ones(len(G))], np.c_[A.reshape(-1, k), np.zeros(len(A))], np.r_[np.zeros(k), 1.][None]])
    r = np.r_[h, b, tmax]
    tol = 1e-12 * (np.abs(r) + np.abs(r).max()) + 1e-30   # relative to the problem's own scale (metres here, ~1e-9)
    best = None
    combos = np.array(list(itertools.combinations(range(len(M)), k + 1)), dtype=np.int64)
    for s in range(0, len(combos), chunk):
        idx = combos[s:s + chunk]
        S, R = M[idx], r[idx]
        ok = np.abs(np.linalg.det(S)) > 1e-12
        if not ok.any():
            continue
        z = np.linalg.solve(S[ok], R[ok][..., None])[..., 0]
        feas = np.all(M @ z.T <= (r + tol)[:, None], axis=0)
        if feas.any():
            j = int(np.argmax(np.where(feas, z[:, -1], -np.inf)))
            if best is None or z[j, -1] > best[1]:
                best = (z[j, :-1].copy(), float(z[j, -1]))
    return best


def frame_of(planes, centre):
    """Returns (origin, basis 3 x k) or (None, reason)."""
    if len(planes) == 1 or len(planes) == 2 and np.linalg.norm(np.cross(planes[0][0], planes[1][0])) < 1e-6:
        n, o = planes[0]
        origin = centre - ((centre - o) @ n) * n
        if len(planes) == 2 and abs((origin - planes[1][1]) @ planes[1][0]) > PLANE_M / 10:
            return None, 'two nearly parallel host source planes that do not coincide at the cluster'
        a = np.cross(n, [1., 0., 0.] if abs(n[0]) < .9 else [0., 1., 0.])
        a /= np.linalg.norm(a)
        return origin, np.c_[a, np.cross(n, a)]
    if len(planes) == 2:
        (n0, o0), (n1, o1) = planes
        d = np.cross(n0, n1)
        d /= np.linalg.norm(d)
        origin = np.linalg.solve(np.array([n0, n1, d]), np.array([n0 @ o0, n1 @ o1, d @ centre]))
        return origin, d[:, None]
    return None, 'touches %d host source faces (no target: a cluster holds no source vertex)' % len(planes)


def closed_manifold(faces):
    """Every undirected edge used exactly twice, in opposite directions (faces >= 0): no self-edge, no repeated directed
    edge, and the directed edges equal their reverses as multisets."""
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        return True
    n = int(faces.max()) + 1
    fwd, rev = np.empty(3 * len(faces), np.int64), np.empty(3 * len(faces), np.int64)
    for k in range(3):
        a, b = faces[:, k], faces[:, (k + 1) % 3]
        if np.any(a == b):
            return False
        fwd[k::3], rev[k::3] = a * n + b, b * n + a
    fwd.sort(); rev.sort()
    return bool(np.all(fwd[1:] != fwd[:-1]) and np.array_equal(fwd, rev))


class Incident:

    def __init__(self, faces):
        flat = faces.ravel()
        order = np.argsort(flat, kind='stable')
        order //= 3
        self.face = order.astype(np.int32) if len(faces) < 2 ** 31 else order
        del order
        self.start = np.concatenate(([0], np.cumsum(np.bincount(flat))))
        self.sets, self.gone = {}, set()

    def _span(self, x):
        return (int(self.start[x]), int(self.start[x + 1])) if 0 <= x < len(self.start) - 1 else (0, 0)

    def __getitem__(self, x):
        s = self.sets.get(x)
        if s is None:
            a, b = self._span(x)
            if x in self.gone or a == b:
                raise KeyError(x)
            s = self.sets[x] = set(self.face[a:b].tolist())
        return s

    def __contains__(self, x):
        if x in self.sets:
            return True
        a, b = self._span(x)
        return a < b and x not in self.gone

    def __setitem__(self, x, faces):
        self.sets[x] = faces
        self.gone.discard(x)

    def pop(self, x, *default):
        try:
            s = self[x]
        except KeyError:
            if default:
                return default[0]
            raise
        del self.sets[x]
        self.gone.add(x)
        return s


def point_set(points):
    """sorted float64 rows, as membership in a set of their tuples has it: -0.0 equals 0.0, a row with NaN equals nothing"""
    points = np.asarray(points, dtype=np.float64)
    return np.unique(np.ascontiguousarray(points[~np.isnan(points).any(axis=1)]).view(XYZ).ravel())


def in_point_set(rows, point):
    q = np.ascontiguousarray(point, dtype=np.float64).reshape(1, 3).view(XYZ).ravel()
    i = int(np.searchsorted(rows, q)[0])
    return i < len(rows) and bool(rows[i] == q[0])


def role_table(case, index):
    """The role column of path INDEX's source table as (sorted face ids, codes, role values), for the lookup
    rec['face_sources'].get(str(sid), {}).get('role') of the decoded table (a repeated face id keeps its last row), read
    without the per-face dicts. A table of another form is returned as its face_sources and looked up with that expression."""
    js, nz = source_table_paths(case, index)
    if nz.is_file() and not js.is_file():
        with np.load(nz, allow_pickle=False) as a:  # what source_tables.decode reads, without building the dicts
            if str(a['format']) != SOURCE_TABLE_FORMAT:
                raise ValueError('%s: unknown source table format %s' % (nz, a['format']))
            json.loads(str(a['meta_json']))
            face, code, values = a['face_id'], a['role_code'], a['role_names'].tolist()
        return last_rows(face, code, values)
    table = load_sources(case, index)['face_sources']
    if type(table) is not dict:
        return table
    face, code, values, number = [], [], [], {}
    for key, row in table.items():
        if type(row) is not dict:
            return table
        try:
            c = number.setdefault((type(row.get('role')), row.get('role')), len(number))
        except TypeError:  # an unhashable role
            return table
        if c == len(values):
            values.append(row.get('role'))
        try:
            k = int(key)
        except ValueError:
            continue
        if str(k) == key and -2 ** 63 <= k < 2 ** 63:  # any other key is never str(int(sid))
            face.append(k); code.append(c)
    return last_rows(np.array(face, np.int64), np.array(code, np.int32), values)


def last_rows(face, code, values):
    order = np.argsort(face, kind='stable')
    face, code = face[order], code[order]
    last = np.ones(len(face), bool)
    last[:-1] = face[1:] != face[:-1]
    return face[last], code[last], values


def removed_role(table, sid, is_host):
    """the role reported for a removed face, as roles.get(oid, {}).get(str(sid), {}) and then 'original_host' or
    row.get('role') gave it"""
    if type(table) is tuple:
        if is_host:
            return 'original_host'
        face, code, values = table
        i = int(np.searchsorted(face, sid))
        return values[code[i]] if i < len(face) and face[i] == sid else None
    row = table.get(str(sid), {})
    return 'original_host' if is_host else row.get('role')


def repair(case, new, link_source_tables=True):
    build = json.loads((case / 'build.json').read_text(encoding='utf8'))
    source = np.load(Path(build['source']) / 'source_surface.npz')
    d = np.load(case / 'difference.npz')
    v0 = d['vertices']
    v = v0.copy()
    faces0 = d['faces']  # read once: the area of a removed face uses its row as loaded
    faces_dtype = faces0.dtype
    f = faces0.astype(np.int64)
    oid, sid = d['original_id'], d['source_face_id']
    host = oid == build['host_original_id']
    alive = np.ones(len(f), bool)
    st = source['vertices'][source['faces']]
    sn = np.cross(st[:, 1] - st[:, 0], st[:, 2] - st[:, 0])
    snl = np.linalg.norm(sn, axis=1)
    source_vertex = point_set(source['vertices'])
    derived = lambda x: not in_point_set(source_vertex, v0[x])
    roles = {}
    for path in build['paths']:
        roles[int(path['original_id'])] = role_table(case, path['index'])
    incident = Incident(f)

    def cross(faces):
        tri = v[f[faces]]
        return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    def host_gate(faces):
        faces = np.asarray(faces, dtype=np.int64)
        if not len(faces):
            return True, 1.
        c = cross(faces)
        a = np.linalg.norm(c, axis=1)
        if np.any(a <= 0):
            return False, -1.
        s = sid[faces]
        cos = np.einsum('ij,ij->i', c, sn[s]) / (a * snl[s])
        tri = v[f[faces]]
        rel = tri - st[s, 0][:, None]
        plane = np.abs(np.einsum('fvi,fi->fv', rel, sn[s]) / snl[s][:, None]).max()
        e0, e1 = st[s, 1] - st[s, 0], st[s, 2] - st[s, 0]
        d00, d01, d11 = (np.einsum('fi,fi->f', p, q) for p, q in ((e0, e0), (e0, e1), (e1, e1)))
        d20, d21 = np.einsum('fvi,fi->fv', rel, e0), np.einsum('fvi,fi->fv', rel, e1)
        den = (d00 * d11 - d01 * d01)[:, None]
        u = (d11[:, None] * d20 - d01[:, None] * d21) / den
        w = (d00[:, None] * d21 - d01[:, None] * d20) / den
        bary = np.stack((1 - u - w, u, w), axis=2)
        ok = cos.min() >= GATE and plane <= PLANE_M and bary.min() >= -BARY and bary.max() <= 1 + BARY
        return bool(ok), float(cos.min())

    def cavity_gate(faces, before):
        if not len(faces):
            return True
        c = cross(faces)
        return bool(np.all(np.linalg.norm(c, axis=1) > 0) and np.all(np.einsum('ij,ij->i', c, before) > 0))

    def fan_ok(vertex, before_cavity):
        faces = sorted(incident[vertex])
        hf = [j for j in faces if host[j]]
        cf = [j for j in faces if not host[j]]
        ok, worst = host_gate(hf)
        return ok and cavity_gate(cf, np.array([before_cavity[j] for j in cf]).reshape(-1, 3)), worst

    def try_ulp(face):
        best = None
        for x in [int(x) for x in f[face] if derived(int(x))]:
            cf = [j for j in incident[x] if not host[j]]
            before = {j: c for j, c in zip(cf, cross(cf))} if cf else {}
            base = v[x].copy()
            for o in itertools.product(range(-MAX_ULP, MAX_ULP + 1), repeat=3):
                p = base.copy()
                for k in range(3):
                    for _ in range(abs(o[k])):
                        p[k] = np.nextafter(p[k], np.inf if o[k] > 0 else -np.inf)
                v[x] = p
                ok, worst = fan_ok(x, before)
                if ok:
                    score = (worst, -float(np.linalg.norm(p - base)))
                    if best is None or score > best[0]:
                        best = (score, x, p.copy())
            v[x] = base
        if best is None:
            return None
        _, x, p = best
        move = dict(kind='ulp', face=face, vertex=x, before=v[x].tolist(), after=p.tolist(),
                    displacement_m=float(np.linalg.norm(p - v[x])))
        v[x] = p
        return move

    def proximity_clusters(face):
        """Derived vertices of the face's two-ring, grouped by single linkage at <= MAX_EDGE_M (size >= 2)."""
        ring1 = set().union(*(incident[int(x)] for x in f[face]))
        ring2 = set().union(*(incident[int(x)] for j in ring1 for x in f[j]))
        cand = sorted({int(x) for j in ring2 for x in f[j] if derived(int(x))})
        if len(cand) < 2:
            return []
        p = v[cand]
        close = np.linalg.norm(p[:, None] - p[None], axis=2) <= MAX_EDGE_M
        label, clusters = [-1] * len(cand), []
        for s in range(len(cand)):
            if label[s] >= 0:
                continue
            stack, members = [s], []
            label[s] = len(clusters)
            while stack:
                a = stack.pop(); members.append(cand[a])
                for b in np.flatnonzero(close[a]):
                    if label[b] < 0:
                        label[b] = label[s]; stack.append(b)
            clusters.append(frozenset(members))
        return [c for c in clusters if len(c) >= 2]

    def try_weld(face):
        """Weld near-coincident derived vertices (each onto an existing member, displacement <= MAX_EDGE_M).

        Candidate sets, smallest first: one cluster holding a vertex of the face; all clusters touching the face's one-ring;
        every cluster of its two-ring together."""
        clusters = proximity_clusters(face)
        if not clusters:
            short = min(np.linalg.norm(v[f[face][(i + 1) % 3]] - v[f[face][i]]) for i in range(3))
            return None, dict(face=face, reasons=['no derived vertices within %.0e m (shortest edge %.2e m)' % (MAX_EDGE_M, short)])
        own = {int(x) for x in f[face]}
        ring1 = {int(x) for j in set().union(*(incident[x] for x in own)) for x in f[j]}
        sets = [[c] for c in clusters if c & own]
        touching = [c for c in clusters if c & ring1]
        if len(touching) > 1:
            sets.append(touching)
        if len(clusters) > len(touching):
            sets.append(clusters)
        reasons = []
        for chosen in sets:
            move, reason = weld(face, chosen)
            if move is not None:
                return move, None
            reasons.append('%d cluster(s) of %s vertices: %s' % (len(chosen), [len(c) for c in chosen], reason))
        return None, dict(face=face, reasons=reasons)

    def try_flip(face):
        """Flip one edge shared with a host face of the SAME source face: no vertex moves, the retained area of that
        source face is unchanged; both new triangles must pass the unchanged gate and the new edge must not exist."""
        tri = [int(x) for x in f[face]]
        for i in range(3):
            a, b, c = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
            others = [g for g in incident[a] & incident[b] if g != face and alive[g]]
            if len(others) != 1 or not host[others[0]] or sid[others[0]] != sid[face]:
                continue
            g = others[0]
            gt = [int(x) for x in f[g]]
            e = next(x for x in gt if x not in (a, b))
            if e in {int(y) for j in incident[c] for y in f[j]}:
                continue  # edge c-e already exists
            saved = f[[face, g]].copy()
            f[face], f[g] = (a, e, c), (e, b, c)
            ok, worst = host_gate([face, g])
            if ok:
                incident[a].discard(g); incident[b].discard(face); incident[c].add(g); incident[e].add(face)
                return dict(kind='flip', face=face, neighbour=int(g), edge=[a, b], new_edge=[c, e], displacement_m=0.)
            f[[face, g]] = saved
        return None

    def weld(face, chosen):
        members = set().union(*chosen)
        touched = set().union(*(incident[x] for x in members))
        cf_all = sorted(j for j in touched if not host[j])
        before = {j: c for j, c in zip(cf_all, cross(cf_all))} if cf_all else {}
        options = [[k for k in sorted(c) if max(np.linalg.norm(v[x] - v[k]) for x in c) <= MAX_EDGE_M] for c in chosen]
        if any(not o for o in options):
            return None, 'cluster diameter exceeds MAX_EDGE_M from every member'
        best = None
        for reps in itertools.islice(itertools.product(*options), 64):
            mapping = {x: r for c, r in zip(chosen, reps) for x in c}
            mapped = {j: [mapping.get(int(x), int(x)) for x in f[j]] for j in touched}
            doomed = {j for j, t in mapped.items() if len(set(t)) < 3}
            fan = sorted(touched - doomed)
            saved = f[fan].copy()
            f[fan] = np.array([mapped[j] for j in fan], dtype=f.dtype).reshape(-1, 3)
            directed = [(int(t[k]), int(t[(k + 1) % 3])) for t in f[fan] for k in range(3)]
            around = [e for e in directed if e[0] in reps or e[1] in reps]
            around_set = set(around)
            manifold = (len(around_set) == len(around) and all((e[1], e[0]) in around_set for e in around)
                        and len({tuple(sorted(map(int, t))) for t in f[fan]}) == len(fan))
            ok, worst = host_gate([j for j in fan if host[j]])
            cf = [j for j in fan if not host[j]]
            ok = manifold and ok and cavity_gate(cf, np.array([before[j] for j in cf]).reshape(-1, 3))
            f[fan] = saved
            if ok and (best is None or worst > best[0]):
                best = (worst, reps, mapping, doomed, fan, mapped)
        if best is None:
            return None, 'no choice of kept vertices keeps the fans closed and every incident face within the gates'
        _, reps, mapping, doomed, fan, mapped = best
        f[fan] = np.array([mapped[j] for j in fan], dtype=f.dtype).reshape(-1, 3)
        removed = []
        for j in sorted(doomed):
            alive[j] = False
            removed.append(dict(face=j, host=bool(host[j]), original_id=int(oid[j]), source_face_id=int(sid[j]),
                                role=removed_role(roles.get(int(oid[j]), {}), int(sid[j]), host[j]),
                                area_m2=float(np.linalg.norm(np.cross(*(v0[faces0[j]][1:] - v0[faces0[j]][0]))) / 2)))
        dropped = sorted(x for x in members if mapping[x] != x)
        for j in doomed:
            for x in f[j]:  # doomed faces keep their current (pre-weld) indices
                if int(x) in incident:
                    incident[int(x)].discard(j)
        for r in set(reps):
            incident[r] = {j for j in fan if r in f[j]}
        for x in dropped:
            incident.pop(x, None)
        return dict(kind='weld', face=face, clusters=[sorted(c) for c in chosen], kept_vertices=list(map(int, reps)),
                    dropped_vertices=dropped,
                    displacement_m=max(float(np.linalg.norm(v[mapping[x]] - v[x])) for x in dropped),
                    removed_faces=removed), None

    def candidate_sets(face):
        """tier 2's cluster sets of a face (try_weld), smallest first"""
        clusters = proximity_clusters(face)
        if not clusters:
            return []
        own = {int(x) for x in f[face]}
        ring1 = {int(x) for j in set().union(*(incident[x] for x in own)) for x in f[j]}
        sets = [[c] for c in clusters if c & own]
        touching = [c for c in clusters if c & ring1]
        if len(touching) > 1:
            sets.append(touching)
        if len(clusters) > len(touching):
            sets.append(clusters)
        return sets

    def place(chosen, targets, before, commit=False):
        """map each cluster onto its smallest member placed at its target; gates as weld(); restored unless commit"""
        members = set().union(*chosen)
        reps = [min(c) for c in chosen]
        mapping = {x: r for c, r in zip(chosen, reps) for x in c}
        touched = set().union(*(incident[x] for x in members))
        mapped = {j: [mapping.get(int(x), int(x)) for x in f[j]] for j in touched}
        doomed = {j for j, t in mapped.items() if len(set(t)) < 3}
        fan = sorted(touched - doomed)
        saved_v, saved_f = v[reps].copy(), f[fan].copy()
        shift = max(float(np.linalg.norm(t - v[x])) for c, t in zip(chosen, targets) for x in c)
        for r, t in zip(reps, targets):
            v[r] = t
        f[fan] = np.array([mapped[j] for j in fan], dtype=f.dtype).reshape(-1, 3)
        directed = [(int(t[k]), int(t[(k + 1) % 3])) for t in f[fan] for k in range(3)]
        around = [e for e in directed if e[0] in reps or e[1] in reps]
        around_set = set(around)
        manifold = (len(around_set) == len(around) and all((e[1], e[0]) in around_set for e in around)
                    and len({tuple(sorted(map(int, t))) for t in f[fan]}) == len(fan))
        ok, worst = host_gate([j for j in fan if host[j]])
        cf = [j for j in fan if not host[j]]
        ok = bool(manifold and ok and shift <= MAX_EDGE_M and cavity_gate(cf, np.array([before[j] for j in cf]).reshape(-1, 3)))
        if not commit:
            v[reps], f[fan] = saved_v, saved_f
            return ok, worst, manifold, shift
        removed = []
        for j in sorted(doomed):
            alive[j] = False
            removed.append(dict(face=j, host=bool(host[j]), original_id=int(oid[j]), source_face_id=int(sid[j]),
                                role=removed_role(roles.get(int(oid[j]), {}), int(sid[j]), host[j]),
                                area_m2=float(np.linalg.norm(np.cross(*(v0[faces0[j]][1:] - v0[faces0[j]][0]))) / 2)))
        dropped = sorted(x for x in members if mapping[x] != x)
        for j in doomed:
            for x in f[j]:  # doomed faces keep their pre-weld indices (not in fan)
                if int(x) in incident:
                    incident[int(x)].discard(j)
        for r in set(reps):
            incident[r] = {j for j in fan if r in f[j]}
        for x in dropped:
            incident.pop(x, None)
        return reps, dropped, removed, shift

    def lp_target(ci, chosen, targets, frame, before):
        """the LP of one cluster with the other clusters at their targets: orientation margins of its mapped fan faces"""
        origin, B = frame
        reps = [min(c) for c in chosen]
        mapping = {x: r for c, r in zip(chosen, reps) for x in c}
        pos = {r: t for r, t in zip(reps, targets)}
        me = reps[ci]
        G, h, A, b = [], [], [], []
        for j in set().union(*(incident[x] for x in chosen[ci])):
            t = [mapping.get(int(x), int(x)) for x in f[j]]
            if len(set(t)) < 3 or me not in t:
                continue
            i = t.index(me)
            a_, b_ = (pos.get(x, v[x]) for x in (t[(i + 1) % 3], t[(i + 2) % 3]))
            g3 = np.cross(sn[sid[j]] / snl[sid[j]], b_ - a_) if host[j] else np.cross(before[j], b_ - a_)
            gx = B.T @ g3
            ng = float(np.linalg.norm(gx))
            if ng == 0.:
                if (origin - a_) @ g3 <= 0:
                    return None
                continue
            G.append(gx / ng); h.append(float((origin - a_) @ g3) / ng)
            if host[j]:  # inside its host source triangle
                s = sid[j]
                for e in range(3):
                    p0, p1 = st[s, e], st[s, (e + 1) % 3]
                    ge = B.T @ np.cross(sn[s] / snl[s], p1 - p0)
                    ne = float(np.linalg.norm(ge))
                    if ne > 0:
                        A.append(-ge / ne); b.append(float((origin - p0) @ np.cross(sn[s] / snl[s], p1 - p0)) / ne)
        k = B.shape[1]
        dirs = np.array([[1.], [-1.]]) if k == 1 else np.c_[np.cos(np.arange(PW_POLY) * 2 * np.pi / PW_POLY), np.sin(np.arange(PW_POLY) * 2 * np.pi / PW_POLY)]
        for x in chosen[ci]:
            xk = B.T @ (v[x] - origin)
            rk2 = MAX_EDGE_M ** 2 - (float(np.linalg.norm(v[x] - origin)) ** 2 - float(xk @ xk))
            if rk2 <= 0 or not G:
                return None
            rk = np.sqrt(rk2) * (1. if k == 1 else np.cos(np.pi / PW_POLY))
            for u in dirs:
                A.append(u); b.append(float(u @ xk) + rk)
        sol = lp_max_margin(np.array(G), np.array(h), np.array(A).reshape(-1, k), np.array(b))
        if sol is None or sol[1] <= 0:
            return None
        return origin + B @ sol[0], sol[1]

    def try_projected(face):
        """tier 4: projected-target weld of tier 2's cluster sets whose member weld is index-manifold"""
        reasons = []
        for chosen in candidate_sets(face):
            label = '%d cluster(s) of %s vertices' % (len(chosen), [len(c) for c in chosen])
            members = set().union(*chosen)
            touched = set().union(*(incident[x] for x in members))
            cf_all = sorted(j for j in touched if not host[j])
            before = {j: c for j, c in zip(cf_all, cross(cf_all))} if cf_all else {}
            if not place(chosen, [v[min(c)].copy() for c in chosen], before)[2]:
                reasons.append(label + ': the member weld is not index-manifold (independent of the target)')
                continue
            frames, options, why = [], [], None
            for c in chosen:
                hs = sorted({int(sid[j]) for j in set().union(*(incident[x] for x in c)) if host[j]})
                if not hs:
                    frames.append(None); options.append([v[k].copy() for k in sorted(c)]); continue
                fr = frame_of([(sn[s] / snl[s], st[s, 0]) for s in hs], v[sorted(c)].mean(0))
                if fr[0] is None:
                    why = fr[1]; break
                origin, B = fr
                frames.append(dict(frame=fr, host_source_faces=hs))
                opts = []
                for k in sorted(c):
                    p = origin + B @ (B.T @ (v[k] - origin))
                    if max(float(np.linalg.norm(p - v[x])) for x in c) <= MAX_EDGE_M and not any(np.array_equal(p, q) for q in opts):
                        opts.append(p)
                options.append(opts)
            if why is not None:
                reasons.append(label + ': ' + why); continue
            best, method = None, 'member_projection'
            for targets in itertools.islice(itertools.product(*options), 64):
                ok, worst, _, shift = place(chosen, list(targets), before)
                if ok and (best is None or (worst, -shift) > best[0]):
                    best = ((worst, -shift), [t.copy() for t in targets], None)
            if best is None:
                targets = [o[0].copy() if o else v[sorted(c)].mean(0) for o, c in zip(options, chosen)]
                margin = None
                for _ in range(2):
                    for ci, fr in enumerate(frames):
                        if fr is None:
                            continue
                        got = lp_target(ci, chosen, targets, fr['frame'], before)
                        if got is not None:
                            targets[ci], margin = got
                ok, worst, _, shift = place(chosen, targets, before)
                if ok:
                    best, method = ((worst, -shift), targets, margin), 'lp'
            if best is None:
                reasons.append(label + ': no projected or LP target keeps the fans closed and every incident face within the gates')
                continue
            _, targets, margin = best
            reps, dropped, removed, shift = place(chosen, targets, before, commit=True)
            return dict(kind='projected_weld', face=face, rule=PW_RULE, method=method, lp_margin_m=margin,
                        clusters=[sorted(c) for c in chosen], kept_vertices=list(map(int, reps)), dropped_vertices=dropped,
                        targets=[list(map(float, t)) for t in targets],
                        frames=[None if fr is None else dict(host_source_faces=fr['host_source_faces'],
                                                             kind='plane' if fr['frame'][1].shape[1] == 2 else 'line') for fr in frames],
                        displacement_m=shift, removed_faces=removed), None
        return None, (reasons or ['no derived-vertex cluster within %.0e m' % MAX_EDGE_M])

    hidx = np.flatnonzero(host)
    c = cross(hidx)
    cos0 = np.einsum('ij,ij->i', c, sn[sid[hidx]]) / (np.linalg.norm(c, axis=1) * snl[sid[hidx]])
    bad = [int(x) for x in hidx[cos0 < GATE]]
    # Passes repeat while any face is repaired: removing one needle of an adjacent pair can unblock the other.
    repairs, pending = [], list(bad)
    while True:
        unresolved, progress = [], False
        for face in pending:
            if not alive[face] or host_gate([face])[0]:
                continue
            move = try_ulp(face)
            if move is None:
                move, failure = try_weld(face)
                if move is None:
                    move = try_flip(face)
                if move is None:
                    unresolved.append(failure)
                    continue
            repairs.append(move)
            progress = True
        pending = [r['face'] for r in unresolved]
        if not unresolved or not progress:
            break
    tier4 = None
    if unresolved:
        tier4 = dict(rule=PW_RULE, started_with=[r['face'] for r in unresolved], repaired=[])
        while True:
            left, progress = [], False
            for face in pending:
                if not alive[face] or host_gate([face])[0]:
                    continue
                move = try_ulp(face)
                if move is None:
                    move, failure = try_weld(face)
                    if move is None:
                        move = try_flip(face)
                    if move is None:
                        move, why = try_projected(face)
                        if move is None:
                            left.append(dict(failure, tier4_reasons=why))
                            continue
                repairs.append(move)
                tier4['repaired'].append(face)
                progress = True
            pending = [r['face'] for r in left]
            unresolved = left
            if not left or not progress:
                break
        tier4['unresolved'] = [r['face'] for r in unresolved]

    del incident, roles, st, faces0  # the passes are done
    keep_faces = np.flatnonzero(alive)
    kept = f[keep_faces]
    used = np.zeros(len(v), bool)
    used[kept] = True
    used = np.flatnonzero(used)  # = np.unique(kept)
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    out_v, out_f = v[used], remap[kept]
    del kept
    out_oid, out_sid = oid[keep_faces], sid[keep_faces]
    out_host = out_oid == build['host_original_id']
    p0 = out_v[out_f[:, 0]]  # the rows of out_v[out_f][:, k], one corner at a time
    oc = np.cross(out_v[out_f[:, 1]] - p0, out_v[out_f[:, 2]] - p0)
    del p0
    oa = np.linalg.norm(oc, axis=1)
    hcos = np.einsum('ij,ij->i', oc[out_host], sn[out_sid[out_host]]) / (oa[out_host] * snl[out_sid[out_host]])
    source_area = snl / 2
    area_before = np.bincount(sid[hidx], weights=np.linalg.norm(c, axis=1) / 2, minlength=len(source_area))
    area_after = np.bincount(out_sid[out_host], weights=oa[out_host] / 2, minlength=len(source_area))
    source_moved = int(sum(not derived(int(x)) for x in np.flatnonzero((v != v0).any(1))))
    report = dict(source_case=case.name, repaired_case=new.name, gate=GATE, max_ulp_per_axis=MAX_ULP,
        max_collapse_edge_m=MAX_EDGE_M, initially_below_gate=bad, repairs=repairs, unresolved=unresolved,
        host_below_gate_after=int((hcos < GATE).sum()), min_host_cos_after=float(hcos.min()),
        min_triangle_area_after_m2=float(oa.min() / 2),
        max_source_area_gain_m2=float((area_after - source_area).max()),
        max_source_area_change_vs_raw_m2=float(np.abs(area_after - area_before).max()),
        removed_face_count=int((~alive).sum()), removed_vertex_count=int(len(v) - len(used)),
        source_vertices_moved=source_moved,
        max_displacement_m=max([r['displacement_m'] for r in repairs], default=0.),
        closed_index_manifold_consistent_winding=closed_manifold(out_f))
    if tier4 is not None:
        report['tier4'] = tier4
    print(json.dumps({k: x for k, x in report.items() if k != 'repairs'}, indent=1))
    for r in repairs:
        print(r['kind'], r['face'], '%.3e' % r['displacement_m'],
              [(x['role'], '%.1e' % x['area_m2']) for x in r.get('removed_faces', [])])
    if (unresolved or report['host_below_gate_after'] or source_moved or oa.min() <= 0
            or report['max_source_area_gain_m2'] > AREA_GAIN_M2
            or not report['closed_index_manifold_consistent_winding']):
        return report, False
    new.mkdir(parents=True)
    tables = {p.name for path in build['paths'] for p in source_table_paths(case, path['index'])}
    for p in case.iterdir():
        if p.is_file() and p.name not in ('difference.npz', 'host_provenance_check.json'):
            if link_source_tables and p.name in tables:
                try:
                    os.link(p, new / p.name)
                    continue
                except OSError as error:
                    print('source table %s copied, not linked: %s' % (p.name, error))
            shutil.copy2(p, new / p.name)
    np.savez_compressed(new / 'difference.npz', vertices=out_v, faces=out_f.astype(faces_dtype),
                        original_id=out_oid, source_face_id=out_sid)
    (new / 'host_needle_repair.json').write_text(json.dumps(report, indent=1), encoding='utf8')
    return report, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('new_case')
    parser.add_argument('--copy-source-tables', action='store_true')
    args = parser.parse_args()
    case, new = ROOT / 'cases' / args.case, ROOT / 'cases' / args.new_case
    if new.exists():
        raise FileExistsError(new)
    _, ok = repair(case, new, link_source_tables=not args.copy_source_tables)
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()

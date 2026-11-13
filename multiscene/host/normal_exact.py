"""Exact evaluator of the eroded surface on the normal-offset route.

Every term is BluffExact's, in the same order: reference relief (broad + meso noise, the kept facets, D1 self-affine
relief) at the base point, recession, then the joint grooves centred on the v2 planes at the surface without grooves
('moved' = positions(pre + amplitude) - surface, computed as erode does for a sheet), total = (pre + groove) * gscale.
The smooth per-vertex fields (meso, w, fade, recess, bare_w, gscale) are interpolated from the coarse vertices as
before. Parameter: the carried 'd' is b - origin, b a point of the coarse base mesh (a vertex, or the barycentric /
red-green midpoint combination of vertices), origin = the exposure's interior origin. The existing refinement and
barycentric recovery work on it unchanged (linear, solved by the same 3 x 3 systems). The base point is b projected
onto envelope_distance = 0. A scene without intrusions and faults takes the version-2 code path below unchanged."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parent)) if str(_p.Path(__file__).resolve().parent) not in _s.path else None
import nogpu  # noqa: F401,E402  (CPU only)
import math
import pickle
from pathlib import Path

import numpy as np

from outcrop.dsgen.morph import facet_noise, facet_noise_c1, noise, self_affine_relief
from outcrop.dsgen.truth import TruthModel, _inside_triangles, _plane_groups
import normal_surface as ns

CARRY = ('meso', 'w', 'fade', 'recess', 'bare_w', 'gscale')


def load_fields(fields_pkl):
    return fields_pkl if isinstance(fields_pkl, dict) else pickle.load(open(fields_pkl, 'rb'))


class NormalExact:
    route = 'normal'

    def __init__(self, fields_pkl, geom):
        d = load_fields(fields_pkl)
        nr = (d.get('expo') or {}).get('normal_route') or {}
        if not (np.asarray(d['fields']['sheet']).all() or nr): raise ValueError('normal-route evaluator expects normal-route fields')
        self.crest_cap = bool(nr.get('crest_cap'))
        s = d['scalars']
        for flag in ('smooth_fields', 'relief_c1', 'relief_cm'):
            if s[flag]: raise ValueError('unsupported erode option %s' % flag)
        if s['groove_end_taper'] is not None: raise ValueError('groove end taper not mirrored')
        self.groove_tip = s.get('groove_tip')
        self._gtf = {}
        self.v3 = bool(d['sets']['faults'] or d['sets']['intrusions'])
        self.s, self.sets, self.fields, self.env = s, d['sets'], d['fields'], d['expo']
        self.origin = np.asarray(self.env['origin_m'], float)
        self.truth = TruthModel(Path(geom)/'source_package')
        self.pgroups = {}

    def coarse_carry(self):
        f = self.fields
        c = {k: np.asarray(f[k], float).copy() for k in CARRY}
        c['d'] = np.asarray(f['base'], float) - self.origin
        return c

    def frame(self, d):
        base = ns.project(self.origin + np.asarray(d, float), self.env)
        return base, ns.normal(base, self.env)

    def at(self, base, n, off, k):
        """erode's patch positions: base - n * offset (sheet) or base - n * cap_offset(offset, k)"""
        off = np.asarray(off, float)
        return base - n * (ns.cap_offset(off, k) if k is not None else off)[:, None]

    def grooves(self, owners, base, base_off, surface, amp, width, reach, n, varied=False, k=None):
        seed = self.s['seed']
        moved = self.at(base, n, base_off + amp, k) - surface          # erode: positions(W, base_off + amp) - surface
        depth, win, prof = np.zeros(len(surface)), np.zeros(len(surface)), np.zeros(len(surface))
        for k, (iid, iv, if_) in enumerate(self.truth.interfaces):
            if iid.split(':', 1)[0] not in owners or not len(if_):
                continue
            per = .5 + .5 * math.sin(k * 12.9898 + seed) if varied else 1.
            fa, fw = (.4 + .6 * per, .7 + .6 * per) if varied else (1., 1.)
            if self.groove_tip is not None and iid not in self._gtf:
                import groove_tip as gt
                self._gtf[iid] = gt.Fracture(iv, if_)
            if iid not in self.pgroups:
                self.pgroups[iid] = self._gtf[iid].groups if self.groove_tip is not None else _plane_groups(iv, if_)
            for gi, (normal, offset, tri) in enumerate(self.pgroups[iid]):
                dd = surface @ normal - offset
                idx = np.nonzero(np.abs(dd) < reach)[0]
                if not len(idx):
                    continue
                proj = surface[idx] - dd[idx, None] * normal
                if self.groove_tip is not None:
                    ins = self._gtf[iid].allow(gi, surface[idx], proj, normal, tri)
                else:
                    ins = _inside_triangles(proj, tri, normal, eps=1e-4)
                idx, proj = idx[ins], proj[ins]
                c = dd[idx] + fa * (moved[idx] @ normal)
                env = np.exp(-(c / (width * fw)) ** 2)
                penv = np.exp(-(c / width) ** 2)
                if self.groove_tip is not None and len(idx):
                    tp = self._gtf[iid].taper(proj, float(self.groove_tip['taper_m']))
                    env, penv = env * tp, penv * tp
                g = fa * amp[idx] * env
                better = g > depth[idx]
                win[idx[better]] = fa * amp[idx[better]]
                depth[idx[better]] = g[better]
                prof[idx] = np.maximum(prof[idx], penv)
        return depth, win, prof

    def evaluate(self, c):
        s = self.s; seed = s['seed']
        base, n = self.frame(c['d'])
        w, meso = c['w'], c['meso']
        x, y, z = base.T
        rough = s['broad'] * noise(x / 4. + y / 7., z / 3., seed + 12)
        rough = rough + meso * (w[:, 0] * noise(y / .3, z / .2, seed + 15) + w[:, 1] * noise(x / .3, z / .2, seed + 16)
                                + w[:, 2] * noise(x / .3, y / .2, seed + 17))
        sa = s.get('self_affine')
        for i, (scale, amp) in enumerate(s['facets']):
            if sa is not None and i not in sa['keep_facets']:
                continue
            fn = facet_noise_c1 if sa is not None and sa['facets_c1'] else facet_noise
            a = np.deg2rad(23 + i * 47)
            cs, sn = math.cos(a), math.sin(a)
            for axis, (v, u) in enumerate(((y, z), (x, z), (x, y))):
                warp = .3 * noise(v / (2 * scale), u / (2 * scale), seed + 105 + i)
                rough = rough + w[:, axis] * amp * s['fscale'] * fn((v * cs + u * sn) / scale + warp,
                                                                     (u * cs - v * sn) / scale - warp, seed + 125 + i + axis)
        if sa is not None:
            rough = rough + self_affine_relief(base, sa['octaves'], sa['seed'])
        rough = rough * s['relief']
        if s['fade_on_cover']:
            rough = rough * c['fade']
        pre = -rough + c['recess']
        k = ns.principal_curvatures(base, self.env) if self.crest_cap else None
        if self.v3:
            return self._step3_v3(c, base, n, pre, k)
        v_f = self.at(base, n, pre, k)                      # with_fault = pre (no fault trench in these models)
        xx, yy, zz = v_f.T
        along = np.clip(noise(xx / 2. + yy / 3., zz / 2.5, seed + 90) + .55, 0, 1)
        jg, _, _ = self.grooves(set(self.sets['joints']), base, pre, v_f, s['jd'] * along * c['fade'], s['jw'], 4 * s['jw'] + .5, n, varied=True, k=k)
        total = (pre + jg) * c['gscale']
        return dict(V=self.at(base, n, total, k), total=total, joint_groove=jg, base=base, normal=n)

    def _step3_v3(self, c, base, n, pre, k):
        s = self.s; seed = s['seed']; zero = np.zeros(len(pre))
        v_ng = self.at(base, n, pre, k)
        x, y, z = v_ng.T
        noise_f = .72 + .28 * noise(z / 3.5, y / 3., seed + 80)
        along = np.clip(noise(x / 2. + y / 3., z / 2.5, seed + 90) + .55, 0, 1)
        if self.sets['faults']:
            a_fault = s['fault_recess'] * (1. - .8 * c['w'][:, 2]) * c['fade'] * noise_f
            fdepth, _, gf = self.grooves(set(self.sets['faults']), base, pre, v_ng, a_fault, s['damage'], 3 * s['damage'] + 1., n, k=k)
        else:
            fdepth, gf = zero, zero
        with_fault = pre + fdepth
        v_f = self.at(base, n, with_fault, k)
        inside_fault = np.clip(1 - gf, 0, 1)
        jg, _, _ = self.grooves(set(self.sets['joints']), base, with_fault, v_f, s['jd'] * along * c['fade'] * inside_fault, s['jw'],
                                4 * s['jw'] + .5, n, varied=True, k=k)
        if self.sets['intrusions']:
            ig, _, _ = self.grooves(set(self.sets['intrusions']), base, with_fault, v_f, .5 * s['jd'] * c['fade'] * inside_fault, s['jw'],
                                    4 * s['jw'] + .5, n, k=k)
        else:
            ig = zero
        total = (pre + (fdepth + np.maximum(jg, ig))) * c['gscale']
        return dict(V=self.at(base, n, total, k), total=total, joint_groove=jg, contact_groove=ig, fault_groove=fdepth, fault_profile=gf,
                    base=base, normal=n)

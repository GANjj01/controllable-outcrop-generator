"""It mirrors outcrop.dsgen.processes.erode for a re-solved (radial) patch without the cm option, term by term: reference
relief (broad + meso noise, facets), recession, then the v2 joint grooves centred on the plane at the solved surface,
every position re-solved along the ray (outcrop.dsgen.morph.radial_surface with the exposure record), exactly like the
patch's own ``apply``. Per-vertex fields that erode builds by mesh smoothing on the 11 cm sheet and that are smooth at
>= 0.3 m (meso, w, fade, recess, bare_w, gscale) are interpolated from the coarse vertices, as the existing cm route
does; everything else is re-evaluated at the point. At the coarse vertices, with their own fields, it must reproduce
the source surface (checked). Inputs: the field export of bluff_reproduce.py (erode with debug['export_fields'])."""
import math
import pickle
import sys
from pathlib import Path

import numpy as np

from outcrop.dsgen.morph import facet_noise, facet_noise_c1, noise, radial_surface, self_affine_relief
from outcrop.dsgen.truth import TruthModel, _inside_triangles, _plane_groups

sys.path.append(str(Path(__file__).resolve().parents[2]))
from frgb import runconfig as rc  # noqa: E402
GEOM = rc.path('geometry.dev_t02_directions')/'geometry/dev_t02_v2_attempt3'
CARRY = ('meso', 'w', 'fade', 'recess', 'bare_w', 'gscale')


class BluffExact:
    def __init__(self, fields_pkl, geom=None):
        d = pickle.load(open(fields_pkl, 'rb'))
        if d['fields']['sheet'].any(): raise ValueError('bluff evaluator expects re-solved patches only')
        s = d['scalars']
        for flag in ('smooth_fields', 'relief_c1', 'relief_cm'):
            if s[flag]: raise ValueError('unsupported erode option %s' % flag)
        if s['groove_end_taper'] is not None: raise ValueError('groove end taper not mirrored')
        if d['sets']['faults'] or d['sets']['intrusions']: raise ValueError('only joint grooves are mirrored')
        self.s, self.sets, self.fields, self.env = s, d['sets'], d['fields'], d['expo']
        self.truth = TruthModel(Path(geom or GEOM)/'source_package')   # geom: another carrier's directory (D1, fx03)
        self.pgroups = {}

    def coarse_carry(self):
        f = self.fields
        c = {k: np.asarray(f[k], float).copy() for k in CARRY}
        c['d'] = np.asarray(f['outward'], float).copy()
        return c

    def solve(self, d, off):
        return radial_surface(d, np.asarray(off, float), self.env)[0]

    def grooves(self, owners, base_off, surface, amp, width, reach, d, varied=False):
        seed = self.s['seed']
        moved = self.solve(d, base_off + amp) - surface
        depth, win, prof = np.zeros(len(surface)), np.zeros(len(surface)), np.zeros(len(surface))
        for k, (iid, iv, if_) in enumerate(self.truth.interfaces):
            if iid.split(':', 1)[0] not in owners or not len(if_):
                continue
            per = .5 + .5 * math.sin(k * 12.9898 + seed) if varied else 1.
            fa, fw = (.4 + .6 * per, .7 + .6 * per) if varied else (1., 1.)
            if iid not in self.pgroups:
                self.pgroups[iid] = _plane_groups(iv, if_)
            for normal, offset, tri in self.pgroups[iid]:
                dd = surface @ normal - offset
                idx = np.nonzero(np.abs(dd) < reach)[0]
                if not len(idx):
                    continue
                proj = surface[idx] - dd[idx, None] * normal
                ins = _inside_triangles(proj, tri, normal, eps=1e-4)
                idx = idx[ins]
                c = dd[idx] + fa * (moved[idx] @ normal)
                env = np.exp(-(c / (width * fw)) ** 2)
                penv = np.exp(-(c / width) ** 2)
                g = fa * amp[idx] * env
                better = g > depth[idx]
                win[idx[better]] = fa * amp[idx[better]]
                depth[idx[better]] = g[better]
                prof[idx] = np.maximum(prof[idx], penv)
        return depth, win, prof

    def evaluate(self, c):
        s = self.s; seed = s['seed']
        d = c['d'] / np.linalg.norm(c['d'], axis=1, keepdims=True)
        w, meso = c['w'], c['meso']
        base = self.solve(d, np.zeros(len(d)))
        x, y, z = base.T
        rough = s['broad'] * noise(x / 4. + y / 7., z / 3., seed + 12)
        rough = rough + meso * (w[:, 0] * noise(y / .3, z / .2, seed + 15) + w[:, 1] * noise(x / .3, z / .2, seed + 16)
                                + w[:, 2] * noise(x / .3, y / .2, seed + 17))
        sa = s.get('self_affine')                           # D1 opt-in mode (exported spec: octaves, seed, kept facets)
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
        v_ng = self.solve(d, pre)
        xx, yy, zz = v_ng.T
        along = np.clip(noise(xx / 2. + yy / 3., zz / 2.5, seed + 90) + .55, 0, 1)
        inside_fault = 1.                                   # no faults in this model
        v_f = self.solve(d, pre)                            # with_fault = pre (no fault trench)
        jg, _, _ = self.grooves(set(self.sets['joints']), pre, v_f, s['jd'] * along * c['fade'] * inside_fault, s['jw'],
                                4 * s['jw'] + .5, d, varied=True)
        total = (pre + jg) * c['gscale']
        return dict(V=self.solve(d, total), total=total, joint_groove=jg, base=base)

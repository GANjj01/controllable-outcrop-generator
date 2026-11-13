STAIN_NAMES = ('width_m', 'factor', 'amplitude_m', 'wavelength_m')
STAIN_BOUNDS = ((.001, .02), (.5, .8), (0., .2), (.5, 5.))      # bounds of width_m, factor, amplitude_m, wavelength_m
STAIN_SCALARS = ('StainWeight', 'StainP0X', 'StainP0Y', 'StainP0Z', 'StainNX', 'StainNY', 'StainNZ', 'StainTX', 'StainTY',
                 'StainTZ', 'StainAmp', 'StainLambda', 'StainPhase', 'StainWidth', 'StainSoft', 'StainFR', 'StainFG', 'StainFB')


def stain_physical(unit):
    if len(unit) != 4 or any(not 0. <= float(x) <= 1. for x in unit):
        raise ValueError('normalized stain parameters must lie in [0,1]')
    return {n: float(lo + (hi - lo) * float(x)) for n, (lo, hi), x in zip(STAIN_NAMES, STAIN_BOUNDS, unit)}


def _stainable_host():
    """the neutral label of the stainable host; read when used, so the Blender side, which never stains, does not need the
    run configuration"""
    import json, sys
    from pathlib import Path
    p1 = Path(__file__).resolve().parents[2]
    if str(p1) not in sys.path:
        sys.path.append(str(p1))
    from frgb import runconfig as frgb_rc
    return json.loads(frgb_rc.path('labels.lithology_aliases_r1').read_text(encoding='utf8'))['stainable_host']


def _host_names():
    """the stainable host under every name a record may hold (frgb.lith_records.host_names): scenes generated before the rename keep the
    former name in their code books and frames"""
    _stainable_host()
    from frgb import lith_records
    return lith_records.host_names()


def raw_stain(parameters, frame, off):
    host = _stainable_host()
    names = _host_names()
    if not frame['lithologies'] or set(frame['lithologies']) - set(names): raise ValueError('Current physical stain evidence only covers the stainable host %s' % host)
    if not 30. <= float(frame['angle_to_layer_normal_deg']) <= 90.: raise ValueError('Prequalified cross-layer stain orientation required')
    return dict(lithologies=names, weight=0. if off else 1., factor=[parameters['factor']] * 3,
                origin_m=frame['origin_m'], normal=frame['normal_enu'], tangent=frame['tangent_enu'],
                amplitude_m=parameters['amplitude_m'], wavelength_m=parameters['wavelength_m'], phase=frame['phase'],
                width_m=parameters['width_m'], soft_m=.25 * parameters['width_m'])


def stain_scalars(stain, eligible):
    """stain: the staged request record (outcrop.workbench.scanned._stain output, local cm)."""
    o, n, t, f = stain['origin_cm'], stain['normal_local'], stain['tangent_local'], stain['factor']
    return dict(StainWeight=float(stain['weight']) if eligible else 0., StainP0X=o[0], StainP0Y=o[1], StainP0Z=o[2],
                StainNX=n[0], StainNY=n[1], StainNZ=n[2], StainTX=t[0], StainTY=t[1], StainTZ=t[2],
                StainAmp=float(stain['amplitude_cm']), StainLambda=float(stain['wavelength_cm']), StainPhase=float(stain['phase']),
                StainWidth=float(stain['width_cm']), StainSoft=float(stain['soft_cm']), StainFR=f[0], StainFG=f[1], StainFB=f[2])


def stain_job_parameters(stain, codebook):
    """Per-code worker parameters: every stain-graph code gets explicit values (weight 0 outside the stained lithologies)."""
    eligible = {int(r['code']) for r in codebook if r['lithology'] in stain['lithologies']}
    return {str(int(r['code'])): stain_scalars(stain, int(r['code']) in eligible) for r in codebook if int(r['code']) in eligible}

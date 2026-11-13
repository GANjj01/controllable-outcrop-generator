"""Environment catalogue: 10 categories, 44 presets, explicit parameters and availability.

Every preset is listed so the scope stays visible. ``available`` turns true only when a
real recipe consumes the parameters; names never stand in for an implementation.
"""
from copy import deepcopy

CATEGORIES = (
    ('arid', 'arid regions'), ('forest', 'forest'), ('grass_shrub', 'grassland and shrubland'), ('alpine', 'high mountains'),
    ('cold', 'cold regions'), ('coast', 'coast'), ('river_lake', 'near rivers and lakes'), ('wetland', 'wetland'),
    ('special', 'special landforms'), ('artificial', 'artificial exposures'))

PRESETS = (
    ('arid', 'desert', 'desert'), ('arid', 'gobi', 'gobi'), ('arid', 'gravel_desert', 'gravel desert'),
    ('arid', 'arid_canyon', 'arid canyon'), ('arid', 'saline_flat', 'saline land'),
    ('forest', 'conifer_forest', 'coniferous forest'), ('forest', 'broadleaf_forest', 'broadleaf forest'),
    ('forest', 'mixed_forest', 'mixed forest'), ('forest', 'rainforest', 'tropical rainforest'),
    ('forest', 'open_woodland', 'open woodland'),
    ('grass_shrub', 'grassland', 'steppe'), ('grass_shrub', 'savanna', 'savanna'),
    ('grass_shrub', 'shrub_slope', 'shrub slope'), ('grass_shrub', 'heathland', 'barren land'),
    ('alpine', 'rocky_mountain', 'bare rock mountains'), ('alpine', 'talus_slope', 'scree slope'),
    ('alpine', 'alpine_meadow', 'alpine meadow'), ('alpine', 'snow_mountain', 'snow mountain'),
    ('cold', 'tundra', 'tundra'), ('cold', 'permafrost', 'permafrost'), ('cold', 'snowfield', 'snowfield'),
    ('cold', 'glacier_margin', 'glacier margin'), ('cold', 'polar_rock', 'polar bare rock'),
    ('coast', 'sand_beach', 'sand beach'), ('coast', 'gravel_beach', 'gravel beach'), ('coast', 'rocky_shore', 'rocky shore'),
    ('coast', 'sea_cliff', 'sea cliff'), ('coast', 'intertidal', 'intertidal zone'),
    ('river_lake', 'river_valley', 'river valley'), ('river_lake', 'riverbed', 'river bed'),
    ('river_lake', 'riverbank', 'river bank'), ('river_lake', 'lakeshore', 'lake shore'),
    ('river_lake', 'alluvial_plain', 'alluvial plain'),
    ('wetland', 'swamp', 'marsh'), ('wetland', 'mudflat', 'mud flat'), ('wetland', 'wet_meadow', 'wet meadow'),
    ('wetland', 'mangrove', 'mangrove'),
    ('special', 'karst', 'karst'), ('special', 'lava_field', 'volcanic lava field'),
    ('special', 'aeolian_landform', 'wind-eroded landforms'),
    ('artificial', 'roadcut', 'road cut'), ('artificial', 'quarry', 'quarry'),
    ('artificial', 'open_pit', 'mine pit'), ('artificial', 'excavation', 'engineering excavation'))

# Parameter schemas: (key, label, unit, min, max, default). Groups follow the spec:
# terrain, ecology, water, surface. Only implemented presets carry schemas.
_P = lambda key, label, unit, low, high, default: dict(key=key, label=label, unit=unit, min=low, max=high, default=default)
_CONTACT = [_P('contact_band_m', 'fixed contact band at the rock toe', 'm', 0.5, 20., 3.), _P('transition_m', 'transition width beyond the contact band', 'm', 1., 80., 12.),
            _P('rock_blend_m', 'toe material blend height (0 = off)', 'm', 0., 5., .6), _P('rock_blend_strength', 'toe material blend strength', '', 0., 1., .8),
            _P('palette_near', 'align near-toe ground with the rock palette', '', 0., 1., .5), _P('palette_far', 'align far ground with the rock palette', '', 0., 1., .15)]

SCHEMAS = {
    'desert': dict(
        terrain=_CONTACT + [_P('dune_height_m', 'dune height', 'm', 0., 30., 6.), _P('dune_wavelength_m', 'dune spacing (wavelength)', 'm', 8., 200., 45.),
                            _P('wind_azimuth_deg', 'prevailing wind direction (from, clockwise from north)', 'deg', 0., 360., 300.),
                            _P('crest_sinuosity', 'dune crest sinuosity', '', 0., 1., .45), _P('lee_slope_deg', 'lee slope angle', 'deg', 15., 34., 31.),
                            _P('burial_depth_m', 'sand burial depth at the toe', 'm', 0., 2., .3)],
        ecology=[_P('rock_density_per_100m2', 'scattered stone density', '/100m²', 0., 5., .05)],
        surface=[_P('red_sand_fraction', 'red sand fraction', '', 0., 1., .35)]),
    'gobi': dict(
        terrain=_CONTACT + [_P('relief_m', 'relief amplitude', 'm', 0., 8., 1.6), _P('relief_scale_m', 'relief scale', 'm', 10., 300., 70.),
                            _P('gully_depth_m', 'gully depth', 'm', 0., 4., .9), _P('gully_spacing_m', 'gully spacing', 'm', 10., 200., 45.)],
        ecology=[_P('block_density_per_100m2', 'boulder density', '/100m²', 0., 200., 22.), _P('block_size_m', 'median boulder size', 'm', .05, 1.5, .28),
                 _P('size_spread', 'size dispersion (log)', '', 0., 1.5, .6), _P('clustering', 'clustering', '', 0., 1., .6),
                 _P('shrub_density_per_100m2', 'dryland shrub and grass density', '/100m²', 0., 10., .35)],
        surface=[_P('gravel_fraction', 'gravel surface fraction', '', 0., 1., .55)]),
    'gravel_desert': dict(
        terrain=_CONTACT + [_P('relief_m', 'micro-relief amplitude', 'm', 0., 3., .35), _P('hollow_depth_m', 'shallow hollow depth', 'm', 0., 2., .45),
                            _P('hollow_spacing_m', 'shallow hollow spacing', 'm', 10., 200., 55.)],
        ecology=[_P('pebble_density_per_m2', 'gravel pavement density', '/m²', 0., 40., 9.), _P('pebble_size_m', 'median gravel size', 'm', .01, .3, .045),
                 _P('size_spread', 'size dispersion (log)', '', 0., 1.5, .45), _P('bare_fraction', 'exposed fine substrate fraction', '', 0., 1., .2),
                 _P('boulder_density_per_100m2', 'large stone density', '/100m²', 0., 5., .15)],
        surface=[_P('pavement_fraction', 'gravel pavement material fraction', '', 0., 1., .8)]),
    'arid_canyon': dict(
        terrain=_CONTACT + [_P('plateau_relief_m', 'terrace relief', 'm', 0., 3., .5)],
        ecology=[_P('talus_density_per_100m2', 'toe scree density', '/100m²', 0., 300., 60.),
                 _P('talus_size_m', 'median toe scree size', 'm', .05, 1.5, .3), _P('talus_spread', 'scree size dispersion (log)', '', 0., 1.5, .55),
                 _P('shrub_density_per_100m2', 'terrace dryland shrub and grass density', '/100m²', 0., 10., .4)],
        surface=[_P('floor_gravel_fraction', 'gravel river bed fraction on the floor', '', 0., 1., .8)]),
    'saline_flat': dict(
        terrain=_CONTACT + [_P('basin_depth_m', 'shallow hollow depth', 'm', 0., 2., .35), _P('basin_scale_m', 'shallow hollow scale', 'm', 10., 300., 60.)],
        surface=[_P('crust_thickness_m', 'salt crust thickness', 'm', .005, .2, .03), _P('polygon_size_m', 'polygonal crack scale', 'm', .3, 10., 1.6),
                 _P('ridge_height_m', 'crack ridge height', 'm', 0., .3, .05), _P('crust_coverage', 'salt crust cover fraction', '', 0., 1., .8)],
        water=[_P('brine_level_m', 'brine level (above the hollow floor)', 'm', -1., 1., .08), _P('wet_fraction', 'wet fraction', '', 0., 1., .25)]),
    'grassland': dict(
        terrain=_CONTACT + [_P('relief_m', 'relief amplitude', 'm', 0., 15., 3.), _P('relief_scale_m', 'relief scale', 'm', 15., 400., 90.)],
        ecology=[_P('grass_height_m', 'grass height', 'm', .05, 1.5, .35), _P('grass_density_per_m2', 'near grass density', '/m²', 0., 30., 6.),
                 _P('bare_patch_fraction', 'local bare ground fraction', '', 0., .8, .12), _P('patch_scale_m', 'patch scale', 'm', 1., 60., 9.),
                 _P('near_radius_m', 'near instance radius', 'm', 10., 120., 45.)],
        surface=[_P('dryness', 'dryness', '', 0., 1., .25)]),
    'savanna': dict(
        terrain=_CONTACT + [_P('relief_m', 'relief amplitude', 'm', 0., 10., 1.8), _P('relief_scale_m', 'relief scale', 'm', 15., 400., 120.)],
        ecology=[_P('tree_density_per_ha', 'tree density', '/ha', 0., 200., 18.), _P('tree_height_m', 'tree height', 'm', 3., 20., 7.5),
                 _P('canopy_ratio', 'crown width / tree height', '', .6, 2.5, 1.4), _P('tree_clustering', 'community clustering', '', 0., 1., .5),
                 _P('min_spacing_m', 'minimum tree spacing', 'm', 2., 60., 9.),
                 _P('grass_height_m', 'grass height', 'm', .05, 2., .55), _P('grass_density_per_m2', 'near grass density', '/m²', 0., 30., 4.),
                 _P('near_radius_m', 'near instance radius', 'm', 10., 120., 45.)],
        surface=[_P('dryness', 'dry or wet state (1 = dry)', '', 0., 1., .7)]),
    'shrub_slope': dict(
        terrain=_CONTACT + [_P('slope_deg', 'slope', 'deg', 0., 40., 18.), _P('aspect_deg', 'slope aspect (downslope direction, clockwise from north)', 'deg', 0., 360., 180.),
                            _P('relief_m', 'slope relief', 'm', 0., 6., 1.2)],
        ecology=[_P('shrub_density_per_100m2', 'shrub density', '/100m²', 0., 60., 9.), _P('shrub_height_m', 'shrub height', 'm', .2, 3., 1.1),
                 _P('cover_patchiness', 'cover patchiness', '', 0., 1., .55), _P('boulder_density_per_100m2', 'boulder density', '/100m²', 0., 20., 1.2)],
        surface=[_P('bare_fraction', 'bare ground fraction', '', 0., 1., .3)]),
    'heathland': dict(
        terrain=_CONTACT + [_P('relief_m', 'relief amplitude', 'm', 0., 10., 2.), _P('relief_scale_m', 'relief scale', 'm', 15., 400., 110.)],
        ecology=[_P('shrub_density_per_100m2', 'low shrub and grass density', '/100m²', 0., 400., 150.), _P('shrub_height_m', 'vegetation height', 'm', .1, 1., .45),
                 _P('stone_density_per_100m2', 'gravel density', '/100m²', 0., 40., 3.)],
        surface=[_P('bare_fraction', 'bare ground / gravel fraction', '', 0., 1., .25), _P('moisture', 'surface wetness', '', 0., 1., .35)]),
}

IMPLEMENTED = set(SCHEMAS)

WEATHER_PRESETS = ('clear', 'overcast', 'rain', 'snow')
WEATHER_FIELDS = (
    _P('cloud_coverage', 'cloud cover', '', 0., 1., None), _P('cloud_base_m', 'cloud base height', 'm', 300., 6000., None),
    _P('fog_density', 'fog density', '', 0., .5, None), _P('wind_speed_mps', 'wind speed', 'm/s', 0., 25., None),
    _P('wind_azimuth_deg', 'wind direction (from)', 'deg', 0., 360., None), _P('precip_density', 'precipitation particle density', 'x', 0., 4., None),
    _P('precip_size', 'precipitation particle size', 'x', .3, 3., None), _P('surface_wetness', 'surface wetness', '', 0., 1., None),
    _P('puddles', 'standing water', '', 0., 1., None), _P('added_snow', 'new snow (thin layer)', '', 0., 1., None),
    _P('sun_lux', 'direct illuminance', 'lux', 0., 120000., None), _P('time_s', 'weather time sampling', 's', 0., 3600., None))

WEATHER_DEFAULTS = {
    'clear': dict(cloud_coverage=.15, cloud_base_m=1500., fog_density=.01, wind_speed_mps=2., wind_azimuth_deg=300.,
                  precip_density=0., precip_size=1., surface_wetness=0., puddles=0., added_snow=0., sun_lux=90000., time_s=10.),
    'overcast': dict(cloud_coverage=.9, cloud_base_m=900., fog_density=.03, wind_speed_mps=4., wind_azimuth_deg=300.,
                     precip_density=0., precip_size=1., surface_wetness=.1, puddles=0., added_snow=0., sun_lux=6000., time_s=10.),
    'rain': dict(cloud_coverage=1., cloud_base_m=600., fog_density=.06, wind_speed_mps=5., wind_azimuth_deg=300.,
                 precip_density=1., precip_size=1., surface_wetness=.85, puddles=.3, added_snow=0., sun_lux=2500., time_s=10.),
    'snow': dict(cloud_coverage=1., cloud_base_m=700., fog_density=.05, wind_speed_mps=3., wind_azimuth_deg=300.,
                 precip_density=1., precip_size=1., surface_wetness=.2, puddles=0., added_snow=.45, sun_lux=3000., time_s=10.),
}


def catalogue():
    """Front-end/CLI listing: categories, all 44 presets, availability and parameter schemas."""
    labels = dict(CATEGORIES)
    presets = []
    for category, identifier, label in PRESETS:
        row = dict(id=identifier, label=label, category=category, category_label=labels[category],
                   available=identifier in IMPLEMENTED)
        if identifier in SCHEMAS:
            row['parameters'] = deepcopy(SCHEMAS[identifier])
        else:
            row['reason'] = 'This subtype belongs to a later batch; its generation recipe is not implemented, and the name does not imply a capability.'
        presets.append(row)
    return dict(categories=[dict(id=c, label=l) for c, l in CATEGORIES], presets=presets,
                weather=dict(presets=list(WEATHER_PRESETS), fields=deepcopy(list(WEATHER_FIELDS)),
                             defaults=deepcopy(WEATHER_DEFAULTS)))


def preset_category(identifier):
    for category, key, _ in PRESETS:
        if key == identifier:
            return category
    raise ValueError('unknown environment preset: ' + str(identifier))

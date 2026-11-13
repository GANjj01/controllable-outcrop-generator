"""Global lithology catalog: stable class codes, neutral labels (group_a/b/c and numbered members; the former names are only
in data/legacy_names.json, for reading old records), display parameters and texture preference.

Classification is strictly the three rock classes of geoscience -- group_a, group_b,
group_c -- with every other grouping a subclass of one of them:
  group_a  group_a_kind_1 (group_a_06, group_a_01, group_a_03, group_a_05, group_a_02)
               group_a_kind_2 / chemical (group_a_04, group_a_07)
  group_b      setting_1 (volcanic): lava (group_b_01, group_b_02, group_b_03) and pyroclastic (group_b_04,
               volcanic breccia; IUGS treats pyroclastic rocks as group_b)
               setting_2: plutonic (group_b_06, group_b_07, group_b_08) and hypabyssal (group_b_09, group_b_10)
  group_c  group_c_kind_1 (group_c_01, group_c_02, group_c_03, group_c_04, group_c_07)
               group_c_kind_2 (group_c_05, group_c_06, group_c_08)

Codes are fixed across scenes (``lithology.png`` stores them directly), so a class id means the same
rock in every image of every batch. Append new lithologies at the end; never renumber.

Display parameters: ``recess`` is the relative differential-erosion depth (weak rock positive,
resistant negative), ``rough`` the small-scale roughness amplitude (m), ``base`` the candidate
weathered-surface tints (linear-ish RGB), ``banding`` the strength of foliation/compositional
banding drawn by the display layer (group_c rocks), ``textures`` the preferred scanned
texture sets (weights; every texture keeps a small probability for every rock, so texture alone
never identifies the class).
"""

from . import legacy as _legacy  # noqa: E402

GROUP_A, GROUP_B, GROUP_C = 'group_a', 'group_b', 'group_c'

_TEX_1 = dict(sandstone_public_a=.45, sandstone_public_b=.45, mudstone_public_a=.10)
_TEX_2 = dict(mudstone_public_a=.6, sandstone_public_b=.3, foliated_public_b=.1)
_TEX_3 = dict(sandstone_public_a=.4, marble_public_b=.3, mudstone_public_a=.3)
_TEX_4 = dict(volcanic_public_a=.35, volcanic_public_b=.35, mafic_public_a=.3)
_TEX_5 = dict(mafic_public_a=.45, volcanic_public_b=.35, volcanic_public_a=.2)
_TEX_6 = dict(granite_public_a=.7, marble_public_b=.15, sandstone_public_a=.15)
_TEX_7 = dict(foliated_public_a=.5, foliated_public_b=.5)
_TEX_8 = dict(marble_public_a=.5, marble_public_b=.5)

CATALOG = [
    # id, class, subclass, recess, rough, banding, base tints, texture preference
    ('group_a_01', GROUP_A, 'group_a_kind_1', 0., .012, 0., [(.60, .48, .34), (.72, .62, .48), (.55, .40, .30), (.66, .60, .52)], _TEX_1),
    ('group_a_02', GROUP_A, 'group_a_kind_1', 1., .006, 0., [(.30, .29, .28), (.25, .26, .28), (.36, .33, .30), (.22, .22, .22)], dict(mudstone_public_a=.6, sandstone_public_b=.3, sandstone_public_a=.1)),
    ('group_a_03', GROUP_A, 'group_a_kind_1', .6, .008, 0., [(.45, .40, .34), (.40, .38, .36), (.50, .44, .36)], dict(sandstone_public_b=.4, mudstone_public_a=.4, sandstone_public_a=.2)),
    ('group_a_04', GROUP_A, 'group_a_kind_2', -.2, .015, 0., [(.72, .70, .64), (.64, .63, .60), (.78, .74, .66)], _TEX_3),
    ('group_a_05', GROUP_A, 'group_a_kind_1', .8, .006, 0., [(.42, .32, .28), (.38, .36, .32), (.48, .40, .34)], dict(mudstone_public_a=.6, sandstone_public_b=.3, sandstone_public_a=.1)),
    ('group_a_06', GROUP_A, 'group_a_kind_1', -.1, .03, 0., [(.55, .48, .40), (.50, .46, .42), (.62, .54, .44)], dict(sandstone_public_a=.6, sandstone_public_b=.3, mudstone_public_a=.1)),
    ('group_a_07', GROUP_A, 'group_a_kind_2', -.15, .014, 0., [(.66, .60, .52), (.70, .66, .60), (.58, .55, .50)], _TEX_3),
    # group_b, setting_1 / volcaniclastic
    ('group_b_01', GROUP_B, 'group_b_kind_1', -.1, .015, 0., [(.20, .20, .21), (.25, .23, .22), (.30, .26, .22), (.18, .18, .19)], _TEX_4),
    ('group_b_02', GROUP_B, 'group_b_kind_1', -.05, .013, 0., [(.40, .38, .36), (.45, .40, .36), (.35, .35, .35)], _TEX_4),
    ('group_b_03', GROUP_B, 'group_b_kind_1', -.1, .012, .15, [(.70, .60, .55), (.75, .68, .62), (.62, .52, .48)], dict(granite_public_a=.4, volcanic_public_a=.4, marble_public_b=.2)),
    ('group_b_04', GROUP_B, 'group_b_kind_2', .7, .008, 0., [(.70, .66, .56), (.62, .58, .50), (.75, .70, .62), (.66, .55, .45)], dict(volcanic_public_a=.5, sandstone_public_b=.3, mudstone_public_a=.2)),
    ('group_b_05', GROUP_B, 'group_b_kind_2', -.05, .03, 0., [(.40, .33, .28), (.45, .38, .33), (.35, .30, .28)], dict(volcanic_public_a=.6, mafic_public_a=.4)),
    # group_b, setting_2
    ('group_b_06', GROUP_B, 'group_b_kind_3', -.2, .02, 0., [(.72, .68, .64), (.75, .66, .60), (.62, .60, .58), (.70, .58, .52)], _TEX_6),
    ('group_b_07', GROUP_B, 'group_b_kind_3', -.15, .018, 0., [(.50, .50, .48), (.45, .46, .44), (.55, .53, .50)], dict(granite_public_a=.5, mafic_public_a=.5)),
    ('group_b_08', GROUP_B, 'group_b_kind_3', -.15, .018, 0., [(.22, .23, .22), (.28, .28, .26), (.25, .25, .22)], _TEX_5),
    ('group_b_09', GROUP_B, 'group_b_kind_4', -.1, .015, 0., [(.20, .21, .21), (.26, .25, .24), (.30, .27, .24)], _TEX_5),
    ('group_b_10', GROUP_B, 'group_b_kind_4', -.25, .03, 0., [(.82, .78, .72), (.78, .70, .64), (.85, .80, .76)], _TEX_6),
    # group_c
    ('group_c_01', GROUP_C, 'group_c_kind_1', .3, .006, .35, [(.22, .24, .26), (.28, .29, .30), (.25, .24, .23)], _TEX_7),
    ('group_c_02', GROUP_C, 'group_c_kind_1', .4, .007, .45, [(.38, .40, .38), (.45, .45, .42), (.40, .38, .36)], _TEX_7),
    ('group_c_03', GROUP_C, 'group_c_kind_1', .2, .012, .6, [(.42, .44, .40), (.50, .48, .44), (.45, .40, .36)], _TEX_7),
    ('group_c_04', GROUP_C, 'group_c_kind_1', -.1, .016, .9, [(.55, .53, .50), (.48, .46, .44), (.60, .56, .52)], dict(foliated_public_b=.4, granite_public_a=.4, foliated_public_a=.2)),
    ('group_c_05', GROUP_C, 'group_c_kind_2', -.3, .014, .1, [(.80, .78, .74), (.75, .70, .66), (.72, .70, .70)], dict(marble_public_b=.4, sandstone_public_a=.3, granite_public_a=.3)),
    ('group_c_06', GROUP_C, 'group_c_kind_2', .1, .012, .3, [(.85, .84, .82), (.78, .78, .76), (.70, .70, .72)], _TEX_8),
    ('group_c_07', GROUP_C, 'group_c_kind_1', -.1, .015, .5, [(.20, .24, .22), (.25, .27, .24), (.22, .22, .20)], dict(foliated_public_a=.4, mafic_public_a=.4, foliated_public_b=.2)),
    ('group_c_08', GROUP_C, 'group_c_kind_2', -.2, .012, .05, [(.25, .25, .27), (.32, .30, .30), (.28, .27, .26)], dict(mafic_public_a=.4, foliated_public_b=.4, volcanic_public_b=.2)),
]

LITHOLOGY = {name: dict(code=i + 1, family=fam, subclass=sub, recess=rec, rough=rough, banding=band, base=base, textures=tex)
             for i, (name, fam, sub, rec, rough, band, base, tex) in enumerate(CATALOG)}
# The group_b setting of each kind (the former subclass values were 'setting/kind' paths; the labels are flat).
SETTING_OF_KIND = {'group_b_kind_1': 'setting_1', 'group_b_kind_2': 'setting_1', 'group_b_kind_3': 'setting_2', 'group_b_kind_4': 'setting_2'}

# values are the rows' places in the order the tables had before the labels, so every sorted list keeps that order.
ORDER = {'group_a_01': 18, 'group_a_02': 20, 'group_a_03': 21, 'group_a_04': 11, 'group_a_05': 13, 'group_a_06': 3, 'group_a_07': 6, 'group_b_01': 2, 'group_b_02': 1, 'group_b_03': 17, 'group_b_04': 23, 'group_b_05': 24, 'group_b_06': 9, 'group_b_07': 5, 'group_b_08': 7, 'group_b_09': 4, 'group_b_10': 14, 'group_c_01': 22, 'group_c_02': 15, 'group_c_03': 19, 'group_c_04': 8, 'group_c_05': 16, 'group_c_06': 12, 'group_c_07': 0, 'group_c_08': 10}

DEFAULT = dict(code=0, family=None, subclass=None, recess=0., rough=.01, banding=0., base=[(.5, .48, .45)], textures={})
FAMILIES = (GROUP_A, GROUP_B, GROUP_C)
FAMILY_CODES = {GROUP_A: 1, GROUP_B: 2, GROUP_C: 3}

LABELS_ZH = dict(group_a_01='Group A 01', group_a_02='Group A 02', group_a_03='Group A 03', group_a_04='Group A 04',
    group_a_05='Group A 05', group_a_06='Group A 06', group_a_07='Group A 07', group_b_01='Group B 01',
    group_b_02='Group B 02', group_b_03='Group B 03', group_b_04='Group B 04', group_b_05='Group B 05',
    group_b_06='Group B 06', group_b_07='Group B 07', group_b_08='Group B 08', group_b_09='Group B 09',
    group_b_10='Group B 10', group_c_01='Group C 01', group_c_02='Group C 02', group_c_03='Group C 03', group_c_04='Group C 04',
    group_c_05='Group C 05', group_c_06='Group C 06', group_c_07='Group C 07', group_c_08='Group C 08')


def display_catalog():
    """Expose the same native catalogue to local parameter editors."""
    return [dict(id=name, label=LABELS_ZH.get(name, name), family=item['family'],
                 subclass=item['subclass']) for name, item in LITHOLOGY.items()]


def _former_row(row):
    return dict(row, family=_legacy.former(row['family']), subclass=_legacy.former(row['subclass']))


def order_key(name):
    # where a lithology name sorts among others (codebooks, display tints): catalogue rows by the order field, then any
    # other name (a former name in an old record, a non-catalogue name) by its text, as every such list sorted before
    return (0, ORDER[name]) if name in ORDER else (1, str(name))


def get(name):
    label = _legacy.name(name)
    if label != name and label in LITHOLOGY:
        return _former_row(LITHOLOGY[label])   # a former name reads its row in the former vocabulary
    return LITHOLOGY.get(name, DEFAULT)


def lithology_records(names):
    """v2 ``lithologies`` table entries with the global codes and classification. Labels give label records; former names
    give records entirely in the former vocabulary; a mix is refused."""
    names = sorted(set(names), key=lambda n: ORDER[_legacy.name(n)])   # the catalogue's order field
    old = [n for n in names if _legacy.name(n) != n]
    if old and len(old) != len(names):
        raise ValueError('lithology_records: former names mixed with labels: %s' % names)
    rows = {n: _former_row(LITHOLOGY[_legacy.name(n)]) if old else LITHOLOGY[n] for n in names}
    return [{'id': n, 'code': rows[n]['code'], 'rock_family': rows[n]['family'],
             'properties': {'subclass': rows[n]['subclass']}} for n in names]

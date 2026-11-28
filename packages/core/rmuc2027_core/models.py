"""Public hardware catalog and immutable, mirrored team lineups for each rules season."""
import json
from pathlib import Path

from . import seasons as season_rules

# Canonical RMUC 2027 roster.
SLOTS=(1,3,4,6,7)
ROLES={1:'heavy',3:'infantry',4:'infantry',6:'aerial',7:'sentry'}
DEFAULT_TEAM_LINEUP={'1':2711,'3':2731,'4':2732,'6':2761,'7':2771}
from .paths import ROOT

def robot_models():
    return json.loads((ROOT/'configs/rules.json').read_text(encoding="utf-8"))['robot_designs']

def model_ids():
    return tuple(sorted(int(k) for k in robot_models()))

def default_team_lineup(season=season_rules.DEFAULT_SEASON):
    return season_rules.default_lineup(season)

def initial_model_stats(design,rules,season=None):
    """Level-one preselected referee performance, using the engine's public tables."""
    role=design['role']
    season=season or design.get('season',season_rules.DEFAULT_SEASON)
    if season=='2027':
        rulebook=rules[rules['seasons']['2027']['rules_key']];specs=rules['seasons']['2027']['specs']
        stats=dict(specs[role])
        if role=='heavy':stats.update(rulebook['heavy']['stages'][0])
        elif role=='infantry':
            defaults = rulebook['profiles']['defaults']
            stats.update(rulebook['profiles']['infantry_chassis'][defaults['infantry_chassis']][0])
            stats.update(rulebook['profiles']['infantry_weapon'][defaults['infantry_weapon']][0])
        elif role=='sentry':stats.update(rulebook['sentry']['automatic']);stats['ammo']=rulebook['initial_ammo']['sentry']
        stats.setdefault('max_heat',0);stats.setdefault('cooling',0);stats.setdefault('power_w',0)
        return {key:stats.get(key,0) for key in ('hp','power_w','max_heat','cooling','damage','ammo')}
    raise ValueError('Only RMUC 2027 is supported')

def model_catalog():
    rules=json.loads((ROOT/'configs/rules.json').read_text(encoding="utf-8"))
    catalog={}
    for key,d in rules['robot_designs'].items():
        legal=[s for s,v in d.get('seasons',{}).items() if v.get('legal')]
        catalog[key]={**d,'initial_stats':initial_model_stats(d,rules),
            'initial_stats_by_season':{s:initial_model_stats(d,rules,s) for s in legal}}
    return catalog

def normalize_team_lineup(lineup=None,season=None):
    """Validate a lineup for its season; without a season the slots decide (2027 only)."""
    if season is None:
        season=season_rules.infer_season(lineup.keys()) if lineup else season_rules.DEFAULT_SEASON
    slots=season_rules.slots(season);roles=season_rules.roles(season)
    value=dict(season_rules.default_lineup(season) if lineup is None else lineup)
    value={str(k):v for k,v in value.items()}
    if set(value)!={str(s) for s in slots}:
        raise ValueError(f"Team lineup must select exactly slots {','.join(map(str,slots))}")
    models=robot_models()
    for slot,model in value.items():
        if isinstance(model,bool) or not isinstance(model,int) or str(model) not in models:
            raise ValueError(f'Unknown robot model for slot {slot}')
        if models[str(model)]['role']!=roles[int(slot)]:
            raise ValueError(f'Slot {slot} requires a {roles[int(slot)]} model')
        if not season_rules.legal(model,season):
            raise ValueError(f'Model {model} does not pass the {season} construction rules')
    return {str(s):value[str(s)] for s in slots}

def lineup_season(lineup):
    return season_rules.infer_season(lineup.keys())

def mirrored_lineups(lineup,season=None):
    value=normalize_team_lineup(lineup,season)
    return {team:[{'slot':int(s),'model':value[s]} for s in value] for team in ('red','blue')}


OFFICIAL_ROLE_CAPS_2027 = {'heavy': 1, 'infantry': 2, 'sentry': 1, 'aerial': 1}
OFFICIAL_MIN_GROUND_2027 = 3
OFFICIAL_MAX_TEAM_SIZE_2027 = 5


def validate_official_lineups(lineups, season=season_rules.DEFAULT_SEASON):
    """Validate RMUC 2027 Table 3-1 & R35 roster composition for official matches.

    Each team must field at least 3 ground robots, at most 5 robots total,
    with per-role caps: heavy <= 1, infantry <= 2, sentry <= 1, aerial <= 1,
    unique positive slot IDs, and models legal for the target season.
    """
    if not isinstance(lineups, dict) or set(lineups) != {'red', 'blue'}:
        raise ValueError('Lineups require red and blue')
    models = robot_models()
    for team in ('red', 'blue'):
        entries = lineups[team]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f'{team} lineup must be a non-empty list')
        if len(entries) > OFFICIAL_MAX_TEAM_SIZE_2027:
            raise ValueError(f'{team} lineup exceeds RMUC 2027 max {OFFICIAL_MAX_TEAM_SIZE_2027} entities')
        seen_slots = set()
        counts = {'heavy': 0, 'infantry': 0, 'sentry': 0, 'aerial': 0}
        ground = 0
        for item in entries:
            if not isinstance(item, dict) or 'slot' not in item or 'model' not in item:
                raise ValueError(f'{team} lineup entries require slot and model')
            slot, model = item['slot'], item['model']
            if isinstance(slot, bool) or not isinstance(slot, int) or slot <= 0 or slot in seen_slots:
                raise ValueError(f'{team} lineup requires unique positive integer slots')
            seen_slots.add(slot)
            if isinstance(model, bool) or not isinstance(model, int) or str(model) not in models:
                raise ValueError(f'Unknown robot model {model} for {team} slot {slot}')
            if not season_rules.legal(model, season):
                raise ValueError(f'Model {model} does not pass the {season} construction rules')
            role = models[str(model)]['role']
            if role not in OFFICIAL_ROLE_CAPS_2027:
                raise ValueError(f'Unsupported role {role} in RMUC {season}')
            counts[role] += 1
            if counts[role] > OFFICIAL_ROLE_CAPS_2027[role]:
                raise ValueError(
                    f'{team} lineup exceeds RMUC 2027 cap for {role} (max {OFFICIAL_ROLE_CAPS_2027[role]})'
                )
            if role != 'aerial':
                ground += 1
        if ground < OFFICIAL_MIN_GROUND_2027:
            raise ValueError(
                f'{team} lineup violates R35: requires at least {OFFICIAL_MIN_GROUND_2027} ground robots (got {ground})'
            )
    return lineups

"""Host-side operator presentation; eligibility is checked again by the engine."""
import json
import math
from functools import lru_cache
from rmuc2027_core.paths import ROOT


@lru_cache(maxsize=1)
def rules():
    return json.loads((ROOT / 'configs/rules.json').read_text(encoding='utf-8'))['rulebook']


def actions(frame, robot):
    book = rules()
    now, team, role = frame['time'], robot['team'], robot['role']
    economy = frame['teams'][team]
    alive = robot['alive']
    locked = ('比赛已结束' if frame['status'] == 'finished' else
              '判罚遮挡中' if robot.get('fpv_blocked_until', 0) > now else
              '机器人已被罚下' if robot.get('red_card') else '')
    result = []

    def add(name, label, value='', reason='', cost=None, detail=''):
        reason = locked or reason
        if cost is not None and economy['resources'] < cost and not reason:
            reason = '队伍经济不足'
        result.append(dict(name=name, value=value, label=label, enabled=not reason,
                           reason=reason, cost=cost, detail=detail))

    if role == 'aerial':
        for value, label in [('outpost', '前哨站'), ('base', '基地')]:
            add('air_target', '目标：敌方' + label, value)
        add('air_launch', '起飞并攻击', reason='' if any(d['state'] != 'disabled' for d in robot.get('drones', [])) else '无可用无人机')
        add('air_return', '召回无人机')
        return result
    dead = '' if alive else '机器人战亡'
    disengaged = alive and now - robot.get('last_shot_time', now) >= 6 - 1e-8 and now - robot.get('last_damage_time', now) >= 6 - 1e-8
    contacts = set(robot.get('last_raw_zones', [])) | {z for z, t in robot.get('zone_contacts', {}).items() if now - t <= book['buffs']['release_delay_s'] + 1e-8}
    local = any(f'{team}_{kind}' in contacts for kind in ('base', 'supply', 'high_point'))
    for owner in ('red', 'blue'):
        position = frame.get('positions', {}).get(f'{owner}_outpost', {})
        local |= f'{owner}_outpost' in contacts and position.get('controller') == team and position.get('control') == 'full'
    local &= not robot.get('weak', False)
    kind = '42mm' if role == 'heavy' else '17mm'
    ammo = book['economy']['ammo'][kind]
    for remote in (False, True):
        prefix = 'remote' if remote else 'local'
        reason = dead or ('' if (disengaged if remote else local) else '需脱战 6 秒' if remote else '需进入可用补给区域且解除虚弱')
        purchased = economy.get('purchased_42' if role == 'heavy' else 'purchased_17', 0)
        if purchased + ammo[prefix + '_batch'] > ammo['team_purchase_cap']:
            reason = reason or '队伍弹药购买额度已用尽'
        add('ammo', ('远程' if remote else '区域') + f"补弹 +{ammo[prefix + '_batch']}", prefix, reason, ammo[prefix + '_cost'], '6 秒后送达' if remote else '立即补给')
    add('heal', '远程回血', reason=dead or ('回血已在配送中' if robot.get('pending_heal') is not None else '' if disengaged else '需脱战 6 秒'), cost=50 + math.ceil(now / 60 * 20), detail='6 秒后生效，恢复量受后勤影响')
    add('revive', '购买立即复活', reason='机器人仍存活' if alive else '', cost=math.ceil(now / 60) * 80 + robot.get('level', 1) * 20)
    if role == 'heavy':
        add('pick', '拾取能量单元', reason=dead or ('能量单元已耗尽' if economy.get('energy_units_left', 0) <= 0 else '已携带两个单元' if robot.get('carried_units', 0) >= 2 else ''), detail='辅助移动，手动移动可取消')
        for level in range(1, 5):
            complete = economy.get('completed_difficulties', [])
            reason = '本级已完成' if level in complete else '需先完成前一级' if level > 1 and level - 1 not in complete else ''
            add('assemble', f'{level} 级装配辅助', str(level), dead or reason, detail='自动拾取并前往装配区')
        add('install', '安装发射机构', reason=dead or ('' if robot.get('launcher') in ('unlocked', 'installing') else '发射机构未解锁或已安装'))
        add('cancel', '取消辅助')
    if role == 'sentry':
        for value, label in [('attack', '进攻'), ('defense', '防御'), ('move', '移动')]:
            cooldown = max(0, robot.get('posture_ready_at', 0) - now)
            add('posture', label + '姿态', value, dead or ('已处于该姿态' if robot.get('posture') == value else f'姿态切换冷却 {cooldown:.1f} 秒' if cooldown > 1e-8 else ''))
    return result

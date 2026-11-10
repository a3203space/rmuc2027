"""Read-only RMUC 2026 telemetry ETL, never a fabricated action dataset.

CLI: python -m rmuc2027_research.rl.offline2026 DATABASE --output DIRECTORY
No optimizer/model fitting is performed. All unknown original actions are masked.
2027 semantic targets stay masked until a landmark/region mapping is confirmed.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sqlite3

from .scenarios import canonical_hash

VERSION = 1
ROLES = ("英雄", "工程", "步兵3", "步兵4")
HORIZONS = (1, 3, 5)
STATE_FIELDS = ("x", "y", "z", "muzzle_angle_reported", "chassis_power_w", "hp", "max_hp",
                "heat17", "limit17", "heat42", "limit42", "shots17", "shots42", "total_coins", "coins", "radar_marked")
SOURCE_FIELDS = ("x", "y", "z", "枪口朝向", "底盘功率", "当前血量", "最大血量", "小热量", "小热量上限",
                 "大热量", "大热量上限", "累计17mm发弹", "累计42mm发弹", "队伍总金币", "队伍剩余金币", "是否易伤")
UNKNOWN_ACTION_FIELDS = ("move_x", "move_y", "chassis_yaw", "aim_yaw", "aim_pitch", "fire",
                         "power_request", "fold", "jump", "assembly")
SCHEMA = dict(version=VERSION, state_fields=STATE_FIELDS, original_action_fields=UNKNOWN_ACTION_FIELDS,
              horizons_s=HORIZONS, coordinate_unit="m_user_confirmed", muzzle_angle_unit="unconfirmed",
              original_actions_observed=False, native_rewards_observed=False, visibility_observed=False,
              default_2027_semantic_targets_valid=False)
SCHEMA_HASH = canonical_hash(SCHEMA)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def readonly_database(path):
    """SQLite URI mode=ro plus query_only; temporary query work stays in memory."""
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    try:
        yield connection
    finally:
        connection.close()


def inspect_database(connection):
    tables = [dict(row) for row in connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE type IN ('table','index','view') ORDER BY type,name")]
    required = {"matches", "timeseries", "events"}
    if not required <= {row["name"] for row in tables if row["type"] == "table"}:
        raise ValueError("The source is not the documented RMUC 2026 dataset")
    columns = {name: [dict(row) for row in connection.execute(f'PRAGMA table_info("{name}")')]
               for name in sorted(required)}
    wanted = {"matches": {"game_id", "赛区", "场次号", "时长秒", "胜方"},
              "timeseries": {"game_id", "robot_id", "时刻秒", "机器人类型", "阵营", *SOURCE_FIELDS},
              "events": {"game_id", "时刻秒", "事件类型", "阵营", "数值"}}
    for name, fields in wanted.items():
        if not fields <= {row["name"] for row in columns[name]}:
            raise ValueError(f"Missing required source fields in {name}")
    return dict(objects=tables, columns=columns, schema_hash=canonical_hash(dict(objects=tables, columns=columns)))


def _allocation(total, capacities):
    if total < 0 or total > sum(capacities.values()):
        raise ValueError("Requested split exceeds available series")
    denom = sum(capacities.values())
    if not denom:
        return dict.fromkeys(capacities, 0)
    exact = {key: total * value / denom for key, value in capacities.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda k: (-(exact[k] - result[k]), k))[:total - sum(result.values())]:
        result[key] += 1
    return result


def split_series(matches, seed=20261110, counts=(186, 40, 40)):
    """All rounds, sides, robots and derived windows in a series stay together."""
    if len(counts) != 3 or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("Split counts must be three nonnegative integers")
    groups = defaultdict(list)
    game_ids = set()
    for row in matches:
        game = int(row["game_id"])
        if game in game_ids:
            raise ValueError("Duplicate matches.game_id")
        game_ids.add(game); groups[(row["赛区"], int(row["场次号"]))].append(game)
    if sum(counts) != len(groups):
        raise ValueError(f"Expected {sum(counts)} series, found {len(groups)}; specify an explicit split")
    by_region = defaultdict(list)
    for key in groups:
        by_region[key[0]].append(key)
    for region in by_region:
        by_region[region].sort(key=lambda key: canonical_hash([seed, "series", *key]))
    sizes = {key: len(value) for key, value in by_region.items()}
    train = _allocation(counts[0], sizes)
    rest = {key: sizes[key] - train[key] for key in sizes}
    val = _allocation(counts[1], rest)
    rows, game_split = [], {}
    for region in sorted(by_region):
        for i, key in enumerate(by_region[region]):
            split = "train" if i < train[region] else "validation" if i < train[region] + val[region] else "test"
            games = sorted(groups[key])
            rows.append(dict(region=region, match_number=key[1], split=split, game_ids=games))
            for game in games:
                game_split[str(game)] = split
    body = dict(version=VERSION, seed=int(seed), grouping=["赛区", "场次号"], counts=dict(zip(("train", "validation", "test"), counts)),
                series=rows, game_split=game_split)
    return {**body, "split_hash": canonical_hash(body)}


def _inside(point, polygon):
    x, y = point; inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def validate_calibration(value=None):
    """A mapping must be supplied explicitly; identity is not silently confirmed."""
    value = dict(value or {})
    value.setdefault("confirmed", False)
    if type(value["confirmed"]) is not bool:
        raise ValueError("Calibration confirmation must be boolean")
    if value["confirmed"]:
        import numpy as np
        matrix = np.asarray(value.get("source_to_field"), dtype=float)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
            raise ValueError("Confirmed calibration requires a nonsingular 2x3 affine matrix")
        if len(value.get("landmarks", [])) < 3 or not value.get("evidence"):
            raise ValueError("Confirmed mapping requires at least three landmarks and evidence")
        source = np.asarray([p["source_xy"] for p in value["landmarks"]], float)
        target = np.asarray([p["field_xy"] for p in value["landmarks"]], float)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("Invalid calibration landmarks")
        if np.linalg.matrix_rank(np.column_stack((source, np.ones(len(source))))) < 3:
            raise ValueError("Calibration landmarks must not be collinear")
        error = np.linalg.norm(np.column_stack((source, np.ones(len(source)))) @ matrix.T - target, axis=1)
        tolerance = float(value.get("max_landmark_error_m", .25))
        if not math.isfinite(tolerance) or tolerance <= 0 or max(error) > tolerance:
            raise ValueError("Calibration does not reproduce its landmarks")
        for region in value.get("semantic_regions", []):
            polygon = region.get("field_polygon", [])
            goal = region.get("target_2027_xy", [])
            if len(polygon) < 3 or len(goal) != 2 or not region.get("name") or not region.get("target_verified"):
                raise ValueError("Semantic regions require verified 2027 goals")
            if not np.isfinite(np.asarray(polygon, float)).all() or not np.isfinite(np.asarray(goal, float)).all():
                raise ValueError("Nonfinite semantic mapping")
            if not (0 <= goal[0] <= 28 and 0 <= goal[1] <= 15):
                raise ValueError("Semantic target lies outside the 2027 field")
    return value


def iter_tracks(connection):
    fields = ",".join('"' + name + '"' for name in SOURCE_FIELDS)
    query = f'SELECT game_id,robot_id,赛区,机器人类型,阵营,时刻秒,{fields} FROM timeseries WHERE 机器人类型 IN (?,?,?,?) ORDER BY game_id,robot_id,时刻秒'
    for key, rows in itertools.groupby(connection.execute(query, ROLES), lambda r: (r["game_id"], r["robot_id"])):
        yield key, list(rows)


def clean_track(rows, calibration=None, first_assembly=None):
    """Return telemetry and masked route labels. Original action targets are absent."""
    import numpy as np
    calibration = validate_calibration(calibration)
    times = np.asarray([r["时刻秒"] for r in rows], dtype=np.float64)
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory keys must be unique and ordered")
    states = np.asarray([[r[field] if r[field] is not None else np.nan for field in SOURCE_FIELDS] for r in rows], dtype=np.float64)
    valid = np.isfinite(states)
    xy = states[:, :2]
    sentinel = np.all(xy == 0, axis=1) | np.all(xy == -1, axis=1)
    position_valid = valid[:, :2].all(axis=1) & ~sentinel
    alive = valid[:, 5] & (states[:, 5] > 0)
    discontinuity = np.zeros(len(rows), dtype=bool)
    # Isolated leave-and-return localisation spikes, without a speed ceiling.
    displacement = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    for i in range(1, len(rows)-1):
        if times[i]-times[i-1] != 1 or times[i+1]-times[i] != 1 or not position_valid[i-1:i+2].all():
            continue
        neighbors = np.concatenate((displacement[max(0, i-5):i-1], displacement[i+1:min(len(displacement), i+5)]))
        neighbors = neighbors[np.isfinite(neighbors)]
        scale = float(np.median(neighbors)) if len(neighbors) else 0.
        residual = np.linalg.norm(xy[i] - (xy[i-1] + xy[i+1])/2)
        if residual > max(1., 6 * scale) and np.dot(xy[i]-xy[i-1], xy[i+1]-xy[i]) < 0:
            discontinuity[i] = True
    position_valid &= ~discontinuity
    field_xy = np.zeros_like(xy)
    field_valid = np.zeros(len(rows), dtype=bool)
    if calibration["confirmed"]:
        matrix = np.asarray(calibration["source_to_field"], float)
        field_xy = np.column_stack((xy, np.ones(len(rows)))) @ matrix.T
        field_valid = position_valid & (field_xy[:, 0] >= 0) & (field_xy[:, 0] <= 28) & (field_xy[:, 1] >= 0) & (field_xy[:, 1] <= 15)
    valid[:, :2] &= position_valid[:, None]
    if rows[0]["机器人类型"] == "工程":
        valid[:, 3] = False  # Per-region constant in source, not a measured orientation.
    raw_goals = np.zeros((len(rows), len(HORIZONS), 2), dtype=np.float32)
    raw_masks = np.zeros((len(rows), len(HORIZONS)), dtype=bool)
    semantic_goals = np.zeros_like(raw_goals)
    semantic_masks = np.zeros_like(raw_masks)
    semantic_class = np.full(raw_masks.shape, -1, dtype=np.int16)
    lookup = {float(t): i for i, t in enumerate(times)}
    engineer = rows[0]["机器人类型"] == "工程"
    prefix = np.ones(len(rows), dtype=bool) if not engineer else (
        times <= first_assembly if first_assembly is not None else np.zeros(len(rows), dtype=bool))
    for i, t in enumerate(times):
        for column, horizon in enumerate(HORIZONS):
            j = lookup.get(float(t+horizon))
            if j is None or j-i != horizon or not prefix[i:j+1].all() or not (position_valid[i:j+1] & alive[i:j+1]).all():
                continue
            raw_goals[i, column] = xy[j]; raw_masks[i, column] = True
            if calibration["confirmed"] and field_valid[i:j+1].all():
                for index, region in enumerate(calibration.get("semantic_regions", [])):
                    if _inside(field_xy[j], region["field_polygon"]):
                        semantic_goals[i, column] = region["target_2027_xy"]
                        semantic_masks[i, column] = True; semantic_class[i, column] = index
                        break
    states = np.where(valid, states, 0).astype(np.float32)
    return dict(time_s=times.astype(np.float32), state=states, state_valid=valid,
                source_xy=np.where(np.isfinite(xy), xy, 0).astype(np.float32), source_position_valid=position_valid,
                field_xy=np.where(np.isfinite(field_xy), field_xy, 0).astype(np.float32), field_position_valid=field_valid,
                alive=alive, discontinuity=discontinuity, engineer_prefix=prefix,
                future_source_xy=raw_goals, future_source_mask=raw_masks,
                semantic_goal_xy=semantic_goals, semantic_goal_mask=semantic_masks, semantic_class=semantic_class,
                original_action_mask=np.zeros((len(rows), len(UNKNOWN_ACTION_FIELDS)), dtype=bool))


def _summary(values):
    import numpy as np
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0}
    q = np.quantile(values, [.5, .95, .99, .999])
    return dict(n=len(values), median=float(q[0]), p95=float(q[1]), p99=float(q[2]), p999=float(q[3]), max=float(values.max()))


def track_speed_statistics(rows):
    """Audit raw speed separately from an explicitly conditional field rectangle.

    There is no velocity cut-off. These are 1 Hz chord speeds, not hardware limits.
    Returns aggregate buffers plus one track-level observation envelope.
    """
    import numpy as np
    a = np.asarray([[r["时刻秒"], r["x"], r["y"], r["当前血量"], r["底盘功率"]] for r in rows], float)
    dt = np.diff(a[:, 0]); delta = np.diff(a[:, 1:3], axis=0)
    velocity = np.linalg.norm(delta, axis=1) / dt
    point_ok = np.isfinite(a[:, 1:4]).all(axis=1) & np.any(a[:, 1:3] != 0, axis=1) & np.any(a[:, 1:3] != -1, axis=1) & (a[:, 3] > 0)
    raw = (dt == 1) & np.isfinite(velocity)
    base = raw & point_ok[:-1] & point_ok[1:]
    bounds = (a[:, 1] >= 0) & (a[:, 1] <= 28) & (a[:, 2] >= 0) & (a[:, 2] <= 15)
    conditional = base & bounds[:-1] & bounds[1:]
    moving = conditional & (velocity > .2)
    # Six-point continuity diagnostic, never a blanket filter on training speeds.
    coherent = np.zeros(len(velocity), dtype=bool)
    for i in range(2, len(velocity)-2):
        if not conditional[i-2:i+3].all():
            continue
        speeds = velocity[i-1:i+2]
        if min(speeds) < .75 * max(speeds):
            continue
        vectors = delta[i-2:i+3]; speeds5 = velocity[i-2:i+3]
        cos = np.sum(vectors[:-1]*vectors[1:], axis=1) / np.maximum(speeds5[:-1]*speeds5[1:], 1e-12)
        coherent[i] = bool(np.all(cos[1:3] >= math.cos(math.pi/4)) and
                           (speeds5[0] <= .2 or cos[0] >= 0) and (speeds5[4] <= .2 or cos[3] >= 0))
    powers = (a[:-1, 4]+a[1:, 4])/2
    envelope = dict(game_id=int(rows[0]["game_id"]), robot_id=int(rows[0]["robot_id"]),
                    region=rows[0]["赛区"], role=rows[0]["机器人类型"], moving=_summary(velocity[moving]))
    return dict(raw=velocity[raw], base=velocity[base], conditional=velocity[conditional], moving=velocity[moving],
                coherent=velocity[coherent], power=np.column_stack((velocity[conditional], powers[conditional]))), envelope


def prepare(database, output, *, seed=20261110, split_counts=(186, 40, 40), calibration=None):
    """Write an immutable preparation directory; never fit or initialize weights."""
    import numpy as np
    source, output = Path(database), Path(output)
    calibration = validate_calibration(calibration)
    source_sha = file_hash(source)
    output.mkdir(parents=True, exist_ok=False)
    buffers = {name: defaultdict(list) for name in ("train", "validation", "test")}
    speed_buffers = {name: defaultdict(list) for name in buffers}
    envelopes = defaultdict(list); track_rows = []; audits = defaultdict(int)
    with readonly_database(source) as connection:
        source_schema = inspect_database(connection)
        matches = [dict(row) for row in connection.execute("SELECT * FROM matches ORDER BY 赛区,场次号,局号")]
        splits = split_series(matches, seed, split_counts)
        match_by_id = {int(row["game_id"]): row for row in matches}
        endings = dict(connection.execute("SELECT game_id,max(时刻秒) FROM timeseries GROUP BY game_id"))
        assembly = {(int(row[0]), row[1]): float(row[2]) for row in connection.execute(
            "SELECT game_id,阵营,min(时刻秒) FROM events WHERE 事件类型='装配成功' GROUP BY game_id,阵营")}
        for (game, robot), rows in iter_tracks(connection):
            if str(game) not in splits["game_split"]:
                raise ValueError("Orphan trajectory game_id")
            split = splits["game_split"][str(game)]; side = rows[0]["阵营"]; role = rows[0]["机器人类型"]
            cleaned = clean_track(rows, calibration, assembly.get((int(game), side)))
            n = len(rows); offset = sum(len(a) for a in buffers[split]["time_s"])
            cleaned.update(game_id=np.full(n, game, np.int64), robot_id=np.full(n, robot, np.int16),
                           role_id=np.full(n, ROLES.index(role), np.int8), side_id=np.full(n, side == "蓝", np.int8),
                           round_boundary=cleaned["time_s"] == endings[game])
            # Historical outcome is a label, never an observation or 2027 reward.
            winner = match_by_id[game]["胜方"]
            cleaned["historical_outcome"] = np.full(n, 1 if winner == side else -1 if winner in ("红", "蓝") else 0, np.int8)
            for name, value in cleaned.items():
                buffers[split][name].append(value)
            track_rows.append(dict(game_id=int(game), robot_id=int(robot), split=split, role=role, side=side,
                                   offset=offset, rows=n, first_assembly_s=assembly.get((int(game), side))))
            audits["tracks"] += 1; audits["rows"] += n
            audits["source_invalid_positions"] += int((~cleaned["source_position_valid"]).sum())
            audits["isolated_discontinuities"] += int(cleaned["discontinuity"].sum())
            audits["raw_future_labels"] += int(cleaned["future_source_mask"].sum())
            audits["semantic_future_labels"] += int(cleaned["semantic_goal_mask"].sum())
            if role in ("步兵3", "步兵4"):
                speeds, envelope = track_speed_statistics(rows)
                envelopes[split].append(envelope)
                for key, value in speeds.items():
                    speed_buffers[split][key].append(value)
    files = []
    for split, fields in buffers.items():
        if not fields:
            continue
        path = output / (split + "-trajectories.npz")
        np.savez_compressed(path, **{key: np.concatenate(value) for key, value in fields.items()})
        files.append(dict(path=path.name, sha256=file_hash(path), bytes=path.stat().st_size, split=split))
    speeds_report = {}
    for split, fields in speed_buffers.items():
        if not fields:
            speeds_report[split] = {}; continue
        values = {key: np.concatenate(parts) for key, parts in fields.items()}
        report = {key: _summary(value) for key, value in values.items() if key != "power"}
        report["power_bins"] = []
        for low, high in ((0, 50), (50, 80), (80, 100), (100, 120), (120, 150), (150, 200), (200, math.inf)):
            mask = (values["power"][:, 1] >= low) & (values["power"][:, 1] < high)
            report["power_bins"].append(dict(low_w=low, high_w=None if not math.isfinite(high) else high,
                                              speed=_summary(values["power"][mask, 0])))
        # Each robot-game contributes one envelope; each game then has equal weight.
        by_game = defaultdict(list)
        for row in envelopes[split]:
            if row["moving"]["n"] >= 20:
                by_game[row["game_id"]].append(row["moving"]["p95"])
        report["game_balanced_moving_p95"] = _summary([sum(v)/len(v) for v in by_game.values()])
        speeds_report[split] = report
    description = source.parent / "RMUC 2026 区域赛数据集使用说明.md"
    source_info = dict(filename=source.name, bytes=source.stat().st_size, sha256=source_sha, schema=source_schema,
                       description_sha256=file_hash(description) if description.exists() else None)
    report = dict(version=VERSION, schema=SCHEMA, schema_hash=SCHEMA_HASH, source=source_info,
                  split_hash=splits["split_hash"], calibration=calibration, calibration_hash=canonical_hash(calibration),
                  counts=dict(audits), velocity_definition="1-second planar chord distance; not instantaneous maximum speed",
                  conditional_bounds_assumption=[0, 28, 0, 15], moving_threshold_mps=.2,
                  speed_statistics=speeds_report,
                  calibration_targets=dict(split="train", ready=bool(calibration["confirmed"]),
                      reason=None if calibration["confirmed"] else "Coordinate landmarks/axes unconfirmed; telemetry-only preparation",
                      robot_game_envelopes=envelopes["train"]),
                  original_actions_observed=False, gradient_updates=0,
                  warnings=["No action, player-visibility, chassis-model or power-limit fields",
                            "Shots events and cumulative counters disagree at second boundaries",
                            "Assembly success is a weak prefix boundary, not an observed arrival/action",
                            "Historical winner is not the 2027 control-degree reward"], tracks=track_rows, files=files)
    etl_source = Path(__file__).read_bytes()
    (output/"etl-source.py").write_bytes(etl_source)
    report["etl_source"] = dict(path="etl-source.py", sha256=hashlib.sha256(etl_source).hexdigest(), bytes=len(etl_source))
    report["etl_runtime"] = dict(python=platform.python_version(), numpy=np.__version__)
    report["cleaning"] = dict(sentinel_xy=[[0, 0], [-1, -1]], isolated_spike_residual_m=1.,
                              isolated_spike_local_median_multiplier=6., velocity_cap_applied=False,
                              semantic_labels_require_confirmed_landmarks=True)
    if file_hash(source) != source_sha:
        raise RuntimeError("Source database changed while preparing the dataset")
    report["source_unchanged"] = True
    for name, value in (("split.json", splits), ("manifest.json", report)):
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--split-counts", nargs=3, type=int, default=(186, 40, 40))
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    result = prepare(args.database, args.output, seed=args.seed, split_counts=tuple(args.split_counts), calibration=calibration)
    print(json.dumps(dict(output=str(args.output), counts=result["counts"], source_unchanged=result["source_unchanged"],
                         gradient_updates=0, semantic_mapping_confirmed=result["calibration"]["confirmed"]), ensure_ascii=False))


if __name__ == "__main__":
    main()

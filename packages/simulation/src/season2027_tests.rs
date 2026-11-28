//! RMUC 2027 rules: lineups, inspection, economy, control, structures, aerial system,
//! heavy launcher, sentry postures, crossings and the timeout ranking.
use super::*;

fn match_() -> Engine {
    Engine::new(42, 420.0, "rmuc2027").unwrap()
}
fn robot(m: &Engine, id: &str) -> usize {
    m.robots.iter().position(|r| r.id == id).unwrap()
}
fn place(m: &mut Engine, i: usize, x: f64, y: f64) {
    let r = &mut m.robots[i];
    r.x = x;
    r.y = y;
    r.z = height_at(&m.arena, x, y);
}
fn zone_center(m: &Engine, id: &str) -> [f64; 2] {
    let z = m.arena.zones.iter().find(|z| z.id == id).unwrap();
    if z.polygon.is_empty() {
        return [z.x, z.y];
    }
    let n = z.polygon.len() as f64;
    [
        z.polygon.iter().map(|p| p[0]).sum::<f64>() / n,
        z.polygon.iter().map(|p| p[1]).sum::<f64>() / n,
    ]
}

#[test]
fn outpost_optical_plates_require_camera_range_facing_and_clear_ray() {
    use rapier3d::prelude::*;
    let mut m = match_();
    let mut r = m.robots[robot(&m, "red-7")].clone();
    let plate = Engine::structure_plates_2027(&m.outposts["blue"], true)[0];
    r.x = plate.center[0] + plate.normal[0] * 2.0;
    r.y = plate.center[1] + plate.normal[1] * 2.0;
    r.z = plate.center[2] - r.muzzle_height_m - 0.08;
    r.heading = 0.0; r.quaternion = [0.0, 0.0, 0.0, 1.0]; r.pitch = 0.0;
    r.aim = (-plate.normal[1]).atan2(-plate.normal[0]);
    assert!(m.visible_outpost_plates(&r).get("blue").unwrap().contains(&plate.face));
    let mut away = r.clone(); away.aim += std::f64::consts::PI;
    assert!(!m.visible_outpost_plates(&away).contains_key("blue"));
    away = r.clone(); away.x += plate.normal[0] * 100.0; away.y += plate.normal[1] * 100.0;
    assert!(!m.visible_outpost_plates(&away).contains_key("blue"));
    let world = m.physics.as_mut().unwrap();
    world.colliders.insert(ColliderBuilder::cuboid(0.5, 0.5, 1.0)
        .translation(Vector::new((r.x + plate.center[0]) / 2.0, (r.y + plate.center[1]) / 2.0, plate.center[2])).build());
    world.update_base_covers(&m.bases);
    assert!(!m.visible_outpost_plates(&r).contains_key("blue"));
}

#[test]
fn operator_camera_mount_follows_gimbal_and_body_pose() {
    use rapier3d::na::{UnitQuaternion, Vector3};
    let m = match_();
    for source in m.robots.iter().filter(|r| r.role != "aerial") {
        for yaw in [0.0, 1.8, -2.7] {
            for pitch in [-0.26, 0.0, 0.78] {
                let mut r = source.clone();
                r.heading = 0.4;
                r.aim = r.heading + yaw;
                r.pitch = pitch;
                let body = UnitQuaternion::from_euler_angles(0.12, -0.08, r.heading);
                let q = body.quaternion();
                r.quaternion = [q.i, q.j, q.k, q.w];
                let eye = Vector3::from(Engine::eye(&r));
                let pivot = Vector3::new(r.x, r.y, r.z)
                    + body * Vector3::new(0.0, 0.0, r.muzzle_height_m);
                let direction = Vector3::from(physics::muzzle(&r).1);
                let offset = eye - pivot;
                assert!((offset.norm() - 0.08).abs() < 1e-10);
                assert!(offset.dot(&direction).abs() < 1e-10);
                let camera = m.camera_metadata(&r);
                assert_eq!(camera["origin"], json!(Engine::eye(&r)));
                assert_eq!(camera["direction"], json!(physics::muzzle(&r).1));
                let center = eye + direction * 5.0;
                assert!(m.point_in_camera(&r, center.into()));
                assert!(!m.point_in_camera(&r, (eye - direction).into()));
            }
        }
    }
}

#[test]
fn wreck_camera_is_live_optical_and_does_not_extend_policy_vision() {
    let mut m = match_();
    let i = robot(&m, "red-3");
    assert_eq!(m.operator_observation_without_arena("red", 3), m.observation_without_arena("red", Some(&[3])));
    m.robots[i].alive = false;
    m.robots[i].hp = 0.0;
    let policy = m.observation_without_arena("red", Some(&[3]));
    assert!(policy["vision"]["3"].is_null());
    let view = m.operator_observation_without_arena("red", 3);
    assert_eq!(view["vision"].as_object().unwrap().len(), 1);
    assert_eq!(view["vision"]["3"]["camera"], m.camera_metadata(&m.robots[i]));
    for r in m.robots.iter().filter(|r| r.id != "red-3") {
        let field = if r.team == "red" { "allies" } else { "enemies" };
        assert_eq!(view["vision"]["3"][field].as_array().unwrap().contains(&json!(r.id)), m.visible(&m.robots[i], r));
    }
    // Camera metadata is recalculated from the current physical wreck, not cached at death.
    m.robots[i].z += 0.25;
    let next = m.operator_observation_without_arena("red", 3);
    assert_ne!(view["vision"]["3"]["camera"]["origin"], next["vision"]["3"]["camera"]["origin"]);
    m.robots[i].alive = true;
    assert_eq!(m.operator_observation_without_arena("red", 3), m.observation_without_arena("red", Some(&[3])));
}

#[test]
fn operator_wreck_camera_does_not_bypass_cards_or_create_a_hangar_camera() {
    let mut m = match_();
    let i = robot(&m, "red-3");
    m.robots[i].alive = false;
    m.robots[i].fpv_blocked_until = 10.0;
    assert!(m.operator_observation_without_arena("red", 3)["vision"]["3"].is_null());
    m.robots[i].fpv_blocked_until = 0.0;
    m.robots[i].red_card = true;
    assert!(m.operator_observation_without_arena("red", 3)["vision"]["3"].is_null());
    assert!(m.operator_observation_without_arena("red", 6)["vision"]["6"].is_null());
}

#[test]
fn wreck_gimbal_ignores_controls_and_keeps_its_body_mount_until_revived() {
    let mut m = match_();
    let i = robot(&m, "red-3");
    m.robots[i].alive = false;
    m.robots[i].hp = 0.0;
    m.robots[i].respawn_required = 1000.0;
    m.robots[i].pitch = 0.25;
    m.robots[i].aim = m.robots[i].heading + 0.4;
    m.advance(&json!({"red":{"3":{"aim":2.0,"pitch":-0.2,"move":[1.0,1.0],"fire":true}}}));
    let r = &m.robots[i];
    assert!((r.pitch - 0.25).abs() < 1e-10);
    assert!((r.aim - r.heading - 0.4).abs() < 1e-10);
    assert!(!r.powered);
    assert!(!m.events.iter().any(|e| e["type"]=="shot" && e["robot"]=="red-3"));
    m.robots[i].alive = true;
    m.robots[i].hp = m.robots[i].max_hp;
    m.advance(&json!({"red":{"3":{"aim":2.0,"pitch":-0.2}}}));
    assert!((m.robots[i].pitch - 0.25).abs() > 0.01);
}

#[test]
fn folded_operator_camera_keeps_its_mount_above_the_lowered_gun() {
    let m = match_();
    let mut r = m.robots[robot(&m, "red-4")].clone();
    r.pitch = 0.0;
    r.quaternion = [0.0, 0.0, 0.0, 1.0];
    let raised = Engine::eye(&r);
    let delta = r.muzzle_height_m - 0.145;
    r.muzzle_height_m = 0.145;
    r.transform_progress = 1.0;
    let folded = Engine::eye(&r);
    assert!((raised[2] - folded[2] - delta).abs() < 1e-10);
    assert!((folded[2] - r.z - 0.225).abs() < 1e-10);
    assert_eq!(m.camera_metadata(&r)["horizontal_fov_deg"], json!(100.0));
    assert_eq!(m.camera_metadata(&r)["vertical_fov_deg"], json!(70.0));
}
fn lineup(models: &[(u32, u32)]) -> Option<BTreeMap<String, Vec<Deployment>>> {
    let list: Vec<Deployment> = models
        .iter()
        .map(|(slot, model)| Deployment {
            slot: *slot,
            model: *model,
        })
        .collect();
    Some([("red".into(), list.clone()), ("blue".into(), list)].into())
}

#[test]
fn default_lineup_fields_heavy_infantry_sentry_and_aerial_system() {
    let m = match_();
    assert_eq!(m.season, 2027);
    let slots: Vec<u32> = m
        .robots
        .iter()
        .filter(|r| r.team == "red")
        .map(|r| r.slot)
        .collect();
    assert_eq!(slots, vec![1, 3, 4, 6, 7]);
    let heavy = &m.robots[robot(&m, "red-1")];
    assert_eq!(
        (
            heavy.role.as_str(),
            heavy.design_slot(),
            heavy.hp,
            heavy.launcher.as_str()
        ),
        ("heavy", 2711, 400.0, "locked")
    );
    let sentry = &m.robots[robot(&m, "red-7")];
    assert_eq!(
        (
            sentry.role.as_str(),
            sentry.hp,
            sentry.ammo,
            sentry.max_heat,
            sentry.posture.as_str()
        ),
        ("sentry", 400.0, 300, 260.0, "move")
    );
    let aerial = &m.robots[robot(&m, "red-6")];
    assert_eq!((aerial.role.as_str(), aerial.drones.len()), ("aerial", 4));
    assert!(aerial.drones.iter().all(|d| d.state == "docked"));
    assert_eq!(
        (m.bases["red"].max_hp, m.outposts["red"].max_hp),
        (10000.0, 1000.0)
    );
    assert_eq!(m.teams["red"].energy_units_left, Some(6));
    for id in ["red_outpost", "red_base", "blue_outpost", "blue_base"] {
        let p = &m.positions[id];
        assert_eq!(
            (p.control.as_str(), p.controller.as_str()),
            ("full", p.owner.as_str())
        );
    }
    assert_eq!(m.snapshot()["rules"], "rmuc-2027-v1.0.0-arena-v1");
}

#[test]
fn only_2027_fields_and_legal_models_are_admitted() {
    let hero = Engine::new_with_lineups(
        42,
        10.0,
        "rmuc2027",
        lineup(&[(1, 1), (3, 3), (4, 4), (6, 2761), (7, 2771)]),
    );
    assert!(hero.is_err());
    let heavy = Engine::new_with_lineups(
        42,
        10.0,
        "rmuc2026",
        lineup(&[(1, 2711), (2, 2), (3, 3), (4, 4), (6, 6), (7, 7)]),
    );
    assert!(heavy.err().unwrap().contains("Only RMUC 2027"));
    let drone = Engine::new_with_lineups(42, 10.0, "rmuc2027", lineup(&[(1, 2711), (6, 6)]));
    assert!(drone.is_err());
    // A 2026 infantry passes the 2027 infantry limits and plays a 2027 slot.
    let mixed = Engine::new_with_lineups(
        42,
        10.0,
        "rmuc2027",
        lineup(&[(1, 2712), (3, 301), (4, 304), (6, 2762), (7, 2772)]),
    )
    .unwrap();
    assert_eq!(mixed.robots[robot(&mixed, "red-3")].design_slot(), 301);
    let twice = Engine::new_with_lineups(42, 10.0, "rmuc2027", lineup(&[(1, 2761), (2, 2762)]));
    assert!(twice.err().unwrap().contains("one aerial system"));
}

#[test]
fn coins_follow_the_2027_schedule() {
    let mut m = match_();
    m.tick = 20;
    m.step(&json!({}));
    assert_eq!(m.teams["red"].resources, 800.0);
    m.tick = 1220;
    m.step(&json!({}));
    assert_eq!(m.teams["red"].resources, 900.0);
    m.tick = 7220;
    m.step(&json!({}));
    assert_eq!(m.teams["red"].resources, 1200.0);
}

#[test]
fn occupying_an_outpost_position_takes_temporary_control_and_strips_the_base_defense() {
    let mut m = match_();
    let blue = robot(&m, "blue-3");
    let card = zone_center(&m, "red_outpost");
    place(&mut m, blue, card[0], card[1]);
    m.physics = Some(physics::PhysicsWorld::new(
        &m.arena,
        &m.robots,
        &config().robot_designs,
    ));
    assert_eq!(m.structure_defense("red", false), 1.0);
    for _ in 0..140 {
        m.robots[blue].x = card[0];
        m.robots[blue].y = card[1];
        m.step(&json!({}));
        if m.positions["red_outpost"].control == "temporary" {
            break;
        }
    }
    let p = &m.positions["red_outpost"];
    assert_eq!(
        (p.control.as_str(), p.controller.as_str()),
        ("temporary", "blue")
    );
    assert_eq!(m.structure_defense("red", false), 0.0);
    assert!(m.fortress_active("red"));
    // Absent for five seconds, the temporary holder hands it back.
    place(&mut m, blue, 20.0, 7.5);
    m.physics = Some(physics::PhysicsWorld::new(
        &m.arena,
        &m.robots,
        &config().robot_designs,
    ));
    for _ in 0..200 {
        m.step(&json!({}));
    }
    let p = &m.positions["red_outpost"];
    assert_eq!((p.control.as_str(), p.controller.as_str()), ("full", "red"));
}

#[test]
fn a_destroyed_outpost_is_rebuilt_for_the_team_that_takes_full_control() {
    let mut m = match_();
    m.outposts.get_mut("red").unwrap().hp = 0.0;
    m.outposts.get_mut("red").unwrap().ever_destroyed = true;
    m.outpost_destroyed("red");
    assert_eq!(m.positions["red_outpost"].control, "temporary");
    assert_eq!(m.positions["red_outpost"].controller, "red");
    for _ in 0..120 {
        m.step(&json!({}));
    }
    let p = &m.positions["red_outpost"];
    assert_eq!(
        (p.control.as_str(), p.controller.as_str()),
        ("full", "blue")
    );
    let o = &m.outposts["red"];
    assert_eq!((o.hp, o.owner.as_deref()), (500.0, Some("blue")));
    // Holding both outpost positions builds control degree for blue.
    for _ in 0..40 {
        m.step(&json!({}));
    }
    assert!(m.teams["blue"].control.unwrap() >= 1.9);
    assert_eq!(m.teams["red"].control.unwrap(), 0.0);
}

#[test]
fn an_aerial_robot_strike_damages_enemy_outpost_armor_and_wears_its_guard() {
    let mut m = match_();
    m.tick = 3620; // outposts stop rotating after 3 minutes
    let a = robot(&m, "blue-6");
    m.step(&json!({"blue":{"6":{"drones":{"1":{"launch":true}}}}}));
    let plate = Engine::structure_plates_2027(&m.outposts["red"], true)[0];
    let n = plate.normal;
    let start = [
        plate.center[0] + n[0] * 0.5,
        plate.center[1] + n[1] * 0.5,
        plate.center[2],
    ];
    let speed = 3.0;
    m.robots[a].drones[0].state = "flying".into();
    m.robots[a].drones[0].launch_until = 0.0;
    m.physics
        .as_mut()
        .unwrap()
        .place_drone(a, 0, start, [-n[0] * speed, -n[1] * speed, 0.0]);
    let before = m.outposts["red"].hp;
    let mut struck = false;
    for _ in 0..20 {
        m.step(&json!({"blue":{"6":{"drones":{"1":{"move":[-n[0],-n[1],0.0]}}}}}));
        struck |= m
            .events
            .iter()
            .any(|e| e["type"] == "drone_strike" && e["valid"] == true);
        if struck {
            break;
        }
    }
    assert!(struck, "{:?}", m.robots[a].drones[0].z);
    assert_eq!(m.outposts["red"].hp, before - 200.0);
    assert_eq!(m.robots[a].drones[0].integrity, 2.0);
    assert_eq!(m.teams["blue"].attack_damage, 200.0);
}

#[test]
fn the_heavy_launcher_unlocks_fits_at_the_workbench_and_installs_with_its_first_shot() {
    let mut m = match_();
    let h = robot(&m, "red-1");
    m.teams.get_mut("red").unwrap().completed_difficulties = vec![1, 2, 3];
    m.step(&json!({}));
    assert_eq!(m.robots[h].launcher, "unlocked");
    let dock = m
        .arena
        .zones
        .iter()
        .find(|z| z.id == "red_workbench")
        .map(|z| [z.x, z.y])
        .unwrap();
    place(&mut m, h, dock[0], dock[1]);
    m.physics = Some(physics::PhysicsWorld::new(
        &m.arena,
        &m.robots,
        &config().robot_designs,
    ));
    for _ in 0..160 {
        m.step(&json!({"red":{"1":{"install_launcher":true,"brake":true}}}));
        if m.robots[h].launcher == "fitted" {
            break;
        }
    }
    assert_eq!(m.robots[h].launcher, "fitted");
    m.robots[h].ammo = 3;
    let mut shots = 0;
    for _ in 0..10 {
        m.step(&json!({"red":{"1":{"fire":true,"aim":0.0,"pitch":0.1,"brake":true}}}));
        shots += m.events.iter().filter(|e| e["type"] == "shot").count();
        if m.robots[h].launcher == "installed" {
            break;
        }
    }
    assert_eq!(m.robots[h].launcher, "installed");
    assert!(shots >= 1);
    assert_eq!(m.robots[h].max_hp, 600.0);
    assert_eq!(m.robots[h].max_heat, 300.0);
    assert!(
        m.projectiles
            .iter()
            .any(|p| p.source == "red-1" && p.damage == 0.0)
            || shots == 1
    );
}

#[test]
fn launcher_unlocks_earlier_for_the_team_that_falls_behind() {
    let mut m = match_();
    m.bases.get_mut("red").unwrap().hp = 7500.0; // blue leads by 2500 HP: 300 - 50 = 250 s
    m.step(&json!({}));
    assert_eq!(m.teams["red"].launcher_unlock_s, Some(250.0));
    assert_eq!(m.teams["blue"].launcher_unlock_s, Some(300.0));
    m.tick = 250 * 20;
    m.step(&json!({}));
    assert_eq!(m.robots[robot(&m, "red-1")].launcher, "unlocked");
    assert_eq!(m.robots[robot(&m, "blue-1")].launcher, "locked");
}

#[test]
fn sentry_postures_trade_power_cooling_and_protection() {
    let mut m = match_();
    let s = robot(&m, "red-7");
    m.step(&json!({}));
    assert_eq!(m.robots[s].power_limit, 150.0); // move posture: 100 W x 1.5
    assert_eq!(m.robots[s].vulnerability, 0.25);
    m.tick = 200;
    m.step(&json!({"red":{"7":{"posture":"attack"}}}));
    assert_eq!(m.robots[s].posture, "attack");
    assert_eq!(
        (m.robots[s].power_limit, m.robots[s].cooling_multiplier),
        (50.0, 3.0)
    );
    m.step(&json!({"red":{"7":{"posture":"defense"}}}));
    assert_eq!(m.robots[s].posture, "attack", "5 s switch cooldown");
    m.tick = 320;
    m.step(&json!({"red":{"7":{"posture":"defense","posture_enhanced":true}}}));
    assert_eq!(
        (m.robots[s].posture.as_str(), m.robots[s].posture_enhanced),
        ("defense", true)
    );
    assert_eq!(m.robots[s].defense_buff, 0.99);
    for _ in 0..320 {
        m.step(&json!({}));
    }
    assert!(!m.robots[s].posture_enhanced, "15 s enhanced budget");
    assert_eq!(m.robots[s].defense_buff, 0.5);
}

#[test]
fn sentry_collects_one_hundred_rounds_per_minute_at_its_supply_area() {
    let mut m = match_();
    let s = robot(&m, "red-7");
    m.tick = 20 * 125;
    let supply = zone_center(&m, "red_supply");
    place(&mut m, s, supply[0], supply[1]);
    m.physics = Some(physics::PhysicsWorld::new(
        &m.arena,
        &m.robots,
        &config().robot_designs,
    ));
    m.step(&json!({}));
    assert_eq!(m.robots[s].ammo, 500);
    m.step(&json!({}));
    assert_eq!(m.robots[s].ammo, 500);
}

#[test]
fn lost_base_hp_slows_respawn_and_remote_healing() {
    let mut m = match_();
    assert_eq!(m.logistics("red"), 1.0);
    m.bases.get_mut("red").unwrap().hp = 6000.0;
    assert!((m.logistics("red") - 0.8).abs() < 1e-12);
    let i = robot(&m, "red-3");
    let p = Projectile {
        id: 1,
        source: "blue-4".into(),
        team: "blue".into(),
        role: "infantry".into(),
        x: 0.0,
        y: 0.0,
        z: 0.0,
        vx: -25.0,
        vy: 0.0,
        vz: 0.0,
        remaining: 1.0,
        damage: 20.0,
        radius: 0.0084,
        deployed: false,
    };
    m.robots[i].hp = 5.0;
    m.tick = 2000; // 100 s: (10 + 10) / 0.8 = 25
    let r = m.robots[i].clone();
    let (origin, _) = physics::muzzle(&r);
    let face = [
        r.x + r.chassis_size_m[0] / 2.0,
        r.y,
        r.z + r.armor_center_z_m,
    ];
    let _ = origin;
    m.robot_hit_2027(
        i,
        &Projectile {
            x: face[0] + 0.2,
            y: face[1],
            z: face[2],
            ..p
        },
        face[0],
        face[1],
        face[2],
    );
    assert!(!m.robots[i].alive);
    assert_eq!(m.robots[i].respawn_required, 25.0);
}

#[test]
fn tunnel_cards_in_order_give_the_through_or_over_buffs() {
    for (cards, buff) in [
        (vec![0, 1, 2], "tunnel_through"),
        (vec![2, 0], "tunnel_over"),
    ] {
        let mut m = match_();
        let i = robot(&m, "red-4");
        for card in cards {
            let c = zone_center(&m, &format!("red_far_tunnel_{card}"));
            place(&mut m, i, c[0], c[1]);
            m.update_buffs_2027();
            m.tick += 4;
        }
        assert!(
            m.robots[i].crossing_buffs.contains_key(buff),
            "{buff}: {:?}",
            m.robots[i].crossing_buffs
        );
        if buff == "tunnel_through" {
            assert!(
                m.robots[i].crossing_cooling_until > m.time(),
                "crossing_cooling_until not updated: {}",
                m.robots[i].crossing_cooling_until
            );
        }
    }
}

#[test]
fn timeout_ranks_control_degree_before_damage() {
    let mut m = Engine::new(42, 2.0, "rmuc2027").unwrap();
    m.teams.get_mut("red").unwrap().control = Some(3.0);
    m.teams.get_mut("blue").unwrap().attack_damage = 900.0;
    for _ in 0..40 {
        m.step(&json!({}));
    }
    assert_eq!(
        (m.winner.as_deref(), m.reason.as_deref()),
        (Some("red"), Some("timeout"))
    );
}

#[test]
fn the_2027_heavy_completes_a_level_one_assembly_physically() {
    let mut m = match_();
    let h = robot(&m, "red-1");
    let zone = m
        .arena
        .zones
        .iter()
        .find(|z| z.id == "red_assembly")
        .unwrap()
        .clone();
    place(&mut m, h, zone.x + 0.14, zone.y + 0.14);
    let r = &mut m.robots[h];
    r.heading = std::f64::consts::FRAC_PI_4;
    r.quaternion = [0.0, 0.0, (r.heading / 2.0).sin(), (r.heading / 2.0).cos()];
    r.carried_units = 1;
    m.physics = Some(physics::PhysicsWorld::new(
        &m.arena,
        &m.robots,
        &config().robot_designs,
    ));
    for _ in 0..500 {
        m.step(&json!({"red":{"1":{"assembly":1,"assembly_insert":true,"assembly_confirm":true,"brake":true}}}));
        if ["complete", "failed"].contains(&m.assemblies["red"].stage.as_str()) {
            break;
        }
    }
    assert_eq!(
        m.assemblies["red"].stage,
        "complete",
        "{}",
        m.assembly_snapshots()["red"]
    );
    assert_eq!(m.robots[h].level, 5);
    assert_eq!(m.robots[h].max_hp, 450.0);
    assert_eq!(m.teams["red"].tech_income, 40.0);
}

#[test]
fn every_heavy_picks_two_energy_units_at_the_resource_hub() {
    // The arm masts stand at the front corners beside the central gimbal; from the SDK's hub
    // approach (resource_pickup_point) each heavy still reaches the 0.70 m units.
    for model in [2711, 2712, 2713] {
        let mut m = Engine::new_with_lineups(
            42,
            420.0,
            "rmuc2027",
            lineup(&[(1, model), (3, 2731), (4, 2732), (6, 2761), (7, 2771)]),
        )
        .unwrap();
        let h = robot(&m, "red-1");
        let station = m.arena.field_geometry["fixtures"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["kind"] == "resource" && f["team"] == "red")
            .unwrap()
            .clone();
        let (point, dir) = (&station["pick_point"], &station["pick_dir"]);
        let (dx, dy) = (dir[0].as_f64().unwrap(), dir[1].as_f64().unwrap());
        let reach = if model == 2712 { 0.48 } else { 0.60 };
        place(
            &mut m,
            h,
            point[0].as_f64().unwrap() - dx * reach,
            point[1].as_f64().unwrap() - dy * reach,
        );
        let r = &mut m.robots[h];
        r.heading = dy.atan2(dx);
        r.aim = r.heading;
        r.quaternion = [0.0, 0.0, (r.heading / 2.0).sin(), (r.heading / 2.0).cos()];
        m.physics = Some(physics::PhysicsWorld::new(
            &m.arena,
            &m.robots,
            &config().robot_designs,
        ));
        for _ in 0..600 {
            m.step(&json!({"red":{"1":{"pick_unit":true,"brake":true,"heading":dy.atan2(dx)}}}));
            if m.robots[h].carried_units == 2 {
                break;
            }
        }
        assert_eq!(
            m.robots[h].carried_units,
            2,
            "{model}: {:?}",
            m.robots[h].unit_pickup.as_ref().map(|p| p.stage.clone())
        );
        assert_eq!(m.teams["red"].energy_units_left, Some(4), "{model}");
    }
}

#[test]
fn launcher_fitting_restarts_when_the_heavy_robot_stops_asking_or_leaves_the_workbench() {
    let mut m = match_();
    let h = robot(&m, "red-1");
    m.teams.get_mut("red").unwrap().completed_difficulties = vec![1, 2, 3];
    m.step(&json!({}));
    let dock = m
        .arena
        .zones
        .iter()
        .find(|z| z.id == "red_workbench")
        .map(|z| [z.x, z.y])
        .unwrap();
    place(&mut m, h, dock[0], dock[1]);
    m.physics = Some(physics::PhysicsWorld::new(
        &m.arena,
        &m.robots,
        &config().robot_designs,
    ));
    for _ in 0..6 {
        m.step(&json!({"red":{"1":{"install_launcher":true,"brake":true}}}));
    }
    assert_eq!(m.robots[h].launcher, "installing");
    assert!(m.robots[h].launcher_progress > 0.0);
    for _ in 0..4 {
        m.step(&json!({"red":{"1":{"brake":true}}}));
    }
    assert_eq!(
        (m.robots[h].launcher.as_str(), m.robots[h].launcher_progress),
        ("unlocked", 0.0)
    );
}

#[test]
fn tunnel_passages_match_the_road_area_plan() {
    // Figure 4-35: the near-highland tunnel's 800 mm passage lies between its rail on the
    // central highland's edge and the far tunnel's north rail, which the tunnels share; the
    // road tunnel has only its portal posts, 800 mm apart, over the top of its ramp.
    let m = match_();
    let walls = |id: &str| {
        m.arena
            .obstacles
            .iter()
            .filter(|w| w.id == id)
            .cloned()
            .collect::<Vec<_>>()
    };
    let (near, far) = (walls("red_near_tunnel_side"), walls("red_far_tunnel_side"));
    assert_eq!((near.len(), far.len()), (1, 2));
    assert!(
        (far[0].y - (near[0].y + near[0].h) - 0.8).abs() < 0.005,
        "near passage {}",
        far[0].y - near[0].y - near[0].h
    );
    assert!((far[1].y - (far[0].y + far[0].h) - 0.8).abs() < 0.005);
    assert!(walls("red_road_tunnel_side").is_empty());
    let posts = walls("red_road_tunnel_post");
    assert_eq!(posts.len(), 2);
    // Posts run along y (rotated a quarter turn): their centres sit 0.8 + 0.1 m apart.
    let centre = |w: &Wall| w.x + w.w / 2.0;
    assert!(((centre(&posts[1]) - centre(&posts[0])).abs() - 0.9).abs() < 1e-6);
    let ramp = m
        .arena
        .terrain
        .iter()
        .find(|t| t.id == "red_road_tunnel_ramp")
        .unwrap();
    // Heights above the tilted floor (1.5 degrees toward the long sides).
    let above = |y: f64| {
        height_at(&m.arena, 9.32, y) - (15.0 - y) * m.arena.floor_slope_deg.to_radians().tan()
    };
    let (foot, top) = (above(ramp.y + 0.01), above(12.9));
    assert!(
        foot < 0.01 && (top - 0.2).abs() < 0.01,
        "ramp {foot} .. {top}"
    );
}

#[test]
fn a_folded_infantry_drives_up_the_road_tunnel_ramp_under_its_portal() {
    let mut m = Engine::new_with_lineups(
        42,
        60.0,
        "rmuc2027",
        lineup(&[(1, 2711), (3, 2731), (4, 2732), (6, 2761), (7, 2771)]),
    )
    .unwrap();
    let i = robot(&m, "red-4");
    let x = 9.32;
    place(&mut m, i, x, 11.7);
    let r = &mut m.robots[i];
    r.heading = std::f64::consts::FRAC_PI_2;
    r.quaternion = [0.0, 0.0, (r.heading / 2.0).sin(), (r.heading / 2.0).cos()];
    m.physics = Some(physics::PhysicsWorld::new(
        &m.arena,
        &m.robots,
        &config().robot_designs,
    ));
    for _ in 0..60 {
        m.step(&json!({"red":{"4":{"transform":true,"heading":std::f64::consts::FRAC_PI_2}}}));
    }
    for _ in 0..400 {
        let r = &m.robots[i];
        m.step(&json!({"red":{"4":{"move":[(x - r.x) * 1.5, 0.6],"heading":std::f64::consts::FRAC_PI_2,"transform":true,"power":120}}}));
        if m.robots[i].y > 13.5 {
            break;
        }
    }
    let r = &m.robots[i];
    assert!(
        r.y > 13.4 && r.z > 0.15 && r.upright > 0.8 && !r.tipped,
        "stopped at y {} z {} up {}",
        r.y,
        r.z,
        r.upright
    );
    assert!(
        r.crossing_buffs.contains_key("tunnel_through"),
        "{:?}",
        r.crossing_buffs
    );
}

#[test]
fn drones_pushed_into_a_spinning_outpost_stay_outside_its_armor_and_tower() {
    // A drone steered at the outpost's axis, slower than a strike at first and then at full
    // speed, never ends a tick inside an armor module, the collar or the tower.
    let mut m = match_();
    m.tick = 600;
    let a = robot(&m, "blue-6");
    m.step(&json!({"blue":{"6":{"drones":{"1":{"launch":true}}}}}));
    let radius = m.robot_models["2761"]["drone"]["radius_m"]
        .as_f64()
        .unwrap();
    let plate = Engine::structure_plates_2027(&m.outposts["red"], true)[1];
    let n = plate.normal;
    let start = [
        plate.center[0] + n[0] * 0.4,
        plate.center[1] + n[1] * 0.4,
        plate.center[2],
    ];
    m.robots[a].drones[0].state = "flying".into();
    m.robots[a].drones[0].launch_until = 0.0;
    m.physics
        .as_mut()
        .unwrap()
        .place_drone(a, 0, start, [-n[0] * 0.8, -n[1] * 0.8, 0.0]);
    let mut contacts = 0;
    for tick in 0..150 {
        let (o, d) = (&m.outposts["red"], &m.robots[a].drones[0]);
        let to = [o.x - d.x, o.y - d.y, plate.center[2] - d.z];
        let length = (to[0] * to[0] + to[1] * to[1]).sqrt().max(1e-9);
        let gain = if tick < 60 { 0.2 } else { 1.0 };
        m.step(&json!({"blue":{"6":{"drones":{"1":{"move":[gain * to[0] / length, gain * to[1] / length, to[2]]}}}}}));
        let (o, d) = (&m.outposts["red"], &m.robots[a].drones[0]);
        let p = [d.x, d.y, d.z];
        assert!(
            structures::outpost_overlap(o, p, radius * 0.98, radius * 0.39).is_none(),
            "tick {tick}: drone inside the outpost at {p:?}"
        );
        contacts += structures::outpost_overlap(o, p, radius * 1.3, radius * 0.6).is_some() as u32;
    }
    assert!(
        contacts > 20,
        "the drone should rest against the outpost, contacts {contacts}"
    );
}

#[test]
fn operator_weapon_status_uses_the_actual_firing_gate() {
    let mut m = match_();
    let i = robot(&m, "red-7");
    let mut commands = vec![Action::default(); m.robots.len()];
    commands[i].fire = true;
    for reason in ["destroyed", "red_card", "weak", "overspeed", "overheat", "cooldown", "no_ammo"] {
        let saved = m.robots[i].clone();
        match reason {
            "destroyed" => m.robots[i].alive = false,
            "red_card" => m.robots[i].red_card = true,
            "weak" => m.robots[i].weak = true,
            "overspeed" => m.robots[i].overspeed_permanent_lock = true,
            "overheat" => m.robots[i].overheated = true,
            "cooldown" => m.robots[i].cooldown = 1.0,
            "no_ammo" => m.robots[i].ammo = 0,
            _ => unreachable!(),
        }
        assert_eq!(m.snapshot()["robots"][i]["fire_block_reason"], reason);
        let ammo = m.robots[i].ammo;
        m.fire_2027(&commands);
        assert_eq!(m.robots[i].ammo, ammo);
        assert!(m.projectiles.is_empty());
        m.robots[i] = saved;
    }
    assert_eq!(m.snapshot()["robots"][i]["fire_block_reason"], "");
    let advertised = m.snapshot()["robots"][i].clone();
    m.fire_2027(&commands);
    assert_eq!(m.robots[i].ammo, 299);
    assert_eq!(m.projectiles.len(), 1);
    assert!((m.robots[i].cooldown - advertised["trigger_interval_s"].as_f64().unwrap()).abs() < 1e-10);
    let shot = &m.projectiles[0];
    let velocity = (shot.vx.powi(2) + shot.vy.powi(2) + shot.vz.powi(2)).sqrt();
    assert!((velocity - advertised["muzzle_speed_mps"].as_f64().unwrap()).abs() < 1e-10);
    assert_eq!(m.snapshot()["robots"][i]["fire_block_reason"], "cooldown");
}

#[test]
fn operator_ammo_purchase_deducts_money_and_remote_delivery_arrives_once() {
    let mut m = match_();
    let i = robot(&m, "red-3");
    let point = zone_center(&m, "red_supply");
    place(&mut m, i, point[0], point[1]);
    m.teams.get_mut("red").unwrap().resources = 500.0;
    // Refresh the RFID contact used by the same purchase RPC as the client.
    m.step(&json!({}));
    m.step(&json!({}));
    let before = m.teams["red"].resources;
    let frame = m.step(&json!({"red":{"3":{"buy_ammo":{"remote":false,"amount":10}}}}));
    assert_eq!(m.robots[i].ammo, 10);
    assert_eq!(m.teams["red"].resources, before - 10.0);
    assert!(frame["events"].as_array().unwrap().iter().any(|e| e["type"]=="ammo_purchase"));
    // RPCs execute on even source ticks.
    m.step(&json!({}));
    let before = m.teams["red"].resources;
    m.step(&json!({"red":{"3":{"buy_ammo":{"remote":true,"amount":100}}}}));
    assert_eq!(m.robots[i].ammo, 10);
    assert_eq!(m.robots[i].pending_ammo.len(), 1);
    assert_eq!(m.teams["red"].resources, before - 150.0);
    for _ in 0..119 { m.step(&json!({})); }
    assert_eq!(m.robots[i].ammo, 10);
    for _ in 0..2 { m.step(&json!({})); }
    assert_eq!(m.robots[i].ammo, 110);
    assert!(m.robots[i].pending_ammo.is_empty());
    for _ in 0..4 { m.step(&json!({})); }
    assert_eq!(m.robots[i].ammo, 110);
}

#[test]
fn radar_station_reports_occluded_ground_enemies_without_optical_leaks() {
    let mut m = match_();
    let red7 = robot(&m, "red-7");
    let blue3 = robot(&m, "blue-3");
    let blue4 = robot(&m, "blue-4");
    // Place red-7 far away facing opposite direction so blue-3 behind red_base is completely occluded.
    place(&mut m, red7, 18.0, 12.0);
    m.robots[red7].heading = 0.0;
    m.robots[red7].aim = 0.0;
    place(&mut m, blue3, 0.80, 8.12);
    m.robots[blue3].hp = 140.0;
    m.robots[blue3].ammo = 88;
    m.robots[blue3].invulnerable = true;
    m.robots[blue3].weak = true;
    // Put blue-4 dead: radar info-wave reports hp=0, alive=false, primary_state=1.
    m.robots[blue4].alive = false;
    m.robots[blue4].hp = 0.0;
    // Set blue economy and fortress reserve ammo to verify 0x0A03 / 0x0A04 fields.
    m.teams.get_mut("blue").unwrap().resources = 450.0;
    m.teams.get_mut("blue").unwrap().total_resources = 1100.0;
    m.teams.get_mut("blue").unwrap().fortress_ammo = 100;
    m.positions.get_mut("blue_outpost").unwrap().controller = "red".into();
    let blue1 = robot(&m, "blue-1");
    let bf = zone_center(&m, "blue_fortress");
    place(&mut m, blue1, bf[0], bf[1]);
    m.robots[blue1].ammo = 5;
    m.update_buffs_2027();
    m.robots[blue3].invulnerable = true;
    m.robots[blue3].weak = true;

    let obs = m.observation_without_arena("red", Some(&[7]));
    let enemies = obs["enemies"].as_array().unwrap();
    assert!(
        !enemies.iter().any(|e| e["id"] == "blue-3"),
        "Occluded blue-3 must stay absent from optical obs.enemies"
    );
    let radar = obs["radar"].as_array().unwrap();
    assert_eq!(radar.len(), 4, "Radar info-wave reports all 4 enemy ground robots regardless of camera LOS");
    let b3 = radar
        .iter()
        .find(|e| e["id"] == "blue-3")
        .expect("Radar info-wave must report occluded enemy blue-3");
    assert!((b3["x"].as_f64().unwrap() - 0.80).abs() < 1e-6);
    assert!((b3["y"].as_f64().unwrap() - 8.12).abs() < 1e-6);
    assert_eq!(b3["slot"].as_u64(), Some(3));
    assert_eq!(b3["alive"].as_bool(), Some(true));
    assert!((b3["hp"].as_f64().unwrap() - 140.0).abs() < 1e-6);
    assert_eq!(b3["ammo"].as_u64(), Some(88));
    assert_eq!(b3["invulnerable"].as_bool(), Some(true));
    assert_eq!(b3["weak"].as_bool(), Some(true));
    assert_eq!(b3["primary_state"].as_u64(), Some(3));
    assert_eq!(b3["recovery_buff"].as_u64(), Some(0));
    // 0x0A03: blue-1 on active own fortress includes fortress reserve ammo (100 / 10 = 10 42mm rounds).
    let b1 = radar.iter().find(|e| e["id"] == "blue-1").unwrap();
    assert_eq!(b1["ammo"].as_u64(), Some(5 + 10));
    let b4 = radar.iter().find(|e| e["id"] == "blue-4").unwrap();
    assert_eq!(b4["alive"].as_bool(), Some(false));
    assert!((b4["hp"].as_f64().unwrap() - 0.0).abs() < 1e-6);
    assert_eq!(b4["primary_state"].as_u64(), Some(1));
    // Must not leak optical-only, private, or protocol-absent fields.
    for forbidden in [
        "vx", "vy", "z", "model_slot", "chassis_type", "heading", "aim", "pitch", "heat",
        "max_hp", "crossing_buffs", "cooling_multiplier",
    ] {
        assert!(b3.get(forbidden).is_none(), "Radar info-wave must not expose {forbidden}");
    }
    // 0x0A04 radar_team: enemy resources, total_resources, position states, and masked zone statuses.
    let rt = &obs["radar_team"];
    assert_eq!(rt["team"].as_str(), Some("blue"));
    assert!((rt["resources"].as_f64().unwrap() - 450.0).abs() < 1e-6);
    assert!((rt["total_resources"].as_f64().unwrap() - 1100.0).abs() < 1e-6);
    assert_eq!(rt["zones"]["blue_fortress"].as_str(), Some("enemy"));
    // blue does not control blue_outpost -> blue_central and blue_outpost masked as "unknown".
    assert_eq!(rt["zones"]["blue_central"].as_str(), Some("unknown"));
    assert_eq!(rt["zones"]["blue_outpost"].as_str(), Some("unknown"));
    assert_eq!(rt["positions"]["blue_outpost"].as_str(), Some("none"));
    assert_eq!(rt["positions"]["blue_base"].as_str(), Some("full"));
    // Aerial hangar (slot 6) and own-team robots are excluded.
    assert!(!radar.iter().any(|e| e["slot"] == 6));
    assert!(radar.iter().all(|e| e["team"] == "blue"));
    // Even if observer red-7 is dead (zero optical cameras), radar info-wave still delivers enemy positions.
    m.robots[red7].alive = false;
    m.robots[red7].hp = 0.0;
    let dead_obs = m.observation_without_arena("red", Some(&[7]));
    assert!(dead_obs["enemies"].as_array().unwrap().is_empty());
    assert_eq!(dead_obs["radar"].as_array().unwrap().len(), 4);
}

#[test]
fn armor_hit_cooldowns_are_independent_for_17mm_42mm_and_aerial_impacts() {
    let mut m = match_();
    let victim = robot(&m, "blue-3");
    let heavy = robot(&m, "red-1");
    m.robots[victim].hp = 400.0;
    m.robots[victim].max_hp = 400.0;
    m.robots[victim].defense_buff = 0.0;
    m.robots[heavy].last_shot_time = 0.0;
    let hit_x = m.robots[victim].x + 0.25;
    let hit_y = m.robots[victim].y;
    let hit_z = m.robots[victim].z + m.robots[victim].armor_center_z_m;
    let p17 = Projectile {
        id: 1, source: "red-3".into(), team: "red".into(), role: "infantry".into(),
        x: hit_x, y: hit_y, z: hit_z, vx: -20.0, vy: 0.0, vz: 0.0,
        remaining: 10.0, damage: 20.0, radius: 0.0084, deployed: false,
    };
    let p42 = Projectile {
        id: 2, source: "red-1".into(), team: "red".into(), role: "heavy".into(),
        x: hit_x, y: hit_y, z: hit_z, vx: -15.0, vy: 0.0, vz: 0.0,
        remaining: 10.0, damage: 200.0, radius: 0.02125, deployed: false,
    };
    m.tick = 20; // t = 1.00 s
    m.robots[heavy].last_shot_time = m.time();
    m.robot_hit_2027(victim, &p17, hit_x, hit_y, hit_z);
    assert_eq!(m.robots[victim].hp, 380.0);
    // 50 ms later (within 42mm's 200ms window relative to the 17mm hit): 42mm MUST register!
    m.tick = 21; // t = 1.05 s
    m.robot_hit_2027(victim, &p42, hit_x, hit_y, hit_z);
    assert_eq!(m.robots[victim].hp, 180.0);
    // Another 42mm hit 100 ms later (within 42mm's own 200ms window) is suppressed.
    m.tick = 23; // t = 1.15 s
    m.robot_hit_2027(victim, &p42, hit_x, hit_y, hit_z);
    assert_eq!(m.robots[victim].hp, 180.0);

    // Base armor: 17mm hit at t=2.00s must NOT suppress 42mm hit at t=2.05s on the same plate.
    m.positions.get_mut("blue_outpost").unwrap().controller = "red".into();
    let plate = Engine::structure_plates_2027(&m.bases["blue"], false)[0].clone();
    let b17 = Projectile {
        id: 3, source: "red-3".into(), team: "red".into(), role: "infantry".into(),
        x: plate.center[0], y: plate.center[1], z: plate.center[2],
        vx: -20.0 * plate.normal[0], vy: -20.0 * plate.normal[1], vz: -20.0 * plate.normal[2],
        remaining: 10.0, damage: 20.0, radius: 0.0084, deployed: false,
    };
    let b42 = Projectile {
        id: 4, source: "red-1".into(), team: "red".into(), role: "heavy".into(),
        x: plate.center[0], y: plate.center[1], z: plate.center[2],
        vx: -15.0 * plate.normal[0], vy: -15.0 * plate.normal[1], vz: -15.0 * plate.normal[2],
        remaining: 10.0, damage: 200.0, radius: 0.02125, deployed: false,
    };
    m.tick = 40;
    m.robots[heavy].last_shot_time = m.time();
    let base_before = m.bases["blue"].hp;
    m.structure_hit_2027("blue", false, &b17, plate.center[0], plate.center[1], plate.center[2]);
    assert_eq!(m.bases["blue"].hp, base_before - 20.0);
    m.tick = 41;
    m.structure_hit_2027("blue", false, &b42, plate.center[0], plate.center[1], plate.center[2]);
    assert_eq!(m.bases["blue"].hp, base_before - 220.0);
}

#[test]
fn respawn_requires_six_continuous_seconds_alive_before_disengaged() {
    let mut m = match_();
    let i = robot(&m, "red-3");
    // At match start, robots default to disengaged (Table 2-1 p.16).
    assert!(m.disengaged_2027(i));
    // Respawn at t = 250 s (more than 6 s after match start, never shot or took damage).
    m.tick = 5000; // t = 250.0 s
    m.respawn_2027(i, false);
    assert!(!m.disengaged_2027(i), "Just-respawned robot must not be immediately disengaged");
    m.tick = 5119; // t = 255.95 s
    assert!(!m.disengaged_2027(i));
    m.tick = 5120; // t = 256.00 s (6.0 s alive)
    assert!(m.disengaged_2027(i));
}

#[test]
fn sentry_posture_cooling_rounds_to_integer_and_enhanced_attack_clears_heat_immediately() {
    let mut m = match_();
    let s = robot(&m, "red-7");
    m.robots[s].posture = "defense".into();
    m.robots[s].cooling_rate = 30.0;
    m.robots[s].cooling_bonus = 10.0; // (30 + 10) * (1/3) = 13.333... -> rounds to integer 13/s -> 1.3 per 0.1s
    m.robots[s].heat = 50.0;
    m.settle_heat_power_2027(s);
    assert!((m.robots[s].heat - 48.7).abs() < 1e-9, "got {}", m.robots[s].heat);
    // Switching into enhanced attack posture immediately zeroes heat and clears non-permanent overheat.
    m.robots[s].heat = 280.0;
    m.robots[s].overheated = true;
    m.robots[s].posture_ready_at = 0.0;
    m.posture_rpc(s, &json!({"posture": "attack", "posture_enhanced": true}));
    assert_eq!(m.robots[s].heat, 0.0);
    assert!(!m.robots[s].overheated);
}

#[test]
fn overheat_disconnects_operator_fpv_but_does_not_dim_automatic_sentry() {
    let mut m = match_();
    let r3 = robot(&m, "red-3");
    let r7 = robot(&m, "red-7");
    let b3 = robot(&m, "blue-3");
    place(&mut m, r3, 10.0, 7.5);
    place(&mut m, r7, 10.0, 8.5);
    place(&mut m, b3, 14.0, 7.5);
    m.robots[r3].heading = 0.0;
    m.robots[r3].aim = 0.0;
    m.robots[r7].heading = 0.0;
    m.robots[r7].aim = 0.0;
    m.physics = Some(physics::PhysicsWorld::new(&m.arena, &m.robots, &config().robot_designs));
    // Overheat infantry red-3: FPV camera disconnects until heat reaches 0, while radar still works.
    m.robots[r3].overheated = true;
    m.robots[r3].heat = 50.0;
    let obs3 = m.observation_without_arena("red", Some(&[3]));
    assert!(obs3["vision"].as_object().unwrap().is_empty());
    assert!(obs3["enemies"].as_array().unwrap().is_empty());
    assert_eq!(obs3["radar"].as_array().unwrap().len(), 4);
    // Overheat automatic sentry red-7: no human operator FPV, keeps full 20m onboard vision range.
    m.robots[r7].overheated = true;
    m.robots[r7].heat = 270.0;
    assert!((m.effective_vision(&m.robots[r7]) - 20.0).abs() < 1e-9);
    let obs7 = m.observation_without_arena("red", Some(&[7]));
    assert!(obs7["vision"].as_object().unwrap().contains_key("7"));
    assert_eq!(obs7["vision"]["7"]["fpv_dimmed"].as_bool(), Some(false));
}

#[test]
fn permanent_overheat_emits_event_and_attack_buff_settles_at_hit_time() {
    let mut m = match_();
    let r3 = robot(&m, "red-3");
    m.robots[r3].ammo = 10;
    m.robots[r3].max_heat = 40.0;
    m.robots[r3].heat = 135.0; // Q2 = 140; next 17mm (+10) -> 145 >= Q2 permanent lock
    let frame = m.step(&json!({"red": {"3": {"fire": true}}}));
    assert!(m.robots[r3].heat_permanent_lock);
    assert!(frame["events"].as_array().unwrap().iter().any(|e| {
        e["type"] == "overheat" && e["robot"] == "red-3" && e["permanent"] == true
    }));

    // Attack buff settles at hit time rather than launch time.
    let mut m2 = match_();
    let r3 = robot(&m2, "red-3");
    let b3 = robot(&m2, "blue-3");
    m2.robots[b3].defense_buff = 0.0;
    m2.robots[r3].attack_multiplier = 2.0;
    let hit_x = m2.robots[b3].x + 0.25;
    let hit_y = m2.robots[b3].y;
    let hit_z = m2.robots[b3].z + m2.robots[b3].armor_center_z_m;
    let p = Projectile {
        id: 10, source: "red-3".into(), team: "red".into(), role: "infantry".into(),
        x: hit_x, y: hit_y, z: hit_z, vx: -20.0, vy: 0.0, vz: 0.0,
        remaining: 10.0, damage: 20.0, radius: 0.0084, deployed: false,
    };
    let hp_before = m2.robots[b3].hp;
    m2.robot_hit_2027(b3, &p, hit_x, hit_y, hit_z);
    assert_eq!(m2.robots[b3].hp, hp_before - 40.0);
    // Buff expires in flight -> next hit deals base 20.
    m2.tick += 2;
    m2.robots[r3].attack_multiplier = 1.0;
    m2.robot_hit_2027(b3, &p, hit_x, hit_y, hit_z);
    assert_eq!(m2.robots[b3].hp, hp_before - 60.0);
}

#[test]
fn heavy_42mm_mask_persists_after_dead_revive_until_ammo_positive() {
    let mut m = match_();
    let h = robot(&m, "red-1");
    m.robots[h].last_shot_time = 10.0;
    m.robots[h].alive = false;
    m.robots[h].death_time = 10.0;
    m.robots[h].ammo = 0;
    m.tick = 260; // t = 13.0 s (3.0 s dead)
    m.update_heavy_42_masks();
    assert!(m.robots[h].heavy_42_masked);
    assert!(!m.heavy_42_enabled("red"));
    // Revive with 0 ammo while still within 4 s of last_shot_time: mask MUST persist!
    m.tick = 265; // t = 13.25 s
    m.respawn_2027(h, true);
    assert!(m.robots[h].alive && m.robots[h].ammo == 0);
    m.update_heavy_42_masks();
    assert!(!m.heavy_42_enabled("red"));
    // Once ammo > 0, mask clears.
    m.robots[h].ammo = 1;
    m.update_heavy_42_masks();
    assert!(m.heavy_42_enabled("red"));

    // Same-tick pending_ammo delivery + immediate 42mm fire MUST unmask during before_step_2027
    // (before physics substeps / after_step_2027), so point-blank same-tick hits are never dropped.
    let mut m2 = match_();
    let h2 = robot(&m2, "red-1");
    let b3 = robot(&m2, "blue-3");
    m2.robots[h2].ammo = 0;
    m2.robots[h2].heavy_42_masked = true;
    m2.robots[h2].weak = false;
    m2.robots[h2].weak_until = None;
    m2.robots[h2].launcher = "installed".into();
    m2.robots[h2].last_shot_time = -10.0;
    m2.robots[h2].pending_ammo.push((0.0, 5));
    let mut actions = vec![Action::default(); m2.robots.len()];
    actions[h2].fire = true;
    m2.before_step_2027(&actions);
    assert!(
        !m2.robots[h2].heavy_42_masked,
        "before_step_2027 must unmask heavy_42_masked after pending_ammo delivery"
    );
    m2.fire_2027(&actions);
    assert!(m2.heavy_42_enabled("red"));
    assert_eq!(m2.robots[h2].ammo, 4);
    let mut shot = m2.projectiles.pop().expect("red-1 must spawn a 42mm projectile");
    m2.robots[b3].zone_contacts.clear();
    m2.robots[b3].invulnerable = false;
    m2.robots[b3].defense_buff = 0.0;
    m2.robots[b3].hp = 400.0;
    m2.robots[b3].max_hp = 400.0;
    let hit_x = m2.robots[b3].x + 0.25;
    let hit_y = m2.robots[b3].y;
    let hit_z = m2.robots[b3].z + m2.robots[b3].armor_center_z_m;
    shot.vx = -15.6;
    shot.vy = 0.0;
    shot.vz = 0.0;
    m2.robot_hit_2027(b3, &shot, hit_x, hit_y, hit_z);
    assert_eq!(
        m2.robots[b3].hp, 200.0,
        "Same-tick 42mm hit before after_step_2027 must register full 200 HP damage"
    );

    // Same check for local buy_ammo RPC unmasking inside before_step_2027.
    let mut m3 = match_();
    let h3 = robot(&m3, "red-1");
    let sup = zone_center(&m3, "red_supply");
    place(&mut m3, h3, sup[0], sup[1]);
    m3.update_buffs_2027();
    m3.teams.get_mut("red").unwrap().resources = 500.0;
    m3.robots[h3].ammo = 0;
    m3.robots[h3].heavy_42_masked = true;
    m3.robots[h3].weak = false;
    m3.robots[h3].weak_until = None;
    let mut buy_actions = vec![Action::default(); m3.robots.len()];
    buy_actions[h3].rpc = json!({"buy_ammo": {"remote": false, "amount": 5}});
    m3.before_step_2027(&buy_actions);
    assert!(m3.robots[h3].ammo >= 5);
    assert!(
        !m3.robots[h3].heavy_42_masked,
        "before_step_2027 must unmask heavy_42_masked after local buy_ammo RPC"
    );
}

#[test]
fn non_projectile_damage_and_kill_fallback_xp_include_capped_infantry() {
    let mut m = match_();
    let r3 = robot(&m, "red-3");
    let r4 = robot(&m, "red-4");
    // Cap red-3 at level 5 (2200 XP), leave red-4 at level 1 (0 XP).
    m.robots[r3].level = 5;
    m.robots[r3].experience = 2200.0;
    m.robots[r4].level = 1;
    m.robots[r4].experience = 0.0;
    // Non-projectile damage (e.g. drone impact 200 HP -> 100 XP) splits across ALL 2 living infantry -> 50 XP each (red-3 capped, red-4 gets 50).
    m.award_damage_xp_2027(None, "red", 100.0, None);
    assert!((m.robots[r3].experience - 2200.0).abs() < 1e-6);
    assert!((m.robots[r4].experience - 50.0).abs() < 1e-6);
    // Kill fallback against level-5 victim: mean XP = (2200 + 50) / 2 = 1125 -> level 3 (threshold 1100).
    // XP = 50 * 5 * (1 + 0.2 * (5 - 3)) = 350; split across 2 living infantry -> 175 to red-4 (NOT 450!).
    m.award_kill_xp_2027(None, "red", 5, Some("17mm"));
    assert!((m.robots[r4].experience - 225.0).abs() < 1e-6, "got {}", m.robots[r4].experience);
}

#[test]
fn crossing_combo_uses_tunnel_through_remaining_and_cannot_infinite_refresh_50_defense() {
    let mut m = match_();
    let i = robot(&m, "red-3");
    // 1. Trigger tunnel_through (120s cooling) at t = 0..2s, then road (5s) at t = 10s.
    // Rule 5.6.4.5: 50% defense lasts max(112s remaining, 5s) = 112s -> until t = 122s.
    for (tick, id) in [(0, "red_road_tunnel_0"), (20, "red_road_tunnel_1"), (40, "red_road_tunnel_2")] {
        m.tick = tick;
        let z = m.arena.zones.iter().find(|z| z.id == id).unwrap().clone();
        m.robots[i].x = z.x; m.robots[i].y = z.y; m.robots[i].z = z.card_z_m.unwrap_or_else(|| height_at(&m.arena, z.x, z.y));
        m.update_buffs_2027();
    }
    assert!(m.robots[i].crossing_buffs.contains_key("tunnel_through"));
    for (tick, id) in [(180, "red_road_0"), (200, "red_road_1")] {
        m.tick = tick;
        let z = m.arena.zones.iter().find(|z| z.id == id).unwrap().clone();
        m.robots[i].x = z.x; m.robots[i].y = z.y; m.robots[i].z = z.card_z_m.unwrap_or_else(|| height_at(&m.arena, z.x, z.y));
        m.update_buffs_2027();
    }
    assert!((m.robots[i].defense_buff - 0.5).abs() < 1e-6);
    assert!((m.robots[i].crossing_defense_until - 122.0).abs() < 1e-6, "got {}", m.robots[i].crossing_defense_until);

    // 2. Repeating the same crossing type cannot extend 50% defense past its original combo expiry.
    let mut m2 = match_();
    let j = robot(&m2, "red-4");
    for (tick, id) in [(0, "red_road_0"), (20, "red_road_1")] {
        m2.tick = tick;
        let z = m2.arena.zones.iter().find(|z| z.id == id).unwrap().clone();
        m2.robots[j].x = z.x; m2.robots[j].y = z.y; m2.robots[j].z = height_at(&m2.arena, z.x, z.y);
        m2.update_buffs_2027();
    }
    // t=2s: highland (30s) while road (until t=6s) is active -> 50% defense until t=32s.
    for (tick, id) in [(30, "red_highland_low"), (40, "red_central")] {
        m2.tick = tick;
        let z = m2.arena.zones.iter().find(|z| z.id == id).unwrap().clone();
        m2.robots[j].x = z.x; m2.robots[j].y = z.y; m2.robots[j].z = height_at(&m2.arena, z.x, z.y);
        m2.update_buffs_2027();
    }
    assert!((m2.robots[j].crossing_combo_until - 32.0).abs() < 1e-6);
    // t=15s: road has expired, repeat highland (30s -> t=45s) alone. Combo MUST remain 32s!
    for (tick, id) in [(280, "red_highland_low"), (300, "red_central")] {
        m2.tick = tick;
        let z = m2.arena.zones.iter().find(|z| z.id == id).unwrap().clone();
        m2.robots[j].x = z.x; m2.robots[j].y = z.y; m2.robots[j].z = height_at(&m2.arena, z.x, z.y);
        m2.update_buffs_2027();
    }
    assert!((m2.robots[j].crossing_combo_until - 32.0).abs() < 1e-6);
    // Move off central card and check t=33s: 50% expired, falls back to 25% highland defense!
    m2.tick = 660;
    place(&mut m2, j, 5.0, 7.5);
    m2.robots[j].zone_contacts.clear();
    m2.update_buffs_2027();
    assert!((m2.robots[j].defense_buff - 0.25).abs() < 1e-6, "got {}", m2.robots[j].defense_buff);
}

#[test]
fn enemy_fortress_hold_is_team_shared_and_retains_three_seconds_across_death() {
    let mut m = match_();
    m.tick = 3700; // t = 185s (> 180s)
    m.positions.get_mut("red_outpost").unwrap().controller = "blue".into();
    let b3 = robot(&m, "blue-3");
    let b4 = robot(&m, "blue-4");
    let rf = zone_center(&m, "red_fortress");
    // blue-3 holds red_fortress for 6.0 s (120 ticks), then dies.
    place(&mut m, b3, rf[0], rf[1]);
    for _ in 0..120 {
        m.update_buffs_2027();
        m.tick += 1;
    }
    assert!(!m.bases["red"].armor_open);
    assert!((m.bases["red"].fortress_enemy_hold - 6.0).abs() < 1e-6);
    m.robots[b3].alive = false;
    // Wait 2.0 s (< 3 s retention window): timer pauses and is retained.
    for _ in 0..40 {
        m.update_buffs_2027();
        m.tick += 1;
    }
    assert!((m.bases["red"].fortress_enemy_hold - 6.0).abs() < 1e-6);
    // Teammate blue-4 relays for remaining 4.0 s (80 ticks) -> opens red base armor!
    place(&mut m, b4, rf[0], rf[1]);
    for _ in 0..80 {
        m.update_buffs_2027();
        m.tick += 1;
    }
    assert!(m.bases["red"].armor_open);
}

#[test]
fn large_rune_timeout_activation_increments_shrink_ring_count() {
    let mut m = match_();
    m.tick = 3600; // t = 180 s (large rune phase)
    m.runes.get_mut("red").unwrap().large_chances = 2;
    m.start_rune_2027("red");
    let now = m.time();
    {
        let r = m.runes.get_mut("red").unwrap();
        r.rounds = 4; // on final 5th group
        r.hit_count = 5;
        r.ring_sum = 40.0;
        r.pair_deadline = now; // 1 s pair window expires right now
    }
    m.update_rune_motor_2027();
    assert_eq!(m.runes["red"].status, "active");
    assert_eq!(
        m.rune_large_count.get("red").copied().unwrap_or(0),
        1,
        "Timeout completion of 5th group must still increment rune_large_count for Table 5-24 ring shrinking"
    );
}

#[test]
fn assembly_leave_two_second_delay_keeps_weapon_locked_with_invulnerability() {
    let mut m = match_();
    let h = robot(&m, "red-1");
    m.robots[h].launcher = "installed".into();
    m.robots[h].ammo = 10;
    let az = zone_center(&m, "red_assembly");
    place(&mut m, h, az[0], az[1]);
    m.assemblies.get_mut("red").unwrap().stage = "inserting".into();
    m.update_buffs_2027();
    assert!(m.robot_invulnerable_2027(h));
    assert_eq!(m.fire_block_reason_2027(h), Some("assembly"));
    // Move heavy off the assembly card; 1 s later (within 2 s release delay) BOTH invulnerability and fire lock hold!
    place(&mut m, h, 5.0, 7.5);
    m.tick = 20;
    m.update_buffs_2027();
    assert!(m.robot_invulnerable_2027(h));
    assert_eq!(m.fire_block_reason_2027(h), Some("assembly"));
    // After 2 s release delay expires, both clear together.
    m.tick = 41;
    m.update_buffs_2027();
    assert!(!m.robot_invulnerable_2027(h));
    assert_eq!(m.fire_block_reason_2027(h), None);
}

#[test]
fn yellow_card_exempts_sentry_fpv_and_keeps_cutoff_independent_from_overpower() {
    let mut m = match_();
    let r3 = robot(&m, "red-3");
    let r7 = robot(&m, "red-7");
    m.tick = 20; // t = 1.0 s
    // Teammate yellow card blocks red-3 FPV (5s) but MUST NOT block automatic sentry red-7 FPV.
    m.step(&json!({"referee": [{"kind": "yellow", "robot": "red-3", "reason": "test"}]}));
    assert!(m.robots[r3].fpv_blocked_until > m.time());
    assert_eq!(m.robots[r7].fpv_blocked_until, 0.0);
    // yellow_both must also spare automatic sentry FPV.
    m.step(&json!({"referee": [{"kind": "yellow_both", "reason": "test"}]}));
    assert_eq!(m.robots[r7].fpv_blocked_until, 0.0);
    // Sentry own yellow card (2s chassis off) must NOT stack onto active 5s overpower cutoff.
    let now = m.time();
    m.robots[r7].power_cutoff_until = now + 5.0;
    m.step(&json!({"referee": [{"kind": "yellow", "robot": "red-7", "reason": "test"}]}));
    assert!((m.robots[r7].power_cutoff_until - (now + 5.0)).abs() < 1e-6);
    assert!((m.robots[r7].penalty_power_cutoff_until - (now + 2.0)).abs() < 1e-6);
    assert_eq!(m.robots[r7].fpv_blocked_until, 0.0);
}

#[test]
fn forfeit_does_not_inflate_winner_attack_damage() {
    let mut m = match_();
    let blue_before = m.teams["blue"].attack_damage;
    m.step(&json!({"referee": [{"kind": "forfeit", "team": "red", "reason": "test"}]}));
    assert_eq!(m.status, "finished");
    assert_eq!(m.winner.as_deref(), Some("blue"));
    assert_eq!(m.bases["red"].hp, 0.0);
    assert_eq!(m.teams["blue"].attack_damage, blue_before);
}

#[test]
fn drones_blocked_from_dart_station_airspace_but_allowed_over_enemy_base() {
    let mut m = match_();
    let air = robot(&m, "red-6");
    // Place drone just outside red dart_station [0..2.3, 0..1.3] and fly toward (1.0, 0.6).
    let start = [2.6, 0.6, 1.5];
    m.robots[air].drones[0].state = "flying".into();
    m.robots[air].drones[0].x = start[0];
    m.robots[air].drones[0].y = start[1];
    m.robots[air].drones[0].z = start[2];
    m.physics.as_mut().unwrap().place_drone(air, 0, start, [-4.0, 0.0, 0.0]);
    for _ in 0..20 {
        m.step(&json!({"red": {"6": {"drones": {"1": {"goto": [1.0, 0.6, 1.5]}}}}}));
    }
    let d = &m.robots[air].drones[0];
    assert!(
        d.x >= 2.3,
        "Drone entered dart_station forbidden airspace: ({}, {})",
        d.x,
        d.y
    );
    // Verify drone CAN enter blue base forbidden zone [24.639..26.539, 6.65..8.35] (R56 is ground-only).
    let base_start = [24.4, 7.5, 1.5];
    m.robots[air].drones[0].x = base_start[0];
    m.robots[air].drones[0].y = base_start[1];
    m.robots[air].drones[0].z = base_start[2];
    m.physics.as_mut().unwrap().place_drone(air, 0, base_start, [3.0, 0.0, 0.0]);
    for _ in 0..15 {
        m.step(&json!({"red": {"6": {"drones": {"1": {"goto": [25.2, 7.5, 1.5]}}}}}));
    }
    assert!(
        m.robots[air].drones[0].x > 24.7,
        "Drone must be allowed over enemy base to perform aerial strikes: x={}",
        m.robots[air].drones[0].x
    );
}

#[test]
fn non_transforming_model_7_cannot_crouch_under_tunnels() {
    let mut m = Engine::new_with_lineups(
        42,
        60.0,
        "rmuc2027",
        lineup(&[(1, 2711), (3, 7), (4, 2732), (6, 2761), (7, 2771)]),
    )
    .unwrap();
    let i = robot(&m, "red-3");
    let nominal_muzzle = m.robots[i].muzzle_height_m;
    for _ in 0..10 {
        m.step(&json!({"red": {"3": {"crouch": true}}}));
    }
    assert!(!m.robots[i].crouched);
    assert!((m.robots[i].muzzle_height_m - nominal_muzzle).abs() < 1e-9);
}

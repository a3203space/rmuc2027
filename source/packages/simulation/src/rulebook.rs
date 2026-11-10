use super::*;

#[derive(Clone, Serialize, Deserialize)]
pub struct RuneState {
    pub status: String,
    pub mode: String,
    pub small_chances: u32,
    pub large_chances: u32,
    pub started: f64,
    pub active_until: f64,
    pub hit_deadline: f64,
    pub pair_deadline: f64,
    pub lit: Vec<usize>,
    pub activated_arms: Vec<usize>,
    pub rounds: u32,
    pub hit_count: u32,
    pub ring_sum: f64,
    pub attack: f64,
    pub defense: f64,
    pub cooling: f64,
    #[serde(default)]
    pub last_hit_at: [Option<f64>; 5],
    #[serde(default)]
    pub last_hit_valid: [bool; 5],
}
impl RuneState {
    pub fn new() -> Self {
        Self {
            status: "idle".into(),
            mode: "small".into(),
            small_chances: 1,
            large_chances: 0,
            started: 0.0,
            active_until: 0.0,
            hit_deadline: 0.0,
            pair_deadline: 0.0,
            lit: Vec::new(),
            activated_arms: Vec::new(),
            rounds: 0,
            hit_count: 0,
            ring_sum: 0.0,
            attack: 1.0,
            defense: 0.0,
            cooling: 1.0,
            last_hit_at: [None; 5],
            last_hit_valid: [false; 5],
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct AssemblyState {
    pub difficulty: u32,
    pub stage: String,
    pub started: f64,
    pub ready_at: f64,
    pub retry_at: f64,
    pub permanent_lock: bool,
    pub attempt_by_difficulty: BTreeMap<u32, u32>,
    /// Successful assemblies seed the next pose; failed retries keep the same pose.
    #[serde(default)]
    pub successes_by_difficulty: BTreeMap<u32, u32>,
    pub inserted: f64,
    pub translated: f64,
    pub rotation: f64,
    pub target_rotation: f64,
    pub inserted_units: u32,
    pub first_unit_at: f64,
    pub attempt: u32,
    pub pose: [f64; 6],
    pub rebuild_hold: BTreeMap<String, f64>,
    pub interrupted: bool,
    /// Displayed core pose, moving at a finite rate toward `pose` or the initial pose.
    #[serde(default)]
    pub current_pose: [f64; 6],
    #[serde(default)]
    pub confirm_progress_s: f64,
    #[serde(default)]
    pub buffer_until: f64,
    #[serde(default)]
    pub repeat_ready_at: f64,
    /// 2027 steps 3-6 on the own core, and the borrowed core of a level-4 attempt.
    #[serde(default, skip_serializing_if = "is_zero_f")]
    pub lifted: f64,
    #[serde(default, skip_serializing_if = "is_zero_f")]
    pub flipped: f64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub confirmed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second: Option<season2027::CoreProgress>,
    #[serde(default, skip_serializing_if = "no_steps")]
    pub step_times: [Option<f64>; 5],
}
fn is_zero_f(v: &f64) -> bool {
    *v == 0.0
}
fn no_steps(v: &[Option<f64>; 5]) -> bool {
    v.iter().all(Option::is_none)
}
impl AssemblyState {
    pub fn core(&self, core: usize) -> season2027::CoreProgress {
        if core == 1 {
            return self.second.clone().unwrap_or_default();
        }
        season2027::CoreProgress {
            inserted: self.inserted,
            lifted: self.lifted,
            flipped: self.flipped,
            rotation: self.rotation,
            target_rotation: self.target_rotation,
            confirm: self.confirm_progress_s,
            confirmed: self.confirmed,
            step_times: self.step_times,
        }
    }
    pub fn set_core(&mut self, core: usize, c: season2027::CoreProgress) {
        if core == 1 {
            self.second = Some(c);
            return;
        }
        self.inserted = c.inserted;
        self.lifted = c.lifted;
        self.flipped = c.flipped;
        self.rotation = c.rotation;
        self.target_rotation = c.target_rotation;
        self.confirm_progress_s = c.confirm;
        self.confirmed = c.confirmed;
        self.step_times = c.step_times;
    }
    pub fn new() -> Self {
        Self {
            difficulty: 0,
            stage: "idle".into(),
            started: 0.0,
            ready_at: 0.0,
            retry_at: 0.0,
            permanent_lock: false,
            attempt_by_difficulty: BTreeMap::new(),
            successes_by_difficulty: BTreeMap::new(),
            inserted: 0.0,
            translated: 0.0,
            rotation: 0.0,
            target_rotation: 0.0,
            inserted_units: 0,
            first_unit_at: 0.0,
            attempt: 0,
            pose: [-0.2, 0.2, 0.8, 0.0, 90.0, 0.0],
            rebuild_hold: BTreeMap::new(),
            interrupted: false,
            current_pose: [-0.2, 0.2, 0.8, 0.0, 90.0, 0.0],
            confirm_progress_s: 0.0,
            buffer_until: 0.0,
            repeat_ready_at: 0.0,
            lifted: 0.0,
            flipped: 0.0,
            confirmed: false,
            second: None,
            step_times: [None; 5],
        }
    }
}
fn n(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or(0.0)
}
/// `x.sin()` kept out of line: beside `x.cos()` LLVM fuses the pair into Apple's
/// `__sincos_stret`, which rounds the last bit differently and would move rune targets.
#[inline(never)]
fn unfused_sin(x: f64) -> f64 {
    x.sin()
}

fn opponent(team: &str) -> &str {
    if team == "red" {
        "blue"
    } else {
        "red"
    }
}

impl Engine {
    pub fn engine_hash(&self) -> String {
        // The embedded sources never change at runtime; hashing them per frame cost ~10% of a step.
        static HASH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        HASH.get_or_init(|| {
            let mut h = Sha256::new();
            h.update(include_bytes!("lib.rs"));
            h.update(include_bytes!("rulebook.rs"));
            h.update(include_bytes!("physics.rs"));
            h.update(include_bytes!("penalties.rs"));
            h.update(include_bytes!("structures.rs"));
            h.update(include_bytes!("pickup.rs"));
            h.update(include_bytes!("season2027.rs"));
            h.update(include_bytes!("vision.rs"));
            h.update(include_bytes!("../../../Cargo.lock"));
            format!("{:x}", h.finalize())
        })
        .clone()
    }

    pub(super) fn raw_zone(&self, i: usize, kind: &str, owner: Option<&str>) -> bool {
        self.raw_zone_2027(i, kind, owner)
    }

    fn gain_xp(&mut self, i: usize, value: f64) {
        self.gain_xp_2027(i, value);
    }

    pub(super) fn dispersed_shot_direction(&mut self, direction: [f64; 3], role: &str) -> [f64; 3] {
        let caliber = if role == "heavy" { "42mm" } else { "17mm" };
        let half_angle = n(&self.sim["projectile_spread_half_angle_deg"], caliber).to_radians();
        if half_angle <= 0.0 {
            return direction;
        }
        // Uniform solid-angle cone about the actual (possibly tilted) muzzle axis.
        // Angular spread gives a footprint proportional to range, without changing
        // muzzle speed or adding artificial sideways forces during flight.
        let cosine = 1.0 - self.random() * (1.0 - half_angle.cos());
        let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
        let azimuth = self.random() * std::f64::consts::TAU;
        let axis = rapier3d::na::Vector3::from(direction).normalize();
        let reference = if axis.z.abs() < 0.9 {
            rapier3d::na::Vector3::z()
        } else {
            rapier3d::na::Vector3::x()
        };
        let u = axis.cross(&reference).normalize();
        let v = axis.cross(&u);
        let shot = axis * cosine + (u * azimuth.cos() + v * azimuth.sin()) * sine;
        [shot.x, shot.y, shot.z]
    }

    fn arm_design(&self) -> Value {
        self.robots
            .iter()
            .position(|r| r.arm_count > 0)
            .and_then(|i| self.physics.as_ref().map(|w| w.design(i)["arm"].clone()))
            .unwrap_or(Value::Null)
    }
    /// World position of the core entry centre E and its unit normal e (5.3.3).
    pub(super) fn core_entry(&self, team: &str) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
        let frame = &self.arena.assembly_frames[team];
        let v = |key: &str, k: usize| frame[key][k].as_f64();
        let (ox, oy, xx, xy, yx, yy) = (
            v("origin", 0)?,
            v("origin", 1)?,
            v("x_axis", 0)?,
            v("x_axis", 1)?,
            v("y_axis", 0)?,
            v("y_axis", 1)?,
        );
        let [x, y, z, theta, phi, _] = self.assemblies[team].current_pose;
        let floor = oy.min(self.arena.height - oy) * self.arena.floor_slope_deg.to_radians().tan();
        let (theta, phi) = (theta.to_radians(), phi.to_radians());
        let (ex, ey, ez) = (phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());
        let normal = [ex * xx + ey * yx, ex * xy + ey * yy, ez];
        // The unit slides parallel to the entry face, horizontally where possible.
        let mut slide = [-normal[1], normal[0], 0.0];
        let length = slide[0].hypot(slide[1]);
        if length < 1e-6 {
            slide = [yx, yy, 0.0];
        } else {
            slide = [slide[0] / length, slide[1] / length, 0.0];
        }
        Some((
            [
                ox + x * xx + y * yx,
                oy + x * xy + y * yy,
                floor + n(frame, "z0_m") + z,
            ],
            normal,
            slide,
        ))
    }
    /// Advance each core toward its commanded pose and derive the heavy robots' arm goals.
    pub(super) fn arm_goals(&mut self, actions: &[Action]) -> Vec<physics::ArmGoal> {
        let initial = [-0.2, 0.2, 0.8, 0.0, 90.0, 0.0];
        for team in ["red", "blue"] {
            let s = self.assemblies.get_mut(team).unwrap();
            let target = if ["moving", "inserting", "buffered"].contains(&s.stage.as_str()) {
                s.pose
            } else {
                initial
            };
            for k in 0..6 {
                let rate = if k < 3 { 0.25 } else { 90.0 };
                s.current_pose[k] = approach(s.current_pose[k], target[k], rate * DT);
            }
        }
        let mut goals = vec![physics::ArmGoal::default(); self.robots.len()];
        for i in 0..self.robots.len() {
            let r = &self.robots[i];
            if r.arm_count == 0 {
                continue;
            }
            let rpc = &actions[i].rpc;
            let team = r.team.clone();
            let goal = &mut goals[i];
            goal.mode = "stow".into();
            let rescue = match rpc.get("rescue") {
                Some(Value::Number(slot)) => slot.as_u64().map(|s| format!("{team}-{s}")),
                Some(Value::String(id)) => Some(id.clone()),
                _ => None,
            };
            if let Some(id) = rescue {
                let target = self.robots.iter().position(|o| {
                    o.id == id && o.team == team && o.role != "aerial" && o.id != r.id
                });
                if let (Some(j), true) = (target, r.alive && !r.weak) {
                    goal.mode = "rescue".into();
                    goal.target = Some(j);
                }
            }
            // Direct arm control: joint targets, or a tool-tip pose in the world/robot frame.
            if let (true, Some(cmd)) = (
                goal.mode == "stow",
                rpc.get("arm").filter(|v| v.is_object()),
            ) {
                let vector = |key: &str, len: usize| {
                    cmd.get(key)
                        .and_then(Value::as_array)
                        .filter(|v| v.len() == len)
                        .and_then(|v| {
                            v.iter()
                                .map(|x| x.as_f64().filter(|x| x.is_finite()))
                                .collect::<Option<Vec<f64>>>()
                        })
                };
                if cmd.get("stow").and_then(Value::as_bool) == Some(true) {
                    goal.mode = "manual".into();
                    goal.joints = Some(physics::arm_stowed(&self.arm_design()));
                } else if let Some(q) = vector("joints", 7) {
                    goal.mode = "manual".into();
                    goal.joints = Some(std::array::from_fn(|k| q[k]));
                } else if let Some(tip) = vector("tip", 3) {
                    let local = cmd.get("frame").and_then(Value::as_str) == Some("robot");
                    let dir = vector("dir", 3).unwrap_or_else(|| {
                        if local {
                            vec![1.0, 0.0, 0.0]
                        } else {
                            vec![r.heading.cos(), r.heading.sin(), 0.0]
                        }
                    });
                    goal.mode = "manual".into();
                    goal.point = Some([tip[0], tip[1], tip[2]]);
                    goal.dir = Some([dir[0], dir[1], dir[2]]);
                    goal.local = local;
                    goal.roll = cmd
                        .get("roll")
                        .and_then(Value::as_f64)
                        .filter(|v| v.is_finite())
                        .unwrap_or(0.0);
                    goal.grip = cmd
                        .get("grip")
                        .and_then(Value::as_f64)
                        .filter(|v| v.is_finite())
                        .unwrap_or(0.0);
                }
            }
            let state = &self.assemblies[&team];
            let mut secondary_core = None;
            {
                // 2027: the tool follows the selected core's step progress; a dual-arm heavy
                // robot works the borrowed level-4 core with its second arm.
                if goal.mode == "stow"
                    && ["moving", "inserting"].contains(&state.stage.as_str())
                    && self.raw_zone(i, "assembly", Some(&team))
                {
                    let choice = rpc
                        .get("assembly_core")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    let core = if state.difficulty == 4 && r.arm_count < 2 {
                        choice.min(1) as usize
                    } else {
                        0
                    };
                    if let Some((point, dir, roll)) = self.core_goal_2027(&team, core) {
                        goal.mode = "assembly".into();
                        goal.point = Some(point);
                        goal.dir = Some(dir);
                        goal.roll = roll;
                    }
                    if state.difficulty == 4 && r.arm_count == 2 {
                        secondary_core = self.core_goal_2027(&team, 1);
                    }
                }
            }
            let mut secondary = physics::ArmGoal::default();
            secondary.mode = "stow".into();
            if let Some((point, dir, roll)) = secondary_core {
                secondary.mode = "assembly".into();
                secondary.point = Some(point);
                secondary.dir = Some(dir);
                secondary.roll = roll;
            }
            if r.arm_count == 2 {
                if let Some(cmd) = rpc.get("arm2").filter(|v| v.is_object()) {
                    if let Some(g) = physics::manual_arm_goal(
                        cmd,
                        r,
                        &self.physics.as_ref().unwrap().design(i)["arm2"],
                    ) {
                        secondary = g;
                    }
                }
            }
            if let Some(pickup) = &r.unit_pickup {
                let selected = if pickup.arm == 1 {
                    &mut secondary
                } else {
                    &mut *goal
                };
                if selected.mode == "stow" {
                    selected.mode = "pick".into();
                    if pickup.stage == "stowing" {
                        selected.joints = Some(physics::arm_stowed(
                            &self.physics.as_ref().unwrap().design(i)
                                [if pickup.arm == 0 { "arm" } else { "arm2" }],
                        ));
                    } else {
                        selected.point = Some([
                            pickup.point[0],
                            pickup.point[1],
                            pickup.point[2] + if pickup.stage == "lifting" { 0.18 } else { 0.0 },
                        ]);
                        selected.dir = Some(pickup.direction);
                        selected.grip = if pickup.stage == "approaching" {
                            1.0
                        } else {
                            0.0
                        };
                    }
                }
            }
            if r.arm_count == 2 {
                goal.secondary = Some(Box::new(secondary));
            }
            let target = goal.target.map(|j| self.robots[j].id.clone());
            self.robots[i].rescue_target = target;
        }
        goals
    }
    pub(super) fn assembly_difficulty(&self, difficulty: u32) -> &Value {
        &self.sim["assembly_difficulties"][difficulty.saturating_sub(1).min(3) as usize]
    }
    pub(super) fn assembly_motion_blocked(&self, i: usize) -> bool {
        let r = &self.robots[i];
        let profile = self.assembly_difficulty(self.assemblies[&r.team].difficulty);
        r.tipped
            || r.upright < n(profile, "min_upright")
            || (r.vx * r.vx + r.vy * r.vy + r.vz * r.vz).sqrt() > n(profile, "chassis_speed_mps")
            || r.angular_velocity.iter().map(|v| v * v).sum::<f64>().sqrt()
                > n(profile, "chassis_angular_speed_rad_s")
    }

    // Progress commands alone cannot exchange a unit: the real tool must reach and face the core.

    pub(super) fn start_rune(&mut self, team: &str) {
        let now = self.time();
        let mode = if now < 180.0 { "small" } else { "large" };
        let s = &self.runes[team];
        if s.status == "active" || s.status == "activating" {
            return;
        }
        let chances = if mode == "small" {
            s.small_chances
        } else {
            s.large_chances
        };
        if chances == 0 {
            return;
        }
        let first = (self.random() * 5.0) as usize % 5;
        let second = (first + 1 + (self.random() * 4.0) as usize) % 5;
        let s = self.runes.get_mut(team).unwrap();
        s.mode = mode.into();
        s.status = "activating".into();
        s.started = now;
        s.hit_deadline = now + 2.5;
        s.pair_deadline = 0.0;
        s.hit_count = 0;
        s.rounds = 0;
        s.activated_arms.clear();
        s.last_hit_at = [None; 5];
        s.last_hit_valid = [false; 5];
        s.ring_sum = 0.0;
        s.lit = if mode == "small" {
            vec![first]
        } else {
            vec![first, second]
        };
        if mode == "small" {
            s.small_chances -= 1;
        } else {
            s.large_chances -= 1;
        }
        self.event("rune_start", json!({"team":team,"mode":mode}));
    }
    pub(super) fn update_rune_motor(&mut self) {
        let now = self.time();
        // Integrate the smooth, nonuniform shaft speed over this frame. It keeps
        // running while idle and while buffs are active, on both faces together.
        let delta = self.rune_motor_integral(now, now + DT);
        self.rune_angle =
            (self.rune_angle + delta * self.rune_direction).rem_euclid(std::f64::consts::TAU);
        for team in ["red", "blue"] {
            let state = self.runes[team].clone();
            if state.status == "active" && now >= state.active_until {
                let r = self.runes.get_mut(team).unwrap();
                r.status = "idle".into();
                r.activated_arms.clear();
                r.lit.clear();
                r.last_hit_at = [None; 5];
                r.last_hit_valid = [false; 5];
                r.attack = 1.0;
                r.defense = 0.0;
                r.cooling = 1.0;
            }
            if state.status == "activating" {
                if now - state.started >= 20.0 {
                    let r = self.runes.get_mut(team).unwrap();
                    r.status = "idle".into();
                    r.activated_arms.clear();
                    r.lit.clear();
                } else if state.mode == "large"
                    && state.pair_deadline > 0.0
                    && now >= state.pair_deadline
                {
                    self.next_rune_pair(team);
                } else if state.pair_deadline == 0.0 && now >= state.hit_deadline {
                    self.reset_rune_pattern(team);
                }
            }
        }
    }
    fn rune_motor_rates(&self, now: f64) -> (f64, f64) {
        self.rune_motor_rates_2027(now)
    }
    fn rune_motor_integral(&self, start: f64, end: f64) -> f64 {
        let (a, _) = self.rune_motor_rates(start);
        let (b, _) = self.rune_motor_rates((start + end) * 0.5);
        let (c, _) = self.rune_motor_rates(end);
        (end - start) * (a + 4.0 * b + c) / 6.0
    }
    pub(super) fn assembly_snapshots(&self) -> Value {
        let mut result = serde_json::to_value(&self.assemblies).unwrap();
        for team in ["red", "blue"] {
            self.assembly_snapshot_2027(team, &mut result[team]);
        }
        result
    }

    pub(super) fn base_snapshots(&self) -> Value {
        let mut result = json!(self.bases);
        for team in ["red", "blue"] {
            result[team]["armor_plates"] = json!(structures::plates(&self.bases[team], false));
            {
                // RMUC 2027: every exposed base module accepts 17 mm, 42 mm and aerial impacts.
                for plate in result[team]["armor_plates"].as_array_mut().unwrap() {
                    plate["heavy_only"] = json!(false);
                    plate["weakspot"] = json!(false);
                }
            }
        }
        result
    }
    pub(super) fn outpost_snapshots(&self) -> Value {
        let mut result = serde_json::to_value(&self.outposts).unwrap();
        for team in ["red", "blue"] {
            result[team]["armor_plates"] =
                json!(Self::structure_plates_2027(&self.outposts[team], true));
            for plate in result[team]["armor_plates"].as_array_mut().unwrap() {
                plate["weakspot"] = json!(false);
            }
            result[team]["rebuild"] = Value::Null;
        }
        result
    }
    fn reset_rune_pattern(&mut self, team: &str) {
        let now = self.time();
        let mode = self.runes[team].mode.clone();
        let a = (self.random() * 5.0) as usize % 5;
        let b = (a + 1 + (self.random() * 4.0) as usize) % 5;
        let r = self.runes.get_mut(team).unwrap();
        r.hit_count = 0;
        r.rounds = 0;
        r.activated_arms.clear();
        r.ring_sum = 0.0;
        r.hit_deadline = now + 2.5;
        r.pair_deadline = 0.0;
        r.lit = if mode == "small" { vec![a] } else { vec![a, b] };
        self.event("rune_reset", json!({"team":team}));
    }
    fn next_rune_pair(&mut self, team: &str) {
        self.runes.get_mut(team).unwrap().rounds += 1;
        if self.runes[team].rounds >= 5 {
            self.activate_rune(team);
            return;
        }
        let now = self.time();
        let a = (self.random() * 5.0) as usize % 5;
        let b = (a + 1 + (self.random() * 4.0) as usize) % 5;
        let r = self.runes.get_mut(team).unwrap();
        r.lit = vec![a, b];
        r.hit_deadline = now + 2.5;
        r.pair_deadline = 0.0;
    }
    /// World positions of a team's five rune targets at the current shaft angle.
    pub(super) fn rune_target_points(&self, team: &str) -> [[f64; 3]; 5] {
        let center_x = n(&self.arena.rune, "x");
        let center_y = n(&self.arena.rune, "y");
        let center_z = n(&self.arena.rune, "z");
        let yaw = n(&self.arena.rune, "axis_yaw");
        let (nx, ny) = (yaw.cos(), yaw.sin());
        // Coaxial faces (5.5.2): red lamps face -axis, blue +axis, each face_offset from the hub.
        let offset = self.arena.rune["face_offset_m"].as_f64().unwrap_or(0.025);
        let radius = self.arena.rune["arm_radius_m"].as_f64().unwrap_or(0.7);
        let sign = if team == "red" { 1.0 } else { -1.0 };
        std::array::from_fn(|arm| {
            let angle = self.rune_angle + arm as f64 * std::f64::consts::TAU / 5.0;
            [
                center_x - sign * offset * nx - ny * radius * angle.cos(),
                center_y - sign * offset * ny + nx * radius * angle.cos(),
                center_z + radius * unfused_sin(angle),
            ]
        })
    }
    pub(super) fn rune_snapshots(&self) -> Value {
        let mut result = serde_json::Map::new();
        for team in ["red", "blue"] {
            let r = &self.runes[team];
            let points = self.rune_target_points(team);
            let targets:Vec<Value>=(0..5).map(|arm|{let [x,y,z]=points[arm];json!({"arm":arm,"x":x,"y":y,"z":z,"radius":0.15,"lit":r.lit.contains(&arm),"hit":(r.status=="activating"||r.status=="active")&&r.activated_arms.contains(&arm),"last_hit_at":r.last_hit_at[arm],"last_hit_valid":r.last_hit_valid[arm]})}).collect();
            let mut value = serde_json::to_value(r).unwrap();
            value["targets"] = json!(targets);
            value["angle"] = json!(self.rune_angle);
            let (speed, acceleration) = self.rune_motor_rates(self.time());
            value["angular_velocity"] = json!(speed * self.rune_direction);
            value["angular_acceleration"] = json!(acceleration * self.rune_direction);
            result.insert(team.into(), value);
        }
        Value::Object(result)
    }
    pub(super) fn observed_runes(&self, team: &str, observers: &[&Robot]) -> Value {
        let all = self.rune_snapshots();
        let mut own = all[team].clone();
        let visible = observers.iter().any(|r| {
            dist(r.x, r.y, n(&self.arena.rune, "x"), n(&self.arena.rune, "y"))
                <= self.effective_vision(r)
        });
        if !visible {
            own["targets"] = json!([]);
            own["angle"] = Value::Null;
            own["angular_velocity"] = Value::Null;
            own["angular_acceleration"] = Value::Null;
        }
        json!({team:own})
    }
    pub(super) fn observed_teams(&self, team: &str) -> Value {
        let mut all = serde_json::to_value(&self.teams).unwrap();
        let other = opponent(team);
        all[other] = json!({"attack_damage":self.teams[other].attack_damage,"score":self.teams[other].attack_damage});
        {
            all[other]["control"] = json!(self.teams[other].control);
        }
        all
    }
    pub(super) fn rune_intersection(
        &self,
        p: &Projectile,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Option<(f64, usize)> {
        let yaw = n(&self.arena.rune, "axis_yaw");
        let (nx, ny) = (yaw.cos(), yaw.sin());
        let normal_speed = dx * nx + dy * ny;
        if normal_speed.abs() < 1e-9 {
            return None;
        }
        let sign = if p.team == "red" { 1.0 } else { -1.0 };
        if normal_speed * sign <= 0.0 {
            return None;
        }
        let mut hits = Vec::new();
        for (arm, [x, y, z]) in self.rune_target_points(&p.team).into_iter().enumerate() {
            let t = ((x - p.x) * nx + (y - p.y) * ny) / normal_speed;
            if (0.0..=1.0).contains(&t) {
                let tangent = -(p.x + dx * t - x) * ny + (p.y + dy * t - y) * nx;
                let distance = tangent.hypot(p.z + dz * t - z);
                if distance <= 0.15 + p.radius {
                    hits.push((t, arm));
                }
            }
        }
        hits.into_iter().min_by(|a, b| a.0.total_cmp(&b.0))
    }
    pub(super) fn rulebook_rune_hit(&mut self, p: &Projectile, arm: usize, x: f64, y: f64, z: f64) {
        let now = self.time();
        let yaw = n(&self.arena.rune, "axis_yaw");
        let (nx, ny) = (yaw.cos(), yaw.sin());
        let valid = p.role != "heavy"
            && (p.vx * nx + p.vy * ny).abs() > 12.0
            && self.runes[&p.team].status == "activating"
            && self.runes[&p.team].lit.contains(&arm);
        let feedback = self.runes.get_mut(&p.team).unwrap();
        feedback.last_hit_at[arm] = Some(now);
        feedback.last_hit_valid[arm] = valid;
        self.event(
            "rune_hit",
            json!({"team":p.team,"robot":p.source,"arm":arm,"valid":valid,"x":x,"y":y,"z":z}),
        );
        if p.role == "heavy" {
            self.event(
                "invalid_action",
                json!({"robot":p.source,"reason":"42mm cannot hit the rune"}),
            );
            return;
        }
        if (p.vx * nx + p.vy * ny).abs() <= 12.0 {
            return;
        }
        let team = p.team.clone();
        let state = self.runes[&team].clone();
        if state.status != "activating" {
            return;
        }
        if !state.lit.contains(&arm) {
            self.reset_rune_pattern(&team);
            return;
        }
        let [tx, ty, tz] = self.rune_target_points(&team)[arm];
        let tangent = -(x - tx) * ny + (y - ty) * nx;
        let offset = tangent.hypot(z - tz);
        let ring = (10.0 - (offset / 0.015).floor()).clamp(1.0, 10.0);
        let r = self.runes.get_mut(&team).unwrap();
        r.hit_count += 1;
        r.activated_arms.push(arm);
        r.ring_sum += ring;
        r.lit.retain(|i| *i != arm);
        if state.mode == "small" {
            if r.hit_count >= 5 {
                self.activate_rune(&team);
            } else {
                let done = self.runes[&team].activated_arms.clone();
                let remaining: Vec<usize> = (0..5).filter(|i| !done.contains(i)).collect();
                let selected =
                    remaining[(self.random() * remaining.len() as f64) as usize % remaining.len()];
                let r = self.runes.get_mut(&team).unwrap();
                r.lit = vec![selected];
                r.hit_deadline = now + 2.5;
            }
        } else if state.pair_deadline == 0.0 {
            self.runes.get_mut(&team).unwrap().pair_deadline = now + 1.0;
        } else {
            self.next_rune_pair(&team);
        }
    }
    fn activate_rune(&mut self, team: &str) {
        let now = self.time();
        let state = self.runes[team].clone();
        let small = state.mode == "small";
        let mut attack = 1.0;
        let mut defense = 0.25;
        let mut cooling = 1.0;
        let duration;
        if small {
            duration = 45.0;
            self.teams.get_mut(team).unwrap().small_bonus_xp = 0.0;
        } else {
            let average = state.ring_sum / state.hit_count.max(1) as f64;
            for band in self.rules["rune"]["large_bands"].as_array().unwrap() {
                if average <= n(band, "max") {
                    attack = n(band, "attack");
                    defense = n(band, "defense");
                    cooling = n(band, "cooling");
                    break;
                }
            }
            let count = state.hit_count.clamp(5, 10);
            duration = self.rules["rune"]["duration_by_hits"][count.to_string()]
                .as_f64()
                .unwrap();
        }
        let r = self.runes.get_mut(team).unwrap();
        r.status = "active".into();
        r.active_until = now + duration;
        r.attack = attack;
        r.defense = defense;
        r.cooling = cooling;
        r.lit.clear();
        if !small {
            let recipients: Vec<usize> = self
                .robots
                .iter()
                .enumerate()
                .filter(|(_, r)| r.team == team && r.alive && { r.role == "infantry" })
                .map(|(i, _)| i)
                .collect();
            let amount = 750.0 / recipients.len().max(1) as f64;
            for i in recipients {
                self.gain_xp(i, amount);
            }
        }
        self.event("rune_activated",json!({"team":team,"mode":state.mode,"duration":duration,"attack":attack,"defense":defense,"cooling":cooling}));
    }

    pub(super) fn public_rule_summary(&self) -> Value {
        self.public_rule_summary_2027()
    }
}

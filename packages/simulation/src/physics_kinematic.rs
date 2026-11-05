//! Validated motion-frame transport. Combat continues at the original timestep.
use super::*;

#[derive(serde::Deserialize)]
pub(crate) struct MotionFrame {
    op: String,
    version: u32,
    sequence: u64,
    dt: f64,
    ids: Vec<String>,
    pub positions: Vec<Vec<[f64; 3]>>,
    pub velocities: Vec<Vec<[f64; 3]>>,
    pub headings: Vec<Vec<f64>>,
    pub folded: Vec<Vec<f64>>,
    pub grounded: Vec<Vec<bool>>,
}

impl MotionFrame {
    pub(crate) fn parse(value: Value, sequence: u64, ids: &[String]) -> Result<Self, String> {
        let frame: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        if frame.op != "kinematic_frame_result" || frame.version != 1
            || frame.sequence != sequence || frame.ids != ids
            || !frame.dt.is_finite() || (frame.dt - PHYSICS_DT).abs() > 1e-12 {
            return Err("Kinematic frame identity or timestep mismatch".into());
        }
        let sizes = [frame.positions.iter().map(Vec::len).collect::<Vec<_>>(),
                     frame.velocities.iter().map(Vec::len).collect(),
                     frame.headings.iter().map(Vec::len).collect(),
                     frame.folded.iter().map(Vec::len).collect(),
                     frame.grounded.iter().map(Vec::len).collect()];
        if ids.is_empty() || sizes.iter().any(|s| s.len() != 6 || s.iter().any(|n| *n != ids.len())) {
            return Err("Expected six complete robot samples".into());
        }
        if frame.positions.iter().flatten().flatten().chain(frame.velocities.iter().flatten().flatten())
            .chain(frame.headings.iter().flatten()).chain(frame.folded.iter().flatten())
            .any(|v| !v.is_finite()) || frame.folded.iter().flatten().any(|v| !(0.0..=1.0).contains(v)) {
            return Err("Invalid kinematic numeric state".into());
        }
        Ok(frame)
    }
}

impl PhysicsWorld {
    /// One transport exchange per rule frame, instead of one per physics substep.
    pub(crate) fn request_motion_frame(&self, arena: &Arena, robots: &[Robot], actions: &[Action],
                                      sequence: u64, now: f64) -> MotionFrame {
        use std::io::{BufRead, Write};
        let entries: Vec<Value> = robots.iter().zip(actions).enumerate().map(|(i,(r,a))| {
            json!({"id":r.id,"model":r.model_slot,"team":r.team,"role":r.role,
                "pos":[r.x,r.y,r.z],"velocity":[r.vx,r.vy,r.vz],"heading":r.heading,
                "folded":r.transform_progress,"grounded":r.grounded,
                "enabled":r.alive && now>=r.power_cutoff_until && r.battery_remaining_wh>0.0 && !r.deployed,
                "alive":r.alive,"radius":r.radius,"power_limit":r.power_limit,
                "motor_budget":a.power.unwrap_or(r.power_limit).clamp(0.,200.).min(r.power_limit+
                    if r.supercap_energy_j>0. {n(&self.designs[i],"voltage_v")*n(&self.designs[i],"supercap_max_current_a")} else {0.}),
                "fly_pass":r.fly_pass,
                "move":a.mv,"heading_goal":a.heading,"yaw_rate":a.yaw_rate,
                "transform":a.transform.unwrap_or(a.crouch),"jump":a.jump,
                "coast":a.coast,"design":if sequence==0 {self.designs[i].clone()} else {Value::Null}})
        }).collect();
        let request=json!({"op":"kinematic_frame","version":1,"sequence":sequence,
            "dt":PHYSICS_DT,"substeps":6,"time":now,"robots":entries,
            "arena":if sequence==0 {serde_json::to_value(arena).unwrap()} else {Value::Null}});
        {
            let stdout=std::io::stdout();let mut out=stdout.lock();
            serde_json::to_writer(&mut out,&request).expect("kinematic frame write");
            out.write_all(b"\n").unwrap();out.flush().unwrap();
        }
        let mut line=String::new();
        std::io::stdin().lock().read_line(&mut line).expect("kinematic frame read");
        let value=serde_json::from_str(&line).expect("kinematic frame JSON");
        MotionFrame::parse(value,sequence,&robots.iter().map(|r|r.id.clone()).collect::<Vec<_>>())
            .expect("invalid kinematic frame")
    }

    pub(crate) fn apply_motion_sample(&mut self, robots: &mut [Robot], actions: &[Action],
                                     frame: &MotionFrame, sample: usize, now: f64, arena: &Arena) {
        for (i,r) in robots.iter_mut().enumerate() {
            let d=&self.designs[i];let h=self.handles[i];let a=&actions[i];
            // A death or power cutoff caused by an earlier combat sample wins
            // over all remaining predicted poses in this frame.
            r.powered=r.alive && now>=r.power_cutoff_until && r.battery_remaining_wh>0.0;
            let previous_heading=r.heading;
            let previous_speed=r.vx.hypot(r.vy);
            if r.powered && !r.deployed && r.role!="aerial" {
                let p=frame.positions[sample][i];let v=frame.velocities[sample][i];
                r.x=p[0];r.y=p[1];r.z=p[2];r.vx=v[0];r.vy=v[1];r.vz=v[2];
                r.heading=frame.headings[sample][i];r.grounded=frame.grounded[sample][i];
                if arena.forbidden.iter().any(|z| forbidden_for(z,&r.team,r.fly_pass) && in_zone_rect(z,r.x,r.y,0.0)) {
                    r.x=self.last[i][0];r.y=self.last[i][1];r.z=self.last[i][2];
                    r.vx=0.;r.vy=0.;r.vz=0.;
                }
            } else {
                r.vx=0.0;r.vy=0.0;r.vz=0.0;
            }
            let heading_rate=((r.heading-previous_heading+std::f64::consts::PI)
                .rem_euclid(std::f64::consts::TAU)-std::f64::consts::PI)/PHYSICS_DT;
            let rotation=UnitQuaternion::from_axis_angle(&Vector::z_axis(),r.heading);
            let pose=Isometry::from_parts(Translation::new(r.x,r.y,r.z),rotation);
            self.bodies[h].set_position(pose,true);
            self.bodies[h].set_linvel(vector![r.vx,r.vy,r.vz],true);
            self.bodies[h].set_angvel(vector![0.0,0.0,heading_rate],true);
            self.bodies[h].reset_forces(true);self.bodies[h].reset_torques(true);
            let rot=rotation.quaternion();r.quaternion=[rot.i,rot.j,rot.k,rot.w];
            r.angular_velocity=[0.0,0.0,heading_rate];
            r.upright=1.0;r.tipped=false;r.self_righting=false;
            advance_weapon_clock(r,a,d,&pose);
            let disabled=!r.powered || r.deployed || r.role=="aerial";
            let budget=a.power.unwrap_or(r.power_limit).clamp(0.,200.).min(r.power_limit+
                if r.supercap_energy_j>0. {n(d,"voltage_v")*n(d,"supercap_max_current_a")} else {0.});
            let speed=r.vx.hypot(r.vy);
            let kinetic=0.5*r.mass_kg*(speed*speed-previous_speed*previous_speed).max(0.)/PHYSICS_DT;
            let drag=(n(d,"rolling_resistance")*r.mass_kg*9.81+0.5*1.225*n(d,"drag_area_m2")*speed*speed)*speed;
            r.motor_power_w=if disabled {0.} else {(3.+(kinetic+drag)/n(d,"drive_efficiency").max(0.1)+heading_rate.abs()*2.).min(budget)};
            let leg_power=if r.powered && d["chassis_type"]=="wheel_legged" {
                n(d,"leg_holding_power_w")+if !r.grounded {r.mass_kg*9.81*r.vz.max(0.)/0.75} else {0.}
            } else {0.};
            account_drive_energy(r,d,disabled,leg_power);
            r.charging_power_w=self.charge_power(arena,r,d,&pose);
            r.supercap_energy_j=(r.supercap_energy_j+r.charging_power_w*PHYSICS_DT*0.9).min(n(d,"supercap_nominal_j"));
            if r.transform_allowed && r.powered {
                let progress=frame.folded[sample][i];
                if (progress-r.transform_progress).abs()>1e-8 {
                    r.transform_progress=progress;
                    r.muzzle_height_m=n(d,"muzzle_z_m")+(n(d,"folded_muzzle_z_m")-n(d,"muzzle_z_m"))*progress;
                    r.crouched=progress>0.999;
                    let mut folded=d.clone();folded["muzzle_z_m"]=json!(r.muzzle_height_m);
                    self.colliders[self.hulls[i]].set_shape(Self::hull(&folded,false,None,true,None).build().shared_shape().clone());
                }
                r.pitch=r.pitch.clamp(-0.261799,0.785398+(0.139626-0.785398)*r.transform_progress);
            }
            self.last[i]=[r.x,r.y,r.z];self.last_alive[i]=r.alive;
        }
        self.bodies.propagate_modified_body_positions_to_colliders(&mut self.colliders);
        self.queries.update(&self.colliders);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Value {
        json!({"op":"kinematic_frame_result","version":1,"sequence":42,"dt":PHYSICS_DT,
            "ids":["red-3"],"positions":vec![vec![[0.,0.,0.]];6],
            "velocities":vec![vec![[0.,0.,0.]];6],"headings":vec![vec![0.];6],
            "folded":vec![vec![0.];6],"grounded":vec![vec![true];6]})
    }
    #[test]
    fn death_freezes_remaining_prediction_and_weapon_clock_advances() {
        let mut engine=Engine::new(42,420.0,"rmuc2027").unwrap();
        let count=engine.robots.len();
        let ids=engine.robots.iter().map(|r|r.id.clone()).collect::<Vec<_>>();
        let positions=engine.robots.iter().map(|r|[r.x,r.y,r.z]).collect::<Vec<_>>();
        let headings=engine.robots.iter().map(|r|r.heading).collect::<Vec<_>>();
        let value=json!({"op":"kinematic_frame_result","version":1,"sequence":0,"dt":PHYSICS_DT,
            "ids":ids,"positions":vec![positions;6],"velocities":vec![vec![[1.,0.,0.];count];6],
            "headings":vec![headings;6],"folded":vec![vec![0.;count];6],"grounded":vec![vec![true;count];6]});
        let mut frame=MotionFrame::parse(value,0,&ids).unwrap();
        let index=engine.robots.iter().position(|r|r.id=="red-3").unwrap();
        let mut actions=vec![Action::default();count];actions[index].aim=Some(1.);
        engine.robots[index].cooldown=0.2;
        let mut world=engine.physics.take().unwrap();
        world.apply_motion_sample(&mut engine.robots,&actions,&frame,0,0.,&engine.arena);
        assert!((engine.robots[index].cooldown-(0.2-PHYSICS_DT)).abs()<1e-10);
        assert!(engine.robots[index].aim>0.);
        let position=[engine.robots[index].x,engine.robots[index].y,engine.robots[index].z];
        engine.robots[index].alive=false;
        frame.positions[1][index][0]+=5.;
        world.apply_motion_sample(&mut engine.robots,&actions,&frame,1,PHYSICS_DT,&engine.arena);
        assert_eq!([engine.robots[index].x,engine.robots[index].y,engine.robots[index].z],position);
        assert_eq!(engine.robots[index].vx,0.);
    }

    #[test]
    fn energy_accounting_keeps_original_capacitor_units() {
        let mut engine=Engine::new(42,420.0,"rmuc2027").unwrap();
        let i=engine.robots.iter().position(|r|r.id=="red-3").unwrap();
        let d=engine.physics.as_ref().unwrap().design(i).clone();
        let r=&mut engine.robots[i];
        r.power_limit=50.;r.motor_power_w=200.;r.supercap_energy_j=1.;
        let battery=r.battery_remaining_wh;
        account_drive_energy(r,&d,false,18.);
        assert!(r.supercap_energy_j.abs()<1e-10);
        assert!((r.power_w-80.).abs()<1e-9);
        assert!((battery-r.battery_remaining_wh-98.*PHYSICS_DT/3600.).abs()<1e-10);
    }

    #[test]
    fn validates_entire_frame_before_application() {
        let ids = vec!["red-3".to_string()];
        assert!(MotionFrame::parse(fixture(),42,&ids).is_ok());
        assert!(MotionFrame::parse(fixture(),43,&ids).is_err());
        for (field,value) in [("positions",json!([])),("ids",json!(["blue-3"])),
                              ("grounded",json!(vec![vec![1];6])),("dt",json!(0.05)),
                              ("folded",json!(vec![vec![2.];6]))] {
            let mut bad=fixture();bad[field]=value;
            assert!(MotionFrame::parse(bad,42,&ids).is_err(),"{field}");
        }
    }
}

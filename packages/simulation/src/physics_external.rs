//! Training-only external integrator protocol. Rapier remains the query mirror.
use super::*;
use std::io::{BufRead, Write};

fn pose(p: &Isometry<Real>) -> Value {
    let q = p.rotation.quaternion();
    json!({"pos":[p.translation.x,p.translation.y,p.translation.z],"quat":[q.w,q.i,q.j,q.k]})
}
impl PhysicsWorld {
    pub(super) fn external_step(&mut self, robots: &[Robot]) {
        let initial = self.external_colliders.is_empty();
        let mut changed = Vec::new();
        let mut dirty_bodies = Vec::new();
        let mut live = Vec::new();
        for (h,c) in self.colliders.iter() {
            let id = h.into_raw_parts().0;
            live.push(id);
            let descriptor = json!({"id":id,"body":c.parent().map(|p|p.into_raw_parts().0),
                "pose":pose(c.position_wrt_parent().unwrap_or(c.position())),
                "friction":c.friction(),"enabled":c.is_enabled(),"sensor":c.is_sensor()});
            let unchanged = self.external_colliders.get(&id).map(|(shape, old)|
                std::ptr::eq(shape.as_ref(), c.shared_shape().as_ref()) && *old == descriptor).unwrap_or(false);
            if !unchanged {
                if let Some(parent) = c.parent() {
                    if !dirty_bodies.contains(&parent) { dirty_bodies.push(parent); }
                }
                let mut update = descriptor.clone();
                update["shape"] = serde_json::to_value(c.shared_shape()).unwrap();
                changed.push(update);
                self.external_colliders.insert(id, (c.shared_shape().clone(), descriptor));
            }
        }
        let removed: Vec<u32> = self.external_colliders.keys().filter(|id| !live.contains(id)).copied().collect();
        for id in &removed { self.external_colliders.remove(id); }
        if initial || !removed.is_empty() {
            for (_, b) in self.bodies.iter_mut() { b.recompute_mass_properties_from_colliders(&self.colliders); }
        } else {
            for handle in dirty_bodies { self.bodies[handle].recompute_mass_properties_from_colliders(&self.colliders); }
        }
        let bodies: Vec<Value> = self.bodies.iter().map(|(h,b)| {
            let mp = &b.mass_properties().local_mprops;
            json!({"id":h.into_raw_parts().0,"dynamic":b.is_dynamic(),"pose":pose(b.position()),
                "next":pose(b.next_position()),"velocity":b.linvel(),"angular":b.angvel(),
                "force":b.user_force(),"torque":b.user_torque(),"mass":b.mass(),
                "mass_properties":mp,"linear_damping":b.linear_damping(),"angular_damping":b.angular_damping()})
        }).collect();
        let request=json!({"op":"external_physics","dt":PHYSICS_DT,"bodies":bodies,
            "colliders":changed,"colliders_delta":!initial,"removed_colliders":removed});
        {
            let stdout=std::io::stdout();
            let mut out=stdout.lock();
            serde_json::to_writer(&mut out,&request).expect("external physics write");
            out.write_all(b"\n").unwrap();out.flush().unwrap();
        }
        let mut line=String::new();
        std::io::stdin().lock().read_line(&mut line).expect("external physics read");
        let response:Value=serde_json::from_str(&line).expect("external physics response JSON");
        assert_eq!(response["op"],"external_physics_result","external physics response required");
        for state in response["bodies"].as_array().expect("external body array") {
            let id=state["id"].as_u64().unwrap() as u32;
            let h=self.bodies.iter().find(|(h,_)| h.into_raw_parts().0==id).map(|(h,_)|h).expect("external body id");
            let b=&mut self.bodies[h];
            let read=|key:&str,n:usize| -> Vec<f64> {
                let a=state[key].as_array().expect("external state array");assert_eq!(a.len(),n);
                a.iter().map(|v| {let f=v.as_f64().unwrap();assert!(f.is_finite());f}).collect()
            };
            let p=read("pos",3);let q=read("quat",4);let v=read("velocity",3);let w=read("angular",3);
            b.set_position(Isometry::from_parts(Translation::new(p[0],p[1],p[2]),UnitQuaternion::new_normalize(Quaternion::new(q[0],q[1],q[2],q[3]))),true);
            b.set_linvel(vector![v[0],v[1],v[2]],true);b.set_angvel(vector![w[0],w[1],w[2]],true);
        }
        if let Some(impacts)=response["impacts"].as_array() {
            for impact in impacts {
                let id=impact["body"].as_u64().unwrap() as u32;
                if let Some(i)=self.handles.iter().position(|h|h.into_raw_parts().0==id) {
                    let p=&impact["pos"];
                    self.impacts.insert(i,json!({"robot":robots[i].id,"x":p[0],"y":p[1],"z":p[2],"impulse_n_s":impact["impulse"]}));
                }
            }
        }
        self.bodies.propagate_modified_body_positions_to_colliders(&mut self.colliders);
        self.queries.update(&self.colliders);
    }
}

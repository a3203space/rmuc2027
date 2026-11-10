//! Referee-side 2027 0x0120 ledger. Only successful operations advance counters.
use super::*;

#[derive(Clone, Default, Serialize)]
pub(crate) struct Ledger {
    pub free_confirm: bool,
    pub free_pending: bool,
    pub local_ammo: u32,
    pub remote_ammo: u32,
    pub remote_heal: u32,
    pub command: u32,
    pub rejection: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ready() -> (Engine,usize) {
        let mut e=Engine::new(42,420.,"rmuc2027").unwrap();e.strict_sentry=true;
        let i=e.robots.iter().position(|r|r.id=="red-7").unwrap();
        e.tick=200;e.teams.get_mut("red").unwrap().resources=1000.;
        e.robots[i].last_shot_time=-100.;e.robots[i].last_damage_time=-100.;
        (e,i)
    }
    #[test]
    fn duplicate_remote_purchase_and_heal_are_idempotent() {
        let (mut e,i)=ready();let word=(3<<21)|(1<<13)|(1<<17);
        e.sentry_command(i,&json!({"sentry_cmd":word}));
        let coins=e.teams["red"].resources;
        assert!(coins<1000.);assert_eq!(e.sentry_ledger["red-7"].remote_ammo,1);
        assert_eq!(e.sentry_ledger["red-7"].remote_heal,1);
        for _ in 0..20 {e.sentry_command(i,&json!({"sentry_cmd":word}));}
        assert_eq!(e.teams["red"].resources,coins);
        assert_eq!(e.robots[i].pending_ammo.len(),1);
    }
    #[test]
    fn counter_gaps_and_failed_lower_fields_block_later_fields() {
        let (mut e,i)=ready();
        e.sentry_command(i,&json!({"sentry_cmd":(1<<21)|(2<<13)}));
        assert_eq!(e.sentry_ledger["red-7"].remote_ammo,0);
        assert_eq!(e.robots[i].posture,"move");
        e.teams.get_mut("red").unwrap().resources=0.;
        e.sentry_command(i,&json!({"sentry_cmd":(1<<21)|(1<<13)}));
        assert_eq!(e.robots[i].posture,"move");
    }
    #[test]
    fn free_revive_requires_confirmation_and_counters_survive_death() {
        let (mut e,i)=ready();e.robots[i].alive=false;e.robots[i].hp=0.;
        e.robots[i].respawn_progress=20.;e.robots[i].respawn_required=10.;
        e.advance(&json!({}));assert!(!e.robots[i].alive);
        e.sentry_command(i,&json!({"sentry_cmd":(3<<21)|1}));assert!(e.robots[i].alive);
    }
    #[test]
    fn early_free_confirmation_is_pending_and_wait_cancels_before_readiness() {
        let (mut e,i)=ready();
        e.robots[i].alive=false;e.robots[i].hp=0.;
        e.robots[i].respawn_progress=9.99;e.robots[i].respawn_required=10.;
        e.sentry_command(i,&json!({"sentry_cmd":(3<<21)|1}));
        assert!(e.sentry_ledger["red-7"].free_pending);
        assert!(e.sentry_ledger["red-7"].rejection.is_empty());
        e.advance(&json!({"red":{"7":{"sentry_cmd":3<<21}}}));
        assert!(!e.robots[i].alive);
        assert!(!e.sentry_ledger["red-7"].free_confirm);
        assert!(!e.sentry_ledger["red-7"].free_pending);
        assert!(e.robots[i].respawn_progress>=10.);
    }
    #[test]
    fn paid_choice_cancels_old_free_intent_at_readiness_boundary() {
        let (mut e,i)=ready();
        e.robots[i].alive=false;e.robots[i].hp=0.;
        e.robots[i].respawn_progress=9.99;e.robots[i].respawn_required=10.;
        e.sentry_command(i,&json!({"sentry_cmd":(3<<21)|1}));
        let before=e.teams["red"].resources;
        e.advance(&json!({"red":{"7":{"sentry_cmd":(3<<21)|2}}}));
        assert!(e.robots[i].alive);
        assert_eq!(e.robots[i].instant_revives,1);
        assert_eq!(e.robots[i].hp,e.robots[i].max_hp);
        assert!(e.teams["red"].resources<before);
    }
    #[test]
    fn early_free_intent_survives_no_new_packet_and_finishes_normally() {
        let (mut e,i)=ready();
        e.robots[i].alive=false;e.robots[i].hp=0.;
        e.robots[i].respawn_progress=9.99;e.robots[i].respawn_required=10.;
        e.sentry_command(i,&json!({"sentry_cmd":(3<<21)|1}));
        let before=e.teams["red"].resources;
        e.advance(&json!({}));
        assert!(e.robots[i].alive);
        assert_eq!(e.robots[i].instant_revives,0);
        assert_eq!(e.teams["red"].resources,before);
        assert!(!e.sentry_ledger["red-7"].free_pending);
    }
    #[test]
    fn electrical_shooter_supply_is_independent_of_motion_and_trigger_timing() {
        let (mut e,i)=ready();e.robots[i].ammo=100;
        assert!(e.referee_shooter_power_on(i));
        e.robots[i].cooldown=10.;e.robots[i].transform_progress=0.5;
        assert!(e.referee_shooter_power_on(i));
        e.robots[i].weak=true;assert!(!e.referee_shooter_power_on(i));
        e.robots[i].weak=false;e.robots[i].ammo=0;
        assert!(!e.referee_shooter_power_on(i));
    }
    #[test]
    fn invalid_reserved_bits_cannot_change_ledger() {
        let (mut e,i)=ready();e.sentry_command(i,&json!({"sentry_cmd":1u32<<31}));
        assert!(e.sentry_ledger.is_empty());
    }
    #[test]
    fn adaptive_armor_uses_module_coordinates_and_original_damage_multiplier() {
        let (mut e,_)=ready();
        e.radar_effects=json!({"armor":{"red":{"center":[67,62],"fraction":0.2}}});
        assert_eq!(e.adaptive_armor_multiplier("red",[67,62],0.135*0.125),0.5);
        assert_eq!(e.adaptive_armor_multiplier("red",[0,0],0.135*0.125),1.5);
        assert_eq!(e.adaptive_armor_multiplier("blue",[0,0],0.135*0.125),1.);
    }
}

impl Engine {
    /// Synchronize the latest received confirmation before progress can cross
    /// its threshold. A wait/paid word must cancel an older free intention.
    pub(super) fn sync_sentry_free_intent(&mut self, i: usize, rpc: &Value) {
        let Some(word) = rpc.get("sentry_cmd").and_then(Value::as_u64)
            .filter(|v| *v < (1 << 25)) else { return; };
        if self.robots[i].red_card { return; }
        let r = &self.robots[i];
        let ledger = self.sentry_ledger.entry(r.id.clone()).or_default();
        ledger.free_confirm = word & 1 != 0;
        ledger.free_pending = ledger.free_confirm && !r.alive
            && r.respawn_progress + 1e-8 < r.respawn_required;
    }

    /// 0x0201 describes electrical supply, not an oracle for whether the next
    /// requested shot will execute. Official rules: weakness locks the shooter
    /// (p75); locked means unpowered and ammunition allowance gates supply (p17).
    fn referee_shooter_power_on(&self, i: usize) -> bool {
        let r = &self.robots[i];
        let now = self.time();
        if !r.alive || r.red_card || r.battery_remaining_wh <= 0. || r.weak || r.role == "aerial" {
            return false;
        }
        if r.overspeed_permanent_lock || r.overspeed_lock_until > now + 1e-8
            || r.overheated || r.heat_permanent_lock { return false; }
        if r.role == "heavy" && (!matches!(r.launcher.as_str(), "fitted" | "installed")
            || self.assembly_active_for(&r.team) && self.raw_zone(i, "assembly", Some(&r.team))) {
            return false;
        }
        r.ammo > 0 || self.fire_reserve_2027(i).0
    }
    pub(super) fn adaptive_armor_multiplier(&self,team:&str,point:[i16;2],area:f64) -> f64 {
        if !self.strict_sentry {return 1.;}
        let config=&self.radar_effects["armor"][team];
        let Some(center)=config["center"].as_array().filter(|a|a.len()==2) else {return 1.};
        let Some(fraction)=config["fraction"].as_f64().filter(|f|f.is_finite() && *f>=0. && *f<=1.) else {return 1.};
        let Some(x)=center[0].as_f64() else {return 1.};
        let Some(y)=center[1].as_f64() else {return 1.};
        let distance=((point[0] as f64-x).powi(2)+(point[1] as f64-y).powi(2))/1e6;
        if distance<=area*fraction/std::f64::consts::PI {0.5} else {1.5}
    }
    pub(super) fn sentry_command(&mut self, i: usize, rpc: &Value) {
        let id = self.robots[i].id.clone();
        let Some(word) = rpc.get("sentry_cmd").and_then(Value::as_u64)
            .filter(|v| *v < (1 << 25)) else {
                if ["revive", "buy_ammo", "remote_heal", "posture", "posture_enhanced", "rune_start"]
                    .iter().any(|k| rpc.get(k).is_some()) {
                    self.event("sentry_command_rejected", json!({"robot":id,"reason":"0x0120 required"}));
                }
                return;
            };
        let word = word as u32;
        let mut ledger = self.sentry_ledger.remove(&id).unwrap_or_default();
        ledger.command = word;
        ledger.free_confirm = word & 1 != 0;
        ledger.free_pending = false;
        ledger.rejection.clear();
        // Ordered fields: a failed operation stops processing the later fields.
        let result = (|| -> Result<(), &'static str> {
            if self.robots[i].red_card { return Err("red_card"); }
            if ledger.free_confirm && !self.robots[i].alive {
                if self.robots[i].respawn_progress + 1e-8 < self.robots[i].respawn_required {
                    // Confirmation before readiness is legal. Keep the intent;
                    // higher fields wait until this operation can complete.
                    ledger.free_pending = true;
                    return Ok(());
                }
                self.respawn_2027(i, false);
            }
            if word & 2 != 0 && !self.robots[i].alive {
                self.legacy_rulebook_rpc_2027(i, &json!({"revive":true}));
                if !self.robots[i].alive { return Err("paid_revive_unavailable"); }
            }
            let local = (word >> 2) & 2047;
            if local > ledger.local_ammo {
                let before = self.teams[&self.robots[i].team].purchased_17;
                self.legacy_rulebook_rpc_2027(i, &json!({"buy_ammo":{"amount":local-ledger.local_ammo,"remote":false}}));
                if self.teams[&self.robots[i].team].purchased_17 == before { return Err("local_ammo_unavailable"); }
                ledger.local_ammo = local;
            } else if local < ledger.local_ammo { return Err("local_counter_regression"); }
            let remote = (word >> 13) & 15;
            if remote > ledger.remote_ammo {
                if remote != ledger.remote_ammo + 1 { return Err("remote_ammo_counter_gap"); }
                let before = self.teams[&self.robots[i].team].purchased_17;
                self.legacy_rulebook_rpc_2027(i, &json!({"buy_ammo":{"remote":true}}));
                if self.teams[&self.robots[i].team].purchased_17 == before { return Err("remote_ammo_unavailable"); }
                ledger.remote_ammo = remote;
            } else if remote < ledger.remote_ammo { return Err("remote_ammo_counter_regression"); }
            let heal = (word >> 17) & 15;
            if heal > ledger.remote_heal {
                if heal != ledger.remote_heal + 1 { return Err("remote_heal_counter_gap"); }
                let before = self.robots[i].pending_heal;
                self.legacy_rulebook_rpc_2027(i, &json!({"remote_heal":true}));
                if before.is_some() || self.robots[i].pending_heal.is_none() { return Err("remote_heal_unavailable"); }
                ledger.remote_heal = heal;
            } else if heal < ledger.remote_heal { return Err("remote_heal_counter_regression"); }
            let posture = (word >> 21) & 7;
            if !(1..=6).contains(&posture) { return Err("invalid_posture"); }
            let name = ["attack", "defense", "move"][((posture-1)%3) as usize];
            let enhanced = posture > 3;
            self.legacy_rulebook_rpc_2027(i, &json!({"posture":name,"posture_enhanced":enhanced}));
            if self.robots[i].posture != name || self.robots[i].posture_enhanced != enhanced {
                return Err("posture_unavailable");
            }
            if word & (1<<24) != 0 { self.legacy_rulebook_rpc_2027(i, &json!({"rune_start":true})); }
            Ok(())
        })();
        if let Err(reason) = result { ledger.rejection = reason.into(); }
        self.sentry_ledger.insert(id, ledger);
    }

    pub(super) fn sentry_telemetry(&self) -> Value {
        let mut result = serde_json::Map::new();
        for (i,r) in self.robots.iter().enumerate().filter(|(_,r)| r.role != "aerial") {
            let cost = (self.time()/60.).ceil()*80. + self.xp_level(i) as f64*20.;
            let quota = (self.rules["economy"]["ammo"]["17mm"]["team_purchase_cap"].as_f64().unwrap_or(0.) as u32)
                .saturating_sub(self.teams[&r.team].purchased_17);
            result.insert(r.id.clone(), json!({"ledger":self.sentry_ledger.get(&r.id).cloned().unwrap_or_default(),
                "level":self.xp_level(i), "disengaged":self.disengaged_2027(i),
                "free_revive":!r.alive && !r.red_card && r.respawn_progress+1e-8>=r.respawn_required,
                "paid_revive":!r.alive && !r.red_card && self.teams[&r.team].resources>=cost,
                "paid_cost":cost,"ammo_quota":quota,
                "power_outputs":if r.alive && !r.red_card && r.battery_remaining_wh>0. {
                    1 | (if r.powered {2} else {0}) |
                    (if self.referee_shooter_power_on(i) {4} else {0})
                } else {0},
                "heal_percent":if !r.alive || r.weak {0.} else if self.contacted_zone_2027(i,"supply",Some(&r.team)) {
                    self.supply_heal_rate(i)*100.
                } else if self.on_controlled_outpost(i) {self.supply_heal_rate(i)*50.} else {0.},
                "rune_available":!["active","activating"].contains(&self.runes[&r.team].status.as_str()) &&
                    (if self.time()<180. {self.runes[&r.team].small_chances} else {self.runes[&r.team].large_chances})>0}));
        }
        Value::Object(result)
    }

    /// Independent ideal autoaim sensor. FPV interruption is not a local sensor outage.
    /// Only target geometry is returned; no health, buffs, ammo or building selection.
    pub fn sentry_sensors(&self, team: &str) -> Value {
        let Some(a) = self.robots.iter().find(|r|r.team==team && r.slot==7) else {return json!([])};
        if !a.alive || a.red_card || a.battery_remaining_wh<=0. { return json!([]); }
        let mut observer = a.clone();
        observer.overheated = false;
        let targets:Vec<Value> = self.robots.iter().filter(|b| b.team!=team && b.alive && b.role!="aerial" && self.visible(&observer,b))
            .map(|b|json!({"id":b.id,"slot":b.slot,"x":b.x,"y":b.y,"z":b.z,
                "vx":b.vx,"vy":b.vy,"vz":b.vz,"armor_center_z_m":b.armor_center_z_m,"radius":b.radius})).collect();
        json!(targets)
    }
}

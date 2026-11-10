import {describe,expect,test} from 'bun:test';
import {combatButton,createCombatInput,releaseCombat} from '../../apps/client/web/combatInput';
describe('mouse combat input',()=>{
 test('short left click persists an edge even after release',()=>{const s=createCombatInput();combatButton(s,0,true);combatButton(s,0,false);expect(s.fire).toBe(false);expect(s.fire_press_seq).toBe(1);});
 test('hold does not duplicate edges and another click increments',()=>{const s=createCombatInput();combatButton(s,0,true);combatButton(s,0,true);expect(s.fire).toBe(true);expect(s.fire_press_seq).toBe(1);combatButton(s,0,false);combatButton(s,0,true);expect(s.fire_press_seq).toBe(2);});
 test('right hold and left fire are independent',()=>{const s=createCombatInput();combatButton(s,2,true);expect(s.aim_assist).toBe(true);expect(s.fire).toBe(false);combatButton(s,0,true);combatButton(s,2,false);expect(s.aim_assist).toBe(false);expect(s.fire).toBe(true);});
 test('neutral clears holds without replaying old presses',()=>{const s=createCombatInput();combatButton(s,0,true);combatButton(s,2,true);releaseCombat(s);expect(s.fire).toBe(false);expect(s.aim_assist).toBe(false);expect(s.fire_press_seq).toBe(1);combatButton(s,0,true);expect(s.fire_press_seq).toBe(2);});
});

/** Mouse edges survive the 20 ms input sampler and the host's physics step. */
export function createCombatInput(){return {fire:false,fire_press_seq:0,aim_assist:false};}
export function combatButton(state:ReturnType<typeof createCombatInput>,button:number,down:boolean){
 if(button===0){if(down&&!state.fire)state.fire_press_seq++;state.fire=down;}
 if(button===2)state.aim_assist=down;
}
export function releaseCombat(state:ReturnType<typeof createCombatInput>){state.fire=false;state.aim_assist=false;}

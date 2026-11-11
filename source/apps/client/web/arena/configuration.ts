import type {RoomState,StrategyEntry} from '../../../../packages/renderer/src/lanTypes';
export function compatible(strategy:StrategyEntry|undefined,slot:number,model:number){return !!strategy&&(strategy.flexible||strategy.mode==='adaptive'&&strategy.robot===model||strategy.mode==='robot'&&strategy.robot===slot);}
export function teamCompatible(strategy:StrategyEntry,room:RoomState,team:'red'|'blue'){
 const expected=strategy.lineup||{'1':2711,'3':2731,'4':2732,'6':2761,'7':2771};
 return strategy.flexible||['team','distributed','squad'].includes(strategy.mode)&&[1,3,4,6,7].every(slot=>expected[String(slot)]===room.seat_configs?.[team]?.[String(slot)].model);
}

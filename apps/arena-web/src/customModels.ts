import type {Frame,Metadata,RobotSkin,TeamId} from './types';

export {fitCustomModel,isCustomAsset,useRobotScene} from '../../../packages/renderer/src/customModels';
export const skinFor=(metadata:Metadata|null|undefined,team:TeamId,model:number|undefined):RobotSkin|undefined=>model===undefined?undefined:metadata?.skins?.[team]?.[String(model)];
// Shows uploaded appearances in a frame: each robot keeps its simulated hardware and takes the chosen mesh.
export function applySkins<T extends Frame|null|undefined>(frame:T,metadata:Metadata|null|undefined):T{
  if(!frame||!metadata?.skins)return frame;
  for(const r of frame.robots){const skin=skinFor(metadata,r.team,r.model_slot??r.slot);if(skin)r.model_asset=skin.asset;}
  return frame;
}

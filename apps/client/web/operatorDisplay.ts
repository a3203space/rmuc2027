import type {LanPlayer} from '../../../packages/renderer/src/lanTypes';

export function cameraLayerClass(display:LanPlayer['display']|undefined,presentation:boolean){
 return `camera-layer ${!presentation&&display?.grayscale?'camera-grayscale':''}`;
}
export function operatorControlAllowed(phase:string,player:LanPlayer){
 return phase==='running'&&player.scene?.status==='running'&&!!player.self.alive&&!player.display.ui_blocked&&!player.display.camera_off&&!player.control?.needs_resume;
}

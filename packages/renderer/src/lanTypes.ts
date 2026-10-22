import type {Frame, Robot} from './types';

export interface PlayerCamera {origin:number[];direction:number[];up?:number[];horizontal_fov_deg:number;vertical_fov_deg:number}
export interface LanPlayer {
  protocol:number;match_id:string;tick:number;team:'red'|'blue';slot:number;self:Robot;
  display:{ui_blocked:boolean;camera_off:boolean;reason:string;logistics:number;resolution_scale:number;camera:PlayerCamera|null};
  scene?:Partial<Frame>;control?:{needs_resume:boolean;status:string;ack_seq:number;manual:boolean};
}
export interface ArenaBridge {
  snapshot:{connect:(fn:(payload:string)=>void)=>void};
  input:(payload:string)=>void;
  viewportReady:()=>void;
  pause:()=>void;
  renderStats:(payload:string)=>void;
}
declare global {
  interface Window {qt?:{webChannelTransport:unknown};QWebChannel?:new(transport:unknown,callback:(channel:{objects:{arena:ArenaBridge}})=>void)=>unknown}
}

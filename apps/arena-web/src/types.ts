import type {AgentInfo} from './agents';
import type {Season,TeamId,TeamLineup,Frame,RobotHardware,Role,MatchEvent} from '../../../packages/renderer/src/types';
export * from '../../../packages/renderer/src/types';
export type ControlMode = 'team' | 'robot' | 'squad' | 'distributed' | 'adaptive';
export const MODE_LABELS: Record<ControlMode,string> = {team:'全队统筹',robot:'单机专精',squad:'单机专精组队',distributed:'多机协同',adaptive:'自由适配'};
export interface Metadata {battle_settings?:Record<string,number>;save_mp4?:boolean;match_mode?:'standard'|'deck';season?:Season;roster?:number[];lineups?:Record<TeamId,{slot:number;model:number}[]>;seed:number;duration:number;map:string;dt?:number;match_id?:string;engine?:string;rules?:string;teams:Record<TeamId,{name:string;version:string;mode:ControlMode;agent?:AgentInfo|null;display_name?:string;cards?:{name:string;version:string;model:number;agent?:AgentInfo|null}[]}>;
  // Per deployed robot: the agent that generated its strategy, keyed by deployment slot.
  agents?:Record<TeamId,Record<string,AgentInfo&{strategy?:string}>>;
  // Uploaded robot appearances, keyed by hardware model number.
  skins?:Record<TeamId,Record<string,RobotSkin>>}
export interface Prematch {stage:'self_check'|'countdown';remaining:number;duration:number}
export interface ServerState {frame:Frame;revision:number;paused:boolean;speed:number;error:string|null;metadata:Metadata|null;sandbox:string;match_id:string|null;events:MatchEvent[];busy:boolean;prematch?:Prematch|null;strategy_version?:string|null}
export interface Replay {format:'rmuc-replay-v1'|'rmuc-replay-v2'|'rmuc-replay-v3'|'rmuc-replay-v4';static?:Partial<Frame>;metadata:Metadata;frames:Frame[];actions:unknown[];diagnostics:unknown[];result:{winner:TeamId|null;reason:string;time:number;digest:string;policy_errors:number}}
export interface CatalogEntry{name:string;version:string;mode:ControlMode;season?:Season;robot?:number;model_name?:string;role?:Role;preset?:{id:'balanced'|'assault'|'guard';name:string;description:string};team_lineup?:TeamLineup;agent?:AgentInfo;display_name?:string}
// Display sidecars of a submission (rmuc_arena/showcase.py): the upload questionnaire and the agent-written showcase.json.
export type Tactic='balanced'|'assault'|'siege'|'guard'|'economy';
export type StrategyTag='rune'|'assembly'|'rescue'|'aerial'|'sniping'|'mobility'|'comms'|'focus_fire'|'outpost_rush'|'base_rush';
export interface StrategyProfile{format:'rmuc-strategy-profile-v1';agent:AgentInfo;tactic:Tactic;tags:StrategyTag[];display_name?:string;author?:string;description?:string;created_at?:number}
export type StyleKey='aggression'|'defense'|'economy'|'objective'|'teamwork'|'adaptability';
export interface Showcase{format:'rmuc-showcase-v1';title:string;tagline:string;summary:string;style:Record<StyleKey,number>;phases:{from:number;to:number;title:string;goal:string;focus:'attack'|'defense'|'economy'|'objective'|'support'}[];units:{name:string;role:string;task:string;route:number[][];anchor?:number[]}[];rules:{when:string;then:string}[];priorities:string[];strengths:string[];weaknesses:string[];generated_by?:AgentInfo}
export interface StrategySummary extends CatalogEntry{key:string;bundled:boolean;profile:StrategyProfile;showcase:Pick<Showcase,'title'|'tagline'|'style'>|null;standing?:{rating:number;games:number;wins:number;losses:number;draws:number};created_at?:number}
export interface StrategyDetail extends Omit<StrategySummary,'showcase'>{showcase:Showcase|null;files:{name:string;bytes:number;lines:number;sha256:string}[]}
// An uploaded robot appearance: a GLB shown in place of its hardware model, which keeps the physics.
export interface RobotSkin{id:string;name:string;base_model:number;forward:'+x'|'-x'|'+z'|'-z';asset:string;author?:string;bytes?:number;created_at?:number}
export interface MatchRecord{has_video?:boolean;video_bytes?:number;video_status?:string|null;video_error?:string|null;id:string;created_at:string;status:string;metadata:Metadata;result:Replay['result']|null;has_replay:boolean;replay_bytes?:number;error:string|null}
export interface Standing{identity:string;name:string;version:string;mode:ControlMode;rating:number;games:number;wins:number;losses:number;draws:number}

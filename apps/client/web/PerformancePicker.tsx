import type {PerformanceProfile,ProfileSaveState} from '../../../packages/renderer/src/lanTypes';
import {EMPTY_PROFILE_SAVE} from './performance';
export function PerformancePicker({profile,reason,save=EMPTY_PROFILE_SAVE,onChange,onReturnToOwn}:{profile:PerformanceProfile;reason:string;save?:ProfileSaveState;onChange:(p:PerformanceProfile)=>void;onReturnToOwn?:()=>void}){
 const value=save.pending?.profile??profile,disabled=!!reason||!!save.pending;
 return <section className="performance-picker" aria-label="裁判性能预选" aria-busy={!!save.pending}>
 <h3>裁判性能预选</h3>
 <fieldset disabled={disabled}><legend>底盘性能</legend><div className="performance-options" role="radiogroup" aria-label="底盘性能">
 {(['hp','power'] as const).map(chassis=><button type="button" role="radio" aria-checked={value.chassis===chassis} key={chassis} onClick={()=>onChange({...value,chassis})}><b>{chassis==='hp'?'血量优先':'功率优先'}</b><small>{chassis==='hp'?'提高血量上限':'提高底盘功率'}</small></button>)}
 </div></fieldset>
 <fieldset disabled={disabled}><legend>发射机构性能</legend><div className="performance-options" role="radiogroup" aria-label="发射机构性能">
 {(['cooling','burst'] as const).map(weapon=><button type="button" role="radio" aria-checked={value.weapon===weapon} key={weapon} onClick={()=>onChange({...value,weapon})}><b>{weapon==='cooling'?'冷却优先':'爆发优先'}</b><small>{weapon==='cooling'?'提高热量冷却速率':'提高热量上限'}</small></button>)}
 </div></fieldset>
 <p className={`performance-feedback ${save.status}`} role="status">{reason||save.message||'当前配置：'+(profile.chassis==='hp'?'血量':'功率')+' / '+(profile.weapon==='cooling'?'冷却':'爆发')+' · 点击自动保存'}</p>
 {onReturnToOwn&&<button type="button" className="performance-return" disabled={!!save.pending} onClick={onReturnToOwn}>返回我的席位</button>}
 </section>;
}

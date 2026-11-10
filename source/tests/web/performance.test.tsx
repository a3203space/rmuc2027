import {describe,expect,test} from 'bun:test';
import {Children,isValidElement,type ReactElement,type ReactNode} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {PerformancePicker} from '../../apps/client/web/PerformancePicker';
import {beginProfileSave,EMPTY_PROFILE_SAVE,finishProfileSave,newerRoom,profileDisabledReason,seatProfile} from '../../apps/client/web/performance';
import type {PerformanceProfile,RoomState} from '../../packages/renderer/src/lanTypes';
const profile:PerformanceProfile={chassis:'hp',weapon:'cooling'};
const room:RoomState={status:'lobby',revision:5,players:[{id:'me',nickname:'自己',team:'red',slot:3,ready:false,host:true,connected:true,profile},{id:'other',nickname:'队友',team:'red',slot:4,ready:false,host:false,connected:true,profile:{chassis:'power',weapon:'burst'}}]};
function buttons(node:ReactNode):ReactElement<any>[]{
 return Children.toArray(node).flatMap(child=>{
  if(!isValidElement(child))return [];
  const props=child.props as any;
  return [...(props.role==='radio'?[child]:[]),...buttons(props.children)];
 });
}
describe('infantry performance selection',()=>{
 test('two independent button groups replace the native select',()=>{const html=renderToStaticMarkup(<PerformancePicker profile={profile} reason="" onChange={()=>{}}/>);expect(html).toContain('血量优先');expect(html).toContain('冷却优先');expect(html.match(/role="radio"/g)).toHaveLength(4);expect(html.match(/aria-checked="true"/g)).toHaveLength(2);expect(html).not.toContain('<select');expect(html).not.toContain('disabled');});
 test('each button sends a complete profile while retaining the other group',()=>{const submitted:PerformanceProfile[]=[];const tree=PerformancePicker({profile,reason:'',onChange:p=>submitted.push(p)});for(const button of buttons(tree))button.props.onClick();expect(submitted).toEqual([{chassis:'hp',weapon:'cooling'},{chassis:'power',weapon:'cooling'},{chassis:'hp',weapon:'cooling'},{chassis:'hp',weapon:'burst'}]);});
 test('own 3 and 4 seats are editable, other seats explain read-only state',()=>{expect(profileDisabledReason(room,'me',{team:'red',slot:3})).toBe('');expect(profileDisabledReason(room,'other',{team:'red',slot:4})).toBe('');expect(profileDisabledReason(room,'me',{team:'red',slot:4})).toContain('其他席位');expect(profileDisabledReason(room,'missing',{team:'red',slot:3})).toContain('请先');});
 test('running, starting and disconnected states give reasons',()=>{for(const status of ['starting','loading','intro','countdown','running'])expect(profileDisabledReason({...room,status},'me',{team:'red',slot:3})).toContain('锁定');expect(profileDisabledReason({...room,players:[{...room.players![0],connected:false}]},'me',{team:'red',slot:3})).toContain('断开');});
 test('transport loss disables selection before the server snapshot changes',()=>{expect(profileDisabledReason(room,'me',{team:'red',slot:3},false)).toContain('客户端连接已中断');});
 test('read-only view shows the inspected player profile',()=>{expect(seatProfile(room,{team:'red',slot:4})).toEqual({chassis:'power',weapon:'burst'});expect(seatProfile(room,{team:'red',slot:3})).toEqual(profile);});
 test('disabled reason and return-to-own entry are visible',()=>{const html=renderToStaticMarkup(<PerformancePicker profile={profile} reason="只能修改自己的步兵性能" onChange={()=>{}} onReturnToOwn={()=>{}}/>);expect(html).toContain('<fieldset disabled=""');expect(html).toContain('只能修改自己的步兵性能');expect(html).toContain('返回我的席位');});
 test('saving locks controls and duplicate submissions preserve the first request',()=>{const pending=beginProfileSave(EMPTY_PROFILE_SAVE,'a',{chassis:'power',weapon:'cooling'});expect(beginProfileSave(pending,'b',profile)).toBe(pending);const html=renderToStaticMarkup(<PerformancePicker profile={profile} save={pending} reason="" onChange={()=>{}}/>);expect(html).toContain('保存中');expect(html).toContain('aria-busy="true"');expect(html).toContain('<fieldset disabled=""');});
 test('only the matching acknowledgement can finish a save',()=>{const pending=beginProfileSave(EMPTY_PROFILE_SAVE,'a',profile);expect(finishProfileSave(pending,{request_id:'old',ok:true})).toBe(pending);const done=finishProfileSave(pending,{request_id:'a',ok:true});expect(done.pending).toBeNull();expect(done.status).toBe('saved');expect(done.message).toContain('重新准备');});
 test('failure and timeout release pending controls and use confirmed selection',()=>{for(const message of ['保存失败','保存超时']){const pending=beginProfileSave(EMPTY_PROFILE_SAVE,'a',{chassis:'power',weapon:'burst'});const failed=finishProfileSave(pending,{request_id:'a',ok:false,message});const html=renderToStaticMarkup(<PerformancePicker profile={profile} save={failed} reason="" onChange={()=>{}}/>);expect(failed.pending).toBeNull();expect(html).toContain(message);expect(html).not.toContain('disabled');expect(html).toContain('aria-checked="true"><b>血量优先');expect(html).toContain('aria-checked="true"><b>冷却优先');}});
 test('older snapshots cannot revert a newly saved profile',()=>{const newer={...room,revision:6};expect(newerRoom(newer,room)).toBe(newer);expect(newerRoom(room,newer)).toBe(newer);expect(newerRoom(newer,null)).toBeNull();});
});

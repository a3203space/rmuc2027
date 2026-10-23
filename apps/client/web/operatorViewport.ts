import {useLayoutEffect,useState,type RefObject} from 'react';

export const OPERATOR_WIDTH=1920,OPERATOR_HEIGHT=1080;
export function fitOperatorViewport(width:number,height:number){
 const scale=Math.max(0,Math.min(width/OPERATOR_WIDTH,height/OPERATOR_HEIGHT));
 return {width:OPERATOR_WIDTH*scale,height:OPERATOR_HEIGHT*scale,left:(width-OPERATOR_WIDTH*scale)/2,top:(height-OPERATOR_HEIGHT*scale)/2,scale};
}

/** Camera pixels and HUD positions share one fitted 1920 × 1080 surface. */
export function useOperatorViewport(ref:RefObject<HTMLDivElement|null>){
 const [rect,setRect]=useState(()=>fitOperatorViewport(window.innerWidth,window.innerHeight));
 useLayoutEffect(()=>{
  const target=ref.current;if(!target)return;
  const update=()=>setRect(fitOperatorViewport(target.clientWidth,target.clientHeight));
  const observer=new ResizeObserver(update);observer.observe(target);update();
  return()=>observer.disconnect();
 },[ref]);
 return rect;
}

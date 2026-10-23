import {useEffect,useState} from 'react';
import type {Robot} from './types';
import {robotImage} from './thumbnails';

export function useRobotImages(robots:readonly Pick<Robot,'model_asset'|'team'>[],showcase=false){
 const [images,setImages]=useState<Record<string,string>>({});
 const signature=[...new Set(robots.filter(r=>r.model_asset).map(r=>`${r.model_asset}|${r.team}`))].sort().join('\n');
 useEffect(()=>{
  let active=true;
  for(const entry of signature.split('\n').filter(Boolean)){
   const [url,team]=entry.split('|') as [string,'red'|'blue'];
   for(const size of showcase?['card','showcase'] as const:['card'] as const){
    const key=size==='card'?entry:`${entry}|showcase`;
    robotImage(url,team,size).then(image=>{if(active)setImages(previous=>previous[key]===image?previous:{...previous,[key]:image});}).catch(error=>console.warn('ARENA_THUMBNAIL',url,String(error)));
   }
  }
  return()=>{active=false;};
 },[signature,showcase]);
 return images;
}

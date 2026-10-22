import {useGLTF} from '@react-three/drei';
import {useMemo} from 'react';
import {batchRigidMeshes} from './rigidMeshes';
export function useRobotScene(url:string){
 const {scene,animations}=useGLTF(url);
 return useMemo(()=>/^\/assets\/(?:robot|drone)-/.test(url)&&!animations.length?batchRigidMeshes(scene):scene,[scene,animations,url]);
}

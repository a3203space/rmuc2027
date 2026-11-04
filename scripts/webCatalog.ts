/** Offline hardware directory. Keep stat selection aligned with core/models.py. */
export function webCatalog(rules:Record<string,any>){
 const season=rules.seasons['2027'],book=rules[season.rules_key];
 const models=Object.fromEntries(Object.entries(rules.robot_designs).map(([id,value])=>{
  const design=value as Record<string,any>,role=design.role,stats={...season.specs[role]};
  if(role==='heavy')Object.assign(stats,book.heavy.stages[0]);
  else if(role==='infantry'){
   const profiles=book.profiles,defaults=profiles.defaults;
   Object.assign(stats,profiles.infantry_chassis[defaults.infantry_chassis][0],profiles.infantry_weapon[defaults.infantry_weapon][0]);
  }else if(role==='sentry')Object.assign(stats,book.sentry.automatic,{ammo:book.initial_ammo.sentry});
  const initial=Object.fromEntries(['hp','power_w','max_heat','cooling','damage','ammo'].map(key=>[key,stats[key]??0]));
  return [id,{...design,initial_stats:initial,initial_stats_by_season:design.seasons?.['2027']?.legal?{'2027':initial}:{}}];
 }));
 return {models,strategies:[]};
}

import type { LoftModel } from './loft-renderer';
export const descriptors=['curvature','collision','deflation','centrality','rhythm','scale'] as const;
export type Scores=Record<typeof descriptors[number],number>;
const clamp=(v:number)=>Math.max(0,Math.min(1,v));
const mean=(v:number[])=>v.length?v.reduce((a,b)=>a+b,0)/v.length:0;
const score=(v:number)=>Math.round((1+9*clamp(v))*10)/10;
/** Diagrammatic geometric proxies, not occupancy or structural validation. */
export function analyzeLoft(model:LoftModel):Scores|null{
 const rows=model.rows.filter(row=>row.length>=3),points=rows.flat();
 if(rows.length<2||!points.length||points.some(p=>p.some(v=>!Number.isFinite(v))))return null;
 const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];for(const p of points)for(let k=0;k<3;k++){low[k]=Math.min(low[k],p[k]);high[k]=Math.max(high[k],p[k]);}const span=high.map((v,k)=>v-low[k]);
 if(Math.max(...span)<1e-6)return null;
 const angles:number[]=[];
 const turn=(curve:number[][])=>{for(let i=1;i<curve.length-1;i++){const a=curve[i].map((v,k)=>v-curve[i-1][k]),b=curve[i+1].map((v,k)=>v-curve[i][k]),la=Math.hypot(...a),lb=Math.hypot(...b);if(la>1e-6&&lb>1e-6)angles.push(Math.acos(Math.max(-1,Math.min(1,a.reduce((v,x,k)=>v+x*b[k],0)/(la*lb)))))} };
 rows.forEach(turn);model.rails.forEach(turn);
 // Total bending over profile length keeps scores independent of tessellation density.
 const curvature=clamp(angles.reduce((a,b)=>a+b,0)/(Math.PI*Math.max(1,rows.length+model.rails.length)));
 const proximities:number[]=[];
 for(let i=0;i<model.rails.length;i++)for(let j=i+1;j<model.rails.length;j++){const a=model.rails[i],b=model.rails[j],n=Math.min(a.length,b.length);if(n)proximities.push(mean(Array.from({length:n},(_,k)=>Math.hypot(...a[k].map((v,d)=>v-b[k][d])))));}
 const planExtent=Math.max(.01,Math.hypot(span[0],span[2]));
 const collision=mean(proximities.map(distance=>clamp(1-distance/(planExtent*.3))));
 const rails=model.rails.flat(),centerX=(low[0]+high[0])/2,centerZ=(low[2]+high[2])/2;
 const centrality=rails.length?mean(rails.map(p=>clamp(1-Math.hypot((p[0]-centerX)/Math.max(.01,span[0]/2),(p[2]-centerZ)/Math.max(.01,span[2]/2))/Math.SQRT2))):0;
 const widths=rows.map(row=>Math.max(...row.map(p=>p[0]))-Math.min(...row.map(p=>p[0]))),widthMean=mean(widths),variation=widthMean>1e-6?Math.sqrt(mean(widths.map(w=>(w-widthMean)**2)))/widthMean:1;
 const rhythm=clamp((model.rails.length-1)/5)*clamp(1-variation);
 return {curvature:score(curvature),collision:score(collision),deflation:score(1-span[1]/Math.max(.01,Math.max(span[0],span[2]))),centrality:score(centrality),rhythm:score(rhythm),scale:score(mean(span.map(v=>clamp(v/2.1))))};
}
/** Compare both absolute ratings and the relative descriptor pattern. Generic
 * size/compression signals must not overwhelm a typology's distinctive traits. */
export function rankTypologies(scores:Scores,presets:{name:string;ratings:number[]}[]){
 const weights=[1.4,1.2,.8,1.3,1.2,.5];
 const spreads=descriptors.map((_,i)=>Math.max(2,Math.max(...presets.map(p=>p.ratings[i]))-Math.min(...presets.map(p=>p.ratings[i]))));
 const observed=descriptors.map(key=>scores[key]),observedMean=mean(observed);
 const weightTotal=weights.reduce((a,b)=>a+b,0);
 return presets.map((preset,index)=>{
  const referenceMean=mean(preset.ratings);
  const absolute=Math.sqrt(observed.reduce((sum,v,i)=>sum+weights[i]*((v-preset.ratings[i])/spreads[i])**2,0)/weightTotal);
  const pattern=Math.sqrt(observed.reduce((sum,v,i)=>sum+weights[i]*(((v-observedMean)-(preset.ratings[i]-referenceMean))/9)**2,0)/weightTotal);
  const distance=9*(.7*absolute+.3*pattern);
  return {...preset,index,distance,similarity:Math.round(100*clamp(1-distance/9))};
 }).sort((a,b)=>a.distance-b.distance||a.index-b.index);
}

export type SpaceCategory='lobby'|'office'|'gathering';
export const psychologicalIntent:Record<SpaceCategory,{goals:string[];risks:string[];context:string[]}>={
 lobby:{goals:['A compressed threshold followed by expansion supports anticipation and orientation.','Vertical openness and a clear circulation hierarchy help visitors understand where to go.'],risks:['Uniform space can lack directional cues.','Competing routes and leftover pockets can weaken legibility.'],context:['Entrance location and destination visibility','Directional daylight and lighting hierarchy']},
 gathering:{goals:['Defined but permeable edges balance visibility, protection, and the choice to stay.','Places to pause beside everyday routes support informal encounters without blocking movement.'],risks:['A distant destination can reduce spontaneous encounters.','An undifferentiated through-route offers little reason to pause.'],context:['Distance to workspaces and daily routes','Visibility, seating, edge permeability, and acoustic privacy']},
 office:{goals:['Distributed work zones support different work modes and degrees of privacy.','Controlled, diffuse daylight supports the intended focus and comfort.'],risks:['Busy circulation through a work zone can create distraction.','Uniform space and uncontrolled glare can undermine the intended comfort.'],context:['Daylight direction, glare, and illumination','Actual traffic, acoustic conditions, and privacy']}
};
/** Transparent design heuristic drawn from the supplied slides, not a psychological prediction. */
export function recommendSpace(scores:Scores,presets:Record<SpaceCategory,{name:string;ratings:number[]}[]>){
 const n=(key:keyof Scores)=>(scores[key]-1)/9;
 const bounded=(v:number)=>Math.max(0,Math.min(1,v));
 const checks:Record<SpaceCategory,{name:string;value:number;reason:string}[]>={
 lobby:[{name:'Circulation legibility',value:bounded(.65*(1-n('collision'))+.35*n('rhythm')),reason:'Less rail convergence and clearer repetition are proxies for orientation.'},{name:'Compression–expansion potential',value:bounded((n('curvature')+n('deflation'))/2),reason:'Bending and compression suggest transition potential; they do not identify an actual entrance.'},{name:'Vertical openness',value:1-n('deflation'),reason:'Lower compression suggests greater vertical openness.'}],
 gathering:[{name:'Encounter potential',value:bounded(.4*n('collision')+.6*n('centrality')),reason:'Converging, centrally concentrated rails suggest opportunities for encounter.'},{name:'Defined edge potential',value:bounded((n('curvature')+n('deflation'))/2),reason:'Curvature and compression suggest spatial definition; visibility and enclosure need review.'},{name:'Spatial hierarchy potential',value:bounded(.4*n('rhythm')+.6*n('curvature')),reason:'Repetition and changes in form suggest differentiation between movement and staying.'}],
 office:[{name:'Distributed zone potential',value:bounded(.6*n('rhythm')+.4*(1-n('centrality'))),reason:'Repeated rails with less central concentration suggest distributed work zones.'},{name:'Separation from through-traffic',value:1-n('collision'),reason:'Lower rail convergence is a proxy for reduced circulation conflict, not measured traffic.'},{name:'Moderate spatial definition',value:bounded(1-Math.abs(n('deflation')-.5)*2),reason:'Moderate compression suggests a balance of openness and definition.'}]
 };
 return (['lobby','office','gathering'] as const).map(category=>{
  const ranked=rankTypologies(scores,presets[category]),referenceFit=bounded(1-ranked[0].distance/9),intentFit=mean(checks[category].map(check=>check.value));
  return {category,typology:ranked[0].name,referenceFit:Math.round(referenceFit*100),intentFit:Math.round(intentFit*100),fit:Math.round((.6*referenceFit+.4*intentFit)*100),checks:checks[category]};
 }).sort((a,b)=>b.fit-a.fit);
}

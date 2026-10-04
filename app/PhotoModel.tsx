"use client";
import {ComponentScreenshotExport} from "./component-screenshots";
import {saveExport} from "./export-destination";
import { analyzeLoft, rankTypologies, recommendSpace, psychologicalIntent } from "./space-analysis";
import { ModelGif } from "./gif-export";
import { illustratorSheet } from "./illustrator-sheet";
import { waterfall } from "./waterfall";
import { createLoftRenderer, loftDisplayMesh, type LoftModel, type LoftCamera } from "./loft-renderer";

import { useEffect, useMemo, useRef, useState } from "react";
import {useWorkspaceState} from "./workspace-state";

/** Uniform vertical rescale only: the floor datum stays fixed and every rise/run
 * relationship in the loft retains the same proportions. The 20 ft footprint
 * is unaffected. One loft unit represents 20 / 2.1 feet in the circulation model. */
export function enforceMinimumArchitecturalHeight(model:LoftModel,minimumFeet=6){
  const ys=[...model.rows.flatMap(row=>row.map(point=>point[1])),...model.rails.flatMap(rail=>rail.map(point=>point[1]))];
  if(ys.length<2)return{model,factor:1,heightFeet:0};
  let low=Infinity,high=-Infinity;for(const y of ys){low=Math.min(low,y);high=Math.max(high,y)}const heightFeet=(high-low)*20/2.1;
  if(heightFeet<=1e-8)return{model,factor:1,heightFeet};
  const factor=Math.max(1,minimumFeet/heightFeet);
  if(factor<=1)return{model,factor:1,heightFeet};
  const scaleLine=(line:Vec[])=>line.map(([x,y,z])=>[x,low+(y-low)*factor,z] as Vec);
  return{model:{rows:model.rows.map(scaleLine),rails:model.rails.map(scaleLine)},factor,heightFeet:minimumFeet};
}
import { Box, Download, Grid3X3, Hand, Layers3, MousePointer2, Redo2, RotateCcw, Route, Ruler, ScanLine, Sparkles, Undo2, XCircle } from "lucide-react";

type StoredImage={key:string;name:string;size:number;uploadedAt:string;folder?:"existing"|"new"};
const source=(key:string)=>`/api/images/content?key=${encodeURIComponent(key)}`;
const W=64,H=48,steps=8;
type Slice={field:Float32Array;z:number;image:HTMLCanvasElement};
type Vec=[number,number,number];
type RealitySettings={supportSpacing:number;clearance:number;maxSlope:number};
const VIEWPORT_PRIMARY="#1ec1f2",VIEWPORT_SECONDARY="#dc24f4",VIEWPORT_TERTIARY="#ff5284";
const viewportColor=(level:"primary"|"secondary"|"tertiary",alpha=1)=>{const rgb=level==="primary"?"30,193,242":level==="secondary"?"220,36,244":"255,82,132";return `rgba(${rgb},${alpha})`};

function fieldFromImage(data:Uint8ClampedArray,depthGain:number){
  const raw=new Float32Array(W*H),result=new Float32Array(W*H),values:number[]=[];
  for(let i=0;i<raw.length;i++){const k=i*4,gray=data[k]*.299+data[k+1]*.587+data[k+2]*.114;values.push(gray)}
  values.sort((a,b)=>a-b);const low=values[Math.floor(values.length*.06)],high=values[Math.floor(values.length*.96)],range=Math.max(18,high-low),gamma=1.9-depthGain*.012;
  for(let i=0;i<raw.length;i++){const k=i*4,gray=data[k]*.299+data[k+1]*.587+data[k+2]*.114,n=Math.max(0,Math.min(1,(gray-low)/range));raw[i]=n<.055?0:Math.pow(n,Math.max(.55,gamma))}
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    let sum=0,weights=0,center=raw[y*W+x];
    for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const xx=x+dx,yy=y+dy;if(xx>=0&&xx<W&&yy>=0&&yy<H){const sample=raw[yy*W+xx],spatial=dx===0&&dy===0?4:1,edge=Math.exp(-Math.abs(sample-center)*10),weight=spatial*edge;sum+=sample*weight;weights+=weight}}
    result[y*W+x]=Math.max(0,Math.min(1,(sum/weights)*.78+center*.22));
  }
  return carvePerimeterVoids(result);
}

/*
 * Cavities in the reference photographs are defined by a continuous rim. Their
 * interiors are not reliably black: the far wall of an opening can be bright.
 * Detect a strong perimeter first, seed the opening from its darker/deeper
 * pixels, then grow that seed across smooth interior tones until it reaches the
 * rim. This keeps lighting inside a void from being mistaken for solid mass.
 */
function carvePerimeterVoids(field:Float32Array){
  const gradient=new Float32Array(field.length),texture=new Float32Array(field.length);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const at=(xx:number,yy:number)=>field[Math.max(0,Math.min(H-1,yy))*W+Math.max(0,Math.min(W-1,xx))];
    const gx=(at(x+1,y-1)+2*at(x+1,y)+at(x+1,y+1)-at(x-1,y-1)-2*at(x-1,y)-at(x-1,y+1))/4;
    const gy=(at(x-1,y+1)+2*at(x,y+1)+at(x+1,y+1)-at(x-1,y-1)-2*at(x,y-1)-at(x+1,y-1))/4;
    gradient[y*W+x]=Math.min(1,Math.hypot(gx,gy));
    let sum=0,sum2=0,count=0;
    for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){const v=at(x+dx,y+dy);sum+=v;sum2+=v*v;count++}
    const mean=sum/count;texture[y*W+x]=Math.sqrt(Math.max(0,sum2/count-mean*mean));
  }

  // Remove the true exterior background so only cavities within the object are seeded.
  const exterior=new Uint8Array(field.length),queue:number[]=[];
  const addExterior=(x:number,y:number)=>{const i=y*W+x;if(!exterior[i]&&field[i]<.14){exterior[i]=1;queue.push(i)}};
  for(let x=0;x<W;x++){addExterior(x,0);addExterior(x,H-1)}
  for(let y=0;y<H;y++){addExterior(0,y);addExterior(W-1,y)}
  for(let q=0;q<queue.length;q++){
    const i=queue[q],x=i%W,y=Math.floor(i/W);
    for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){const xx=x+dx,yy=y+dy;if(xx<0||xx>=W||yy<0||yy>=H)continue;const n=yy*W+xx;if(!exterior[n]&&field[n]<.2){exterior[n]=1;queue.push(n)}}
  }

  const seed=new Uint8Array(field.length);
  for(let i=0;i<field.length;i++)if(!exterior[i]&&field[i]<.28)seed[i]=1;
  const grown=new Uint8Array(seed),growQueue:number[]=[];
  for(let i=0;i<seed.length;i++)if(seed[i])growQueue.push(i);
  for(let q=0;q<growQueue.length;q++){
    const i=growQueue[q],x=i%W,y=Math.floor(i/W);
    for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){
      const xx=x+dx,yy=y+dy;if(xx<1||xx>=W-1||yy<1||yy>=H-1)continue;const n=yy*W+xx;
      // Strong gradients form the cavity rim; smooth bright pixels may still be inside it.
      const interiorTone=field[n]<.73||texture[n]<.075;
      if(!grown[n]&&!exterior[n]&&interiorTone&&gradient[n]<.21){grown[n]=1;growQueue.push(n)}
    }
  }

  // Keep only substantial, enclosed regions with an edge-rich boundary.
  const keep=new Uint8Array(field.length),seen=new Uint8Array(field.length);
  for(let start=0;start<grown.length;start++){
    if(!grown[start]||seen[start])continue;
    const component:number[]=[start];seen[start]=1;let boundary=0,edgeSum=0,touchesFrame=false;
    for(let q=0;q<component.length;q++){
      const i=component[q],x=i%W,y=Math.floor(i/W);if(x<=1||x>=W-2||y<=1||y>=H-2)touchesFrame=true;
      for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){const xx=x+dx,yy=y+dy;if(xx<0||xx>=W||yy<0||yy>=H)continue;const n=yy*W+xx;if(grown[n]&&!seen[n]){seen[n]=1;component.push(n)}else if(!grown[n]){boundary++;edgeSum+=gradient[n]}}
    }
    const rimStrength=edgeSum/Math.max(1,boundary),area=component.length;
    if(!touchesFrame&&area>=8&&area<field.length*.36&&rimStrength>.055)for(const i of component)keep[i]=1;
  }

  const carved=new Float32Array(field);
  for(let i=0;i<carved.length;i++)if(keep[i])carved[i]=Math.min(carved[i],.045);
  return carved;
}

function drawVoidPerimeters(ctx:CanvasRenderingContext2D,field:Float32Array){
  const cut=.14;ctx.beginPath();
  for(let y=1;y<H-1;y++)for(let x=1;x<W-1;x++){
    const i=y*W+x;if(field[i]>=cut)continue;
    if(field[i-1]>=cut||field[i+1]>=cut||field[i-W]>=cut||field[i+W]>=cut)ctx.rect(x-.12,y-.12,.24,.24);
  }
  ctx.stroke();
}
function transformVoids(field:Float32Array,amount:number){
  if(amount===0)return field;
  const radius=Math.max(1,Math.ceil(Math.abs(amount)/20)),blend=Math.abs(amount)/100,expanded=new Float32Array(field.length);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    let extreme=amount>0?1:0;
    for(let dy=-radius;dy<=radius;dy++)for(let dx=-radius;dx<=radius;dx++){
      if(dx*dx+dy*dy>radius*radius)continue;
      const xx=Math.max(0,Math.min(W-1,x+dx)),yy=Math.max(0,Math.min(H-1,y+dy)),sample=field[yy*W+xx];
      extreme=amount>0?Math.min(extreme,sample):Math.max(extreme,sample);
    }
    const original=field[y*W+x];
    expanded[y*W+x]=original*(1-blend)+extreme*blend;
  }
  return expanded;
}
function sliceImage(field:Float32Array,z:number):Slice{
  const image=document.createElement("canvas");image.width=W;image.height=H;
  const ctx=image.getContext("2d")!,pixels=ctx.createImageData(W,H);
  for(let i=0;i<field.length;i++){const k=i*4,d=field[i],g=Math.round(120+130*Math.sqrt(d));pixels.data[k]=g;pixels.data[k+1]=Math.min(255,g+5);pixels.data[k+2]=Math.min(255,g+8);pixels.data[k+3]=Math.round(255*Math.pow(d,1.25))}
  ctx.putImageData(pixels,0,0);return{field,z,image};
}
function interpolate(fields:Float32Array[],interval:number,continuity:number){
  if(!fields.length)return[] as Slice[];
  if(fields.length===1)return[sliceImage(fields[0],0)];
  const generated:{field:Float32Array;z:number}[]=[],curveMix=continuity/100;
  for(let i=0;i<fields.length-1;i++)for(let j=0;j<steps;j++){
    const t=j/steps,t2=t*t,t3=t2*t,f=new Float32Array(W*H),p0=fields[Math.max(0,i-1)],p1=fields[i],p2=fields[i+1],p3=fields[Math.min(fields.length-1,i+2)];
    for(let p=0;p<f.length;p++){
      const linear=p1[p]*(1-t)+p2[p]*t;
      const cubic=.5*((2*p1[p])+(-p0[p]+p2[p])*t+(2*p0[p]-5*p1[p]+4*p2[p]-p3[p])*t2+(-p0[p]+3*p1[p]-3*p2[p]+p3[p])*t3);
      f[p]=Math.max(0,Math.min(1,linear*(1-curveMix)+cubic*curveMix));
    }
    generated.push({field:f,z:((fields.length-1)/2-i-t)*interval});
  }
  generated.push({field:new Float32Array(fields.at(-1)!),z:(-(fields.length-1)/2)*interval});
  const passes=Math.round(continuity/28),strength=.18+curveMix*.42;
  for(let pass=0;pass<passes;pass++){
    const next=generated.map(item=>new Float32Array(item.field));
    for(let z=1;z<generated.length-1;z++)for(let p=0;p<W*H;p++){
      const average=(generated[z-1].field[p]+generated[z].field[p]*2+generated[z+1].field[p])/4;
      next[z][p]=generated[z].field[p]*(1-strength)+average*strength;
    }
    for(let z=1;z<generated.length-1;z++)generated[z].field=next[z];
  }
  return generated.map(item=>sliceImage(item.field,item.z));
}

function rationalizeSlices(slices:Slice[],settings:RealitySettings){
  if(!slices.length)return slices;
  const threshold=.32,groundY=H-3,plateThickness=2;
  const supportStep=Math.max(4,Math.round(settings.supportSpacing/2.5));
  const clearRadius=Math.max(1,Math.round(settings.clearance/3));
  return slices.map((slice,sliceIndex)=>{
    const grounded=new Float32Array(W*H);let bottom=-1;
    for(let y=0;y<H;y++)for(let x=0;x<W;x++)if(slice.field[y*W+x]>=threshold)bottom=Math.max(bottom,y);
    const shift=bottom>=0?Math.max(0,groundY-bottom):0;
    for(let y=0;y<H;y++)for(let x=0;x<W;x++){const yy=y+shift;if(yy<H)grounded[yy*W+x]=slice.field[y*W+x]}

    // Ground-bearing supports: regular piers extend the nearest substantial
    // mass down to the datum instead of allowing isolated geometry to float.
    for(let x=2;x<W-2;x+=supportStep){let bearing=-1;
      for(let y=0;y<=groundY;y++)if(grounded[y*W+x]>=threshold||grounded[y*W+x-1]>=threshold||grounded[y*W+x+1]>=threshold)bearing=y;
      if(bearing>=0)for(let y=bearing;y<=groundY;y++)for(let dx=-1;dx<=1;dx++)grounded[y*W+x+dx]=Math.max(grounded[y*W+x+dx],.44);
    }

    // Insert one continuous occupiable floor only. Its small section-to-section
    // offset makes a walkable ground surface, never a stack of building levels.
    // Openings crossing that surface remain clear as circulation thresholds.
    const rampOffset=Math.round(((sliceIndex/Math.max(1,slices.length-1))-.5)*4*(12/settings.maxSlope));
    const fy=groundY-rampOffset;let minX=W,maxX=-1;
    for(let x=1;x<W-1;x++)for(let y=Math.max(1,fy-5);y<=Math.min(H-2,fy+3);y++)if(grounded[y*W+x]>=threshold){minX=Math.min(minX,x);maxX=Math.max(maxX,x)}
    if(maxX-minX>=8){
      for(let y=fy;y<Math.min(H,fy+plateThickness);y++)for(let x=minX;x<=maxX;x++)grounded[y*W+x]=Math.max(grounded[y*W+x],.4);
      for(let x=minX+1;x<maxX;x++)if(grounded[Math.max(0,fy-3)*W+x]<.18){
        for(let yy=Math.max(1,fy-6);yy<fy;yy++)for(let xx=Math.max(minX,x-clearRadius);xx<=Math.min(maxX,x+clearRadius);xx++)grounded[yy*W+xx]=Math.min(grounded[yy*W+xx],.04);
      }
    }
    return sliceImage(grounded,slice.z);
  });
}

function gridAlignSlices(slices:Slice[],gridSize:number,alignment:number){
  const step=Math.max(3,Math.round(gridSize)),strength=Math.max(0,Math.min(1,alignment/100));
  const sample=(field:Float32Array,x:number,y:number)=>{const x0=Math.max(0,Math.min(W-1,Math.floor(x))),y0=Math.max(0,Math.min(H-1,Math.floor(y))),x1=Math.min(W-1,x0+1),y1=Math.min(H-1,y0+1),tx=x-x0,ty=y-y0;return field[y0*W+x0]*(1-tx)*(1-ty)+field[y0*W+x1]*tx*(1-ty)+field[y1*W+x0]*(1-tx)*ty+field[y1*W+x1]*tx*ty};
  return slices.map(slice=>{
    let aligned=new Float32Array(W*H);
    for(let y=0;y<H;y++)for(let x=0;x<W;x++){
      const gx=Math.round(x/step)*step,gy=Math.round(y/step)*step,sx=x+(gx-x)*strength,sy=y+(gy-y)*strength;
      aligned[y*W+x]=sample(slice.field,sx,sy);
    }
    // A light relaxation pass removes hard snap seams while retaining the
    // shared grid registration of exterior and void boundaries.
    const relaxed=new Float32Array(aligned);
    const relax=.08+.22*strength;
    for(let y=1;y<H-1;y++)for(let x=1;x<W-1;x++){let sum=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)sum+=aligned[(y+dy)*W+x+dx];relaxed[y*W+x]=aligned[y*W+x]*(1-relax)+sum/9*relax}
    aligned=relaxed;return sliceImage(aligned,slice.z);
  });
}
function contours(ctx:CanvasRenderingContext2D,field:Float32Array){
  ctx.beginPath();const cut=.3;
  for(let y=0;y<H-1;y++)for(let x=0;x<W-1;x++){
    const v=[field[y*W+x],field[y*W+x+1],field[(y+1)*W+x+1],field[(y+1)*W+x]];
    const p:[[number,number],[number,number],[number,number],[number,number]]=[[x,y],[x+1,y],[x+1,y+1],[x,y+1]];
    const hits:[number,number][]=[];
    for(let e=0;e<4;e++){const n=(e+1)%4;if((v[e]>=cut)===(v[n]>=cut))continue;const t=(cut-v[e])/(v[n]-v[e]);hits.push([p[e][0]+t*(p[n][0]-p[e][0]),p[e][1]+t*(p[n][1]-p[e][1])])}
    for(let k=0;k+1<hits.length;k+=2){ctx.moveTo(...hits[k]);ctx.lineTo(...hits[k+1])}
  }
  ctx.stroke();
}
function exportObj(slices:Slice[]){
  if(!slices.length)return;
  const nx=30,ny=23,threshold=.32;
  const layers=[{field:null as Float32Array|null,z:slices[0].z+.3},...slices.map(s=>({field:s.field,z:s.z})),{field:null as Float32Array|null,z:slices.at(-1)!.z-.3}];
  let obj="# Smoothed photo-derived CT volume; arbitrary units\n",index=1;
  const tetra=[[0,5,1,6],[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6]];
  const add=(a:Vec,b:Vec,c:Vec)=>{for(const [x,y,z] of [a,b,c])obj+=`v ${x.toFixed(4)} ${y.toFixed(4)} ${z.toFixed(4)}\n`;obj+=`f ${index} ${index+1} ${index+2}\n`;index+=3};
  for(let z=0;z<layers.length-1;z++)for(let y=0;y<ny-1;y++)for(let x=0;x<nx-1;x++){
    const corners:[[number,number,number],[number,number,number],[number,number,number],[number,number,number],[number,number,number],[number,number,number],[number,number,number],[number,number,number]]=[[x,y,z],[x+1,y,z],[x+1,y+1,z],[x,y+1,z],[x,y,z+1],[x+1,y,z+1],[x+1,y+1,z+1],[x,y+1,z+1]];
    const values=corners.map(([cx,cy,cz])=>layers[cz].field?.[Math.round(cy*(H-1)/(ny-1))*W+Math.round(cx*(W-1)/(nx-1))]??0);
    if(values.every(v=>v<threshold)||values.every(v=>v>=threshold))continue;
    const point=(i:number):Vec=>[(corners[i][0]/(nx-1)-.5)*2.25,(.5-corners[i][1]/(ny-1))*1.7,layers[corners[i][2]].z];
    const cross=(a:number,b:number):Vec=>{const t=(threshold-values[a])/(values[b]-values[a]),A=point(a),B=point(b);return[A[0]+t*(B[0]-A[0]),A[1]+t*(B[1]-A[1]),A[2]+t*(B[2]-A[2])]};
    for(const ids of tetra){const inside=ids.filter(i=>values[i]>=threshold),outside=ids.filter(i=>values[i]<threshold);if(!inside.length||!outside.length)continue;
      if(inside.length===1){const p=outside.map(o=>cross(inside[0],o));add(p[0],p[1],p[2])}
      else if(outside.length===1){const p=inside.map(i=>cross(i,outside[0]));add(p[0],p[1],p[2])}
      else{const p=inside.flatMap(i=>outside.map(o=>cross(i,o)));add(p[0],p[1],p[2]);add(p[1],p[3],p[2])}
    }
  }
  void saveExport(new Blob([obj],{type:"text/plain"}),"ct-volume-study.obj");
}

type SectionBoundary={y:number;kind:"solid-start"|"solid-end"};
function boundaryProfiles(slices:Slice[]){
  const nx=32,threshold=.32;
  return slices.map(slice=>Array.from({length:nx},(_,x)=>{
    const sx=Math.round(x*(W-1)/(nx-1)),boundaries:SectionBoundary[]=[];
    let inside=false;
    for(let y=0;y<H;y++){
      const value=slice.field[y*W+sx],next=value>=threshold;
      if(next!==inside){
        const previous=y? slice.field[(y-1)*W+sx]:0;
        const fraction=y?Math.max(0,Math.min(1,(threshold-previous)/(value-previous||1))):0;
        const crossing=Math.max(0,y-1+fraction);
        boundaries.push({y:.5-crossing/(H-1),kind:next?"solid-start":"solid-end"});inside=next;
      }
    }
    if(inside)boundaries.push({y:-.5,kind:"solid-end"});
    return boundaries;
  }));
}

function outerProfiles(boundaries:SectionBoundary[][][]){
  return boundaries.map(slice=>slice.map(column=>{
    const top=column[0]?.y??.015,bottom=column.at(-1)?.y??-.015;
    return{top,bottom};
  }));
}

function downloadRhinoFile(file:any,rhino:any,version:7|8,filename:string){
  const options=new rhino.File3dmWriteOptions();options.version=version;options.saveUserData=true;
  const bytes=file.toByteArrayOptions(options);
  void saveExport(new Blob([bytes],{type:"application/octet-stream"}),filename);options.delete();
}

async function exportRhinoSubd(slices:Slice[],version:7|8){
  if(slices.length<2)return;
  const [{default:rhino3dm},{default:wasmUrl}]=await Promise.all([import("rhino3dm"),import("rhino3dm/rhino3dm.wasm?url")]);
  const rhino=await rhino3dm({locateFile:()=>wasmUrl});
  const boundaries=boundaryProfiles(slices),nx=boundaries[0].length,nz=boundaries.length;
  const mesh=new rhino.Mesh(),vertices=mesh.vertices(),faces=mesh.faces();
  const cache=new Map<string,number>();
  const vertex=(z:number,x:number,b:number)=>{
    const key=`${z}:${x}:${b}`,existing=cache.get(key);if(existing!==undefined)return existing;
    const p=boundaries[z][x][b];if(!p)return-1;
    const index=vertices.addPoint3d((x/(nx-1)-.5)*2.25,p.y*1.7,slices[z].z);cache.set(key,index);return index;
  };
  // Loft every threshold crossing, not only the first and last perimeter.
  for(let z=0;z<nz-1;z++)for(let x=0;x<nx-1;x++){
    const count=Math.min(boundaries[z][x].length,boundaries[z][x+1].length,boundaries[z+1][x].length,boundaries[z+1][x+1].length);
    for(let b=0;b<count;b++){
      const a=vertex(z,x,b),c=vertex(z,x+1,b),d=vertex(z+1,x+1,b),e=vertex(z+1,x,b);
      if(a>=0&&c>=0&&d>=0&&e>=0)faces.addQuadFace(a,c,d,e);
    }
  }
  // Close the left and right sides of every solid band while leaving the gaps open.
  for(let z=0;z<nz-1;z++){
    for(const x of [0,nx-1]){
      const count=Math.min(boundaries[z][x].length,boundaries[z+1][x].length);
      for(let b=0;b+1<count;b+=2){const a=vertex(z,x,b),c=vertex(z,x,b+1),d=vertex(z+1,x,b+1),e=vertex(z+1,x,b);if(a>=0&&c>=0&&d>=0&&e>=0)faces.addQuadFace(a,c,d,e)}
    }
  }
  // Cap only the material bands at the first and last section; void bands stay open.
  for(let x=0;x<nx-1;x++){
    for(const z of [0,nz-1]){
      const count=Math.min(boundaries[z][x].length,boundaries[z][x+1].length);
      for(let b=0;b+1<count;b+=2){const a=vertex(z,x,b),c=vertex(z,x+1,b),d=vertex(z,x+1,b+1),e=vertex(z,x,b+1);if(a>=0&&c>=0&&d>=0&&e>=0)faces.addQuadFace(a,c,d,e)}
    }
  }
  mesh.normals().computeNormals();
  mesh.setUserString("Form","Multi-boundary photo-derived CT SubD");
  mesh.setUserString("RhinoWorkflow","Quad boundary loft preserving internal void bands. Run ToSubD with Creases=No.");
  mesh.setUserString("SourceSections",String(slices.length));
  const file=new rhino.File3dm();file.applicationName="Sand Scan";file.applicationDetails=`Photo to CT Volume · Rhino ${version} SubD massing export`;
  file.startSectionComments=`Editable multi-boundary CT loft written for Rhino ${version}. Outer and internal section boundaries are preserved as quad surfaces. Select CT_SUBD_MASSING and run ToSubD with Creases=No.`;
  const layerIndex=file.layers().addLayer("CT_SUBD_MASSING",{r:124,g:205,b:194,a:255});
  const attributes=new rhino.ObjectAttributes();attributes.name="CT_SUBD_MASSING";attributes.layerIndex=layerIndex;
  attributes.setUserString("NextCommand","_ToSubD _Creases=_No");
  file.objects().addMesh(mesh,attributes);
  downloadRhinoFile(file,rhino,version,`sand-scan-subd-rhino${version}.3dm`);
  attributes.delete();mesh.delete();file.delete();
}

async function exportRhinoSections(slices:Slice[],version:7|8){
  if(!slices.length)return;
  const [{default:rhino3dm},{default:wasmUrl}]=await Promise.all([import("rhino3dm"),import("rhino3dm/rhino3dm.wasm?url")]);
  const rhino=await rhino3dm({locateFile:()=>wasmUrl}),file=new rhino.File3dm();
  file.applicationName="Sand Scan";file.applicationDetails=`CT section surfaces · Rhino ${version}`;
  file.startSectionComments=`Individual planar section surfaces written for Rhino ${version}. Each named mesh remains independent; open cells preserve the detected voids.`;
  const layerIndex=file.layers().addLayer("CT_SECTION_SURFACES",{r:242,g:105,b:82,a:255}),nx=48,ny=36,threshold=.32;
  slices.forEach((slice,index)=>{
    const mesh=new rhino.Mesh(),vertices=mesh.vertices(),faces=mesh.faces();
    for(let y=0;y<ny-1;y++)for(let x=0;x<nx-1;x++){
      const sx=Math.round((x+.5)*(W-1)/(nx-1)),sy=Math.round((y+.5)*(H-1)/(ny-1));if(slice.field[sy*W+sx]<threshold)continue;
      const x0=(x/(nx-1)-.5)*2.25,x1=((x+1)/(nx-1)-.5)*2.25,y0=(.5-y/(ny-1))*1.7,y1=(.5-(y+1)/(ny-1))*1.7;
      const a=vertices.addPoint3d(x0,y0,slice.z),b=vertices.addPoint3d(x1,y0,slice.z),c=vertices.addPoint3d(x1,y1,slice.z),d=vertices.addPoint3d(x0,y1,slice.z);faces.addQuadFace(a,b,c,d);
    }
    mesh.normals().computeNormals();mesh.setUserString("GeometryType","Independent CT section surface");mesh.setUserString("SectionIndex",String(index+1));mesh.setUserString("Depth",slice.z.toFixed(5));
    const attributes=new rhino.ObjectAttributes();attributes.name=`SECTION_${String(index+1).padStart(3,"0")}`;attributes.layerIndex=layerIndex;attributes.setUserString("Source","Smoothed SubD section field");file.objects().addMesh(mesh,attributes);attributes.delete();mesh.delete();
  });
  downloadRhinoFile(file,rhino,version,`sand-scan-surfaces-rhino${version}.3dm`);file.delete();
}

type Point2=[number,number];
const pointDistance=(a:Point2,b:Point2)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
function loopArea(points:Point2[]){
  let area=0;for(let i=0;i<points.length-1;i++)area+=points[i][0]*points[i+1][1]-points[i+1][0]*points[i][1];return Math.abs(area)*.5;
}
function loopPerimeter(points:Point2[]){
  let length=0;for(let i=0;i<points.length-1;i++)length+=pointDistance(points[i],points[i+1]);return length;
}
function sectionContourLoops(field:Float32Array,threshold=.32){
  const segments:[Point2,Point2][]=[],key=([x,y]:Point2)=>`${x.toFixed(5)}:${y.toFixed(5)}`;
  for(let y=0;y<H-1;y++)for(let x=0;x<W-1;x++){
    const values=[field[y*W+x],field[y*W+x+1],field[(y+1)*W+x+1],field[(y+1)*W+x]],corners:Point2[]=[[x,y],[x+1,y],[x+1,y+1],[x,y+1]],hits:Point2[]=[];
    for(let edge=0;edge<4;edge++){
      const next=(edge+1)%4;if((values[edge]>=threshold)===(values[next]>=threshold))continue;
      const t=(threshold-values[edge])/(values[next]-values[edge]);hits.push([corners[edge][0]+t*(corners[next][0]-corners[edge][0]),corners[edge][1]+t*(corners[next][1]-corners[edge][1])]);
    }
    if(hits.length===2)segments.push([hits[0],hits[1]]);
    else if(hits.length===4){segments.push([hits[0],hits[1]]);segments.push([hits[2],hits[3]])}
  }
  const endpoints=new Map<string,{segment:number;end:0|1}[]>();
  segments.forEach((segment,index)=>segment.forEach((point,end)=>{const k=key(point),items=endpoints.get(k)??[];items.push({segment:index,end:end as 0|1});endpoints.set(k,items)}));
  const used=new Uint8Array(segments.length),loops:Point2[][]=[];
  for(let start=0;start<segments.length;start++){
    if(used[start])continue;used[start]=1;const points=[segments[start][0],segments[start][1]];
    const extend=(atStart:boolean)=>{
      while(true){const point=atStart?points[0]:points.at(-1)!,match=(endpoints.get(key(point))??[]).find(item=>!used[item.segment]);if(!match)break;
        used[match.segment]=1;const segment=segments[match.segment],next=segment[match.end===0?1:0];if(atStart)points.unshift(next);else points.push(next);
      }
    };
    extend(false);extend(true);
    // Open chains are marching-square artifacts, not usable section boundaries.
    // Also reject tiny closed islands so texture and compression noise never
    // becomes a Rhino curve or a distracting line in the 3D stack.
    if(points.length>3&&key(points[0])===key(points.at(-1)!)){
      points[points.length-1]=points[0];
      const xs=points.map(point=>point[0]),ys=points.map(point=>point[1]);
      const spanX=Math.max(...xs)-Math.min(...xs),spanY=Math.max(...ys)-Math.min(...ys);
      if(loopArea(points)>=4.5&&loopPerimeter(points)>=7&&Math.max(spanX,spanY)>=2.6&&Math.min(spanX,spanY)>=1.15)loops.push(points);
    }
  }
  return loops;
}

function contourFidelity(points:Point2[],fidelity:number){
  if(points.length<5)return points;
  const closed=points[0][0]===points.at(-1)![0]&&points[0][1]===points.at(-1)![1],source=closed?points.slice(0,-1):points;
  const stride=Math.max(1,Math.round(1+(100-fidelity)/11)),sampled=source.filter((_,index)=>index%stride===0);
  if(sampled.length<3)return points;
  // Fidelity controls simplification, not cleanliness. Even at 100%, apply a
  // light cyclic fairing pass so pixel stair-steps and one-cell hooks disappear.
  let result=sampled.map(point=>[point[0],point[1]] as Point2);
  const smoothing=.18+(100-fidelity)/100*.34,passes=fidelity<45?3:2;
  for(let pass=0;pass<passes;pass++)result=result.map((point,index)=>{
    const before=result[(index-1+result.length)%result.length],after=result[(index+1)%result.length];
    return [point[0]*(1-smoothing)+(before[0]+after[0])*.5*smoothing,point[1]*(1-smoothing)+(before[1]+after[1])*.5*smoothing] as Point2;
  });
  if(closed)result.push(result[0]);return result;
}

function keepLargestConnectedField(field:Float32Array,threshold=.32){
  const visited=new Uint8Array(W*H),components:number[][]=[];
  for(let start=0;start<field.length;start++){
    if(visited[start]||field[start]<threshold)continue;
    const component:number[]=[],queue=[start];visited[start]=1;
    for(let cursor=0;cursor<queue.length;cursor++){
      const index=queue[cursor],x=index%W,y=Math.floor(index/W);component.push(index);
      const neighbors=[x>0?index-1:-1,x<W-1?index+1:-1,y>0?index-W:-1,y<H-1?index+W:-1];
      for(const next of neighbors)if(next>=0&&!visited[next]&&field[next]>=threshold){visited[next]=1;queue.push(next)}
    }
    components.push(component);
  }
  const largest=components.sort((a,b)=>b.length-a.length)[0]??[],result=new Float32Array(field.length);
  for(const index of largest)result[index]=field[index];
  return result;
}

type ArchitecturalAnchor={point:Point2;role:string};
function architecturalAnchors(section:Point2[][],targetCount:number){
  const ranked=[...section].sort((a,b)=>loopArea(b)-loopArea(a)),outer=ranked[0];
  if(!outer)return[] as ArchitecturalAnchor[];
  let source=outer.length>1&&outer[0][0]===outer.at(-1)![0]&&outer[0][1]===outer.at(-1)![1]?outer.slice(0,-1):outer;
  if(source.length<3)return[] as ArchitecturalAnchor[];
  // Every section uses one identical point system: clockwise arc-length
  // stations around the dominant perimeter. The start station is always the
  // uppermost point, so row_00 in every section describes the same place in
  // the perimeter sequence instead of a different architectural category.
  const signed=source.reduce((area,point,index)=>{const next=source[(index+1)%source.length];return area+point[0]*next[1]-next[0]*point[1]},0);
  if(signed<0)source=[...source].reverse();
  const start=source.reduce((best,point,index)=>point[1]<source[best][1]||(point[1]===source[best][1]&&point[0]<source[best][0])?index:best,0);
  source=[...source.slice(start),...source.slice(0,start)];
  const closed=[...source,source[0]],lengths=[0];for(let index=1;index<closed.length;index++)lengths.push(lengths[index-1]+pointDistance(closed[index-1],closed[index]));
  const total=lengths.at(-1)??1;
  return Array.from({length:targetCount},(_,row)=>{const distance=total*row/targetCount;let segment=1;while(segment<lengths.length-1&&lengths[segment]<distance)segment++;const a=closed[segment-1],b=closed[segment],span=Math.max(.0001,lengths[segment]-lengths[segment-1]),t=(distance-lengths[segment-1])/span;return{role:`row_${String(row).padStart(2,"0")}`,point:[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t] as Point2}});
}

type RuledStrip={a:Vec[];b:Vec[];kind:"outer"|"void"};
type PointOffsets=Record<string,Vec>;
type LongitudinalFamily={primary:Vec[][];primaryRoles:string[];offset:Vec[][];tweens:Vec[][];branches:Vec[][];branchPlates:RuledStrip[];ruled:RuledStrip[];anchors:{point:Vec;role:string}[]};
type CirculationSettings={primaryCount:number;pathWidth:number;activeBranches:number;junctionWidth:number;smoothing:number};
type CirculationRibbon={center:Vec[];left:Vec[];right:Vec[];kind:"spine"|"branch"};
type CirculationGeometry={ribbons:CirculationRibbon[];floorY:number;topY:number;trimmedCount:number};
const CHUNK_HALF=1.05;

function smoothSpatialRoute(line:Vec[],amount:number){
  let result=line.filter((point,index)=>index===0||Math.hypot(point[0]-line[index-1][0],point[1]-line[index-1][1],point[2]-line[index-1][2])>.025);
  const passes=Math.round(Math.max(0,Math.min(100,amount))/20),weight=.34;
  for(let pass=0;pass<passes;pass++)result=result.map((point,index)=>index===0||index===result.length-1?point:[point[0]*(1-weight)+(result[index-1][0]+result[index+1][0])*.5*weight,point[1]*(1-weight)+(result[index-1][1]+result[index+1][1])*.5*weight,point[2]*(1-weight)+(result[index-1][2]+result[index+1][2])*.5*weight] as Vec);
  return result;
}

function buildCirculationGeometry(families:LongitudinalFamily,settings:CirculationSettings,heightInches:number,bifurcationsPerPrimary:number,includeBranches=true):CirculationGeometry{
  const floorY=-CHUNK_HALF+.08,heightFeet=heightInches/12,topY=Math.min(CHUNK_HALF,floorY+heightFeet/20*(CHUNK_HALF*2));
  if(!families.primary.length)return{ribbons:[],floorY,topY,trimmedCount:0};
  const selectedCount=Math.max(1,Math.min(families.primary.length,settings.primaryCount));
  const selectedIndices=Array.from({length:selectedCount},(_,index)=>selectedCount===1?0:Math.round(index*(families.primary.length-1)/(selectedCount-1)));
  const sources:[Vec[],"spine"|"branch"][]=[];
  selectedIndices.forEach(primaryIndex=>{
    sources.push([families.primary[primaryIndex],"spine"]);
    if(includeBranches){const branchStart=primaryIndex*Math.max(0,bifurcationsPerPrimary);families.branches.slice(branchStart,branchStart+Math.max(0,settings.activeBranches)).forEach(line=>sources.push([line,"branch"]))}
  });
  let trimmedCount=0;
  const ribbons=sources.map(([source,kind])=>{
    const spatial=source.map(([x,y,z])=>{if(Math.abs(x)>CHUNK_HALF||Math.abs(z)>CHUNK_HALF||y<-.85||y>.85)trimmedCount++;const vertical=(Math.max(-.85,Math.min(.85,y))+.85)/1.7;return[Math.max(-CHUNK_HALF,Math.min(CHUNK_HALF,x)),floorY+vertical*(topY-floorY),Math.max(-CHUNK_HALF,Math.min(CHUNK_HALF,z))] as Vec});
    const center=smoothSpatialRoute(spatial,settings.smoothing),baseHalf=Math.max(.12,settings.pathWidth/20*CHUNK_HALF);
    const left:Vec[]=[],right:Vec[]=[];
    center.forEach((point,index)=>{const before=center[Math.max(0,index-1)],after=center[Math.min(center.length-1,index+1)],tx=after[0]-before[0],tz=after[2]-before[2],length=Math.hypot(tx,tz),nx=length>.0001?-tz/length:1,nz=length>.0001?tx/length:0,progress=index/Math.max(1,center.length-1),junction=Math.pow(Math.sin(Math.PI*progress),2),widen=1+settings.junctionWidth/100*(kind==="branch"?.55:.32)*junction,half=baseHalf*widen;left.push([Math.max(-CHUNK_HALF,Math.min(CHUNK_HALF,point[0]+nx*half)),point[1],Math.max(-CHUNK_HALF,Math.min(CHUNK_HALF,point[2]+nz*half))]);right.push([Math.max(-CHUNK_HALF,Math.min(CHUNK_HALF,point[0]-nx*half)),point[1],Math.max(-CHUNK_HALF,Math.min(CHUNK_HALF,point[2]-nz*half))])});
    return{center,left,right,kind};
  }).filter(ribbon=>ribbon.center.length>1);
  return{ribbons,floorY,topY,trimmedCount};
}

function longitudinalFamilies(slices:Slice[],loops:Point2[][][],primaryCount:number,smoothing:number,offsetsPerPrimary:number,bifurcationsPerPrimary:number,bifurcationDistance:number,bifurcationScale:number,branchDirections:number[],pointOffsets:PointOffsets):LongitudinalFamily{
  const sampled=loops.map((section,index)=>new Map(architecturalAnchors(section,primaryCount).map(anchor=>[anchor.role,[(anchor.point[0]/(W-1)-.5)*2.25,(.5-anchor.point[1]/(H-1))*1.7,slices[index].z] as Vec])));
  const roles=Array.from({length:primaryCount},(_,index)=>`row_${String(index).padStart(2,"0")}`);
  const minimumRowLength=Math.max(2,Math.ceil(slices.length*.6));
  const entries=roles.map(role=>({role,line:sampled.map(section=>section.get(role)).filter(Boolean) as Vec[]})).filter(entry=>entry.line.length>=minimumRowLength).map(entry=>({
    ...entry,
    // Fair each role as one ordered longitudinal row. Every displayed point is
    // now part of a continuous row rather than an isolated sectional sample.
    line:entry.line.map((point,index,line)=>{
      if(index===0||index===line.length-1)return point;
      const before=line[index-1],after=line[index+1];
      const fair=Math.max(0,Math.min(.48,smoothing/100*.48));return[point[0]*(1-fair)+(before[0]+after[0])*.5*fair,point[1]*(1-fair)+(before[1]+after[1])*.5*fair,point[2]] as Vec;
    })
  }));
  const primary=entries.map(entry=>entry.line),primaryRoles=entries.map(entry=>entry.role),offset:Vec[][]=[],ordered:Vec[][]=[];
  // Each primary line owns an identical number of interpolated offset lines
  // toward the next primary. They all inherit the same section depths, so the
  // hierarchy remains one ordered perimeter system rather than separate cages.
  for(let index=0;index<entries.length;index++){
    const left=entries[index].line,right=entries[(index+1)%entries.length].line,rightByDepth=new Map(right.map(point=>[point[2].toFixed(6),point]));ordered.push(left);
    for(let offsetIndex=1;offsetIndex<=offsetsPerPrimary;offsetIndex++){
      const t=offsetIndex/(offsetsPerPrimary+1),line=left.map(point=>{const match=rightByDepth.get(point[2].toFixed(6));return match?[point[0]+(match[0]-point[0])*t,point[1]+(match[1]-point[1])*t,point[2]] as Vec:null}).filter((point):point is Vec=>Boolean(point));
      if(line.length>=minimumRowLength){offset.push(line);ordered.push(line)}
    }
  }
  const ruled:RuledStrip[]=[];
  for(let index=0;index<ordered.length;index++){
      const left=ordered[index],right=ordered[(index+1)%ordered.length],rightByDepth=new Map(right.map(point=>[point[2].toFixed(6),point]));
      const pairs=left.map(point=>[point,rightByDepth.get(point[2].toFixed(6))] as const).filter((pair):pair is readonly[Vec,Vec]=>Boolean(pair[1]));
      if(pairs.length>=minimumRowLength)ruled.push({a:pairs.map(pair=>pair[0]),b:pairs.map(pair=>pair[1]),kind:"outer"});
  }
  const branches:Vec[][]=[],branchPlates:RuledStrip[]=[];
  entries.forEach((entry,primaryIndex)=>{
    for(let branchIndex=0;branchIndex<bifurcationsPerPrimary;branchIndex++){
      const branchId=primaryIndex*bifurcationsPerPrimary+branchIndex,steering=(branchDirections[branchId]??0)*Math.PI/180,cs=Math.cos(steering),sn=Math.sin(steering);
      // Every branch in a family follows the same neighboring rail. It leaves
      // and rejoins its primary at the ends, while the scale exponent makes
      // successive branches fan outward with increasingly larger gaps.
      const target=entries[(primaryIndex+1)%entries.length],targetByDepth=new Map(target.line.map(point=>[point[2].toFixed(6),point]));
      const distanceFactor=Math.max(.05,Math.min(1,bifurcationDistance/100)),scaleExponent=1+Math.max(0,Math.min(100,bifurcationScale))/50;
      const level=Math.pow((branchIndex+1)/Math.max(1,bifurcationsPerPrimary),scaleExponent),reach=Math.min(.95,distanceFactor*(.28+level*.72));
      const branch=entry.line.map((point,index,line)=>{const match=targetByDepth.get(point[2].toFixed(6));if(!match)return point;const progress=index/Math.max(1,line.length-1),envelope=Math.pow(Math.sin(Math.PI*progress),1.2),spread=envelope*reach,dx=match[0]-point[0],dy=match[1]-point[1],rx=dx*cs-dy*sn,ry=dx*sn+dy*cs,offset=pointOffsets[`${branchId}:${index}`]??[0,0,0];return[point[0]+rx*spread+offset[0],point[1]+ry*spread+offset[1],point[2]+offset[2]] as Vec});
      if(branch.length>=minimumRowLength){branches.push(branch);branchPlates.push({a:entry.line,b:branch,kind:"outer"})}
    }
  });
  return{primary,primaryRoles,offset,tweens:[],branches,branchPlates,ruled,anchors:entries.flatMap(entry=>entry.line.map(point=>({role:entry.role,point})))};
}

async function exportRhinoSectionLines(slices:Slice[],version:7|8,fidelity:number){
  if(!slices.length)return;
  const [{default:rhino3dm},{default:wasmUrl}]=await Promise.all([import("rhino3dm"),import("rhino3dm/rhino3dm.wasm?url")]);
  const rhino=await rhino3dm({locateFile:()=>wasmUrl}),file=new rhino.File3dm();
  file.applicationName="Sand Scan";file.applicationDetails=`CT section boundary curves · Rhino ${version}`;
  file.startSectionComments=`Joined section contours written for Rhino ${version}. Exterior perimeters and internal void boundaries are editable curve objects; no pixels or meshes are included.`;
  const layerIndex=file.layers().addLayer("CT_SECTION_BOUNDARY_LINES",{r:139,g:217,b:207,a:255});let curveCount=0;
  slices.forEach((slice,sectionIndex)=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity)).forEach((loop,boundaryIndex)=>{
    const points=loop.map(([x,y])=>[(x/(W-1)-.5)*2.25,(.5-y/(H-1))*1.7,slice.z]),curve=new rhino.PolylineCurve(points),attributes=new rhino.ObjectAttributes();
    attributes.name=`SECTION_${String(sectionIndex+1).padStart(3,"0")}_BOUNDARY_${String(boundaryIndex+1).padStart(2,"0")}`;attributes.layerIndex=layerIndex;
    attributes.setUserString("GeometryType","Joined CT section boundary curve");attributes.setUserString("SectionIndex",String(sectionIndex+1));attributes.setUserString("Depth",slice.z.toFixed(5));attributes.setUserString("BoundaryRole","Exterior perimeter or enclosed void");attributes.setUserString("CurveFidelity",String(fidelity));
    file.objects().addCurve(curve,attributes);curveCount++;attributes.delete();curve.delete();
  }));
  file.startSectionComments+=` ${curveCount} joined boundary curves exported from ${slices.length} sections.`;
  downloadRhinoFile(file,rhino,version,`sand-scan-boundary-lines-rhino${version}.3dm`);file.delete();
}

const EXPORT_TILE_INCHES=240;
function exportDepth(slices:Slice[],index:number,height:number){
  if(slices.length<2)return height*.5;
  const first=slices[0].z,last=slices.at(-1)!.z,span=last-first||1;
  return(slices[index].z-first)/span*height;
}

function addPrintableForm(file:any,rhino:any,slices:Slice[],height:number,primaryCount:number,offsetsPerPrimary:number){
  const railCount=primaryCount*(offsetsPerPrimary+1);
  const rings=slices.map((slice,index)=>{
    const loops=sectionContourLoops(slice.field),anchors=architecturalAnchors(loops,railCount);
    return anchors.length===railCount?anchors.map(anchor=>[(anchor.point[0]/(W-1)-.5)*EXPORT_TILE_INCHES,(.5-anchor.point[1]/(H-1))*EXPORT_TILE_INCHES,exportDepth(slices,index,height)] as Vec):null;
  }).filter((ring):ring is Vec[]=>Boolean(ring));
  if(rings.length<2)return 0;
  const mesh=new rhino.Mesh(),vertices=mesh.vertices(),faces=mesh.faces();
  const ids=rings.map(ring=>ring.map(point=>vertices.addPoint3d(...point)));
  for(let section=0;section<ids.length-1;section++)for(let rail=0;rail<railCount;rail++){
    const next=(rail+1)%railCount;faces.addQuadFace(ids[section][rail],ids[section][next],ids[section+1][next],ids[section+1][rail]);
  }
  const cap=(ring:Vec[],ringIds:number[],reverse:boolean)=>{
    const center=ring.reduce((sum,point)=>[sum[0]+point[0],sum[1]+point[1],sum[2]+point[2]] as Vec,[0,0,0] as Vec).map(value=>value/ring.length) as Vec;
    const centerId=vertices.addPoint3d(...center);
    for(let rail=0;rail<railCount;rail++){const next=(rail+1)%railCount;reverse?faces.addTriangleFace(centerId,ringIds[next],ringIds[rail]):faces.addTriangleFace(centerId,ringIds[rail],ringIds[next]);}
  };
  cap(rings[0],ids[0],true);cap(rings.at(-1)!,ids.at(-1)!,false);
  mesh.normals().computeNormals();
  mesh.setUserString("GeometryType","Watertight SubD-derived printable mesh");
  mesh.setUserString("Watertight","true");mesh.setUserString("SourceSections",String(rings.length));mesh.setUserString("TileEnvelope","20ft x 20ft maximum");mesh.setUserString("OverallHeight",`${height/12}ft`);
  mesh.setUserString("RailLogic",`${primaryCount} primary lines with ${offsetsPerPrimary} offsets per primary (${railCount} ordered rails total)`);
  mesh.setUserString("PrintNote","Closed shared-vertex shell; verify scale and wall thickness before fabrication.");
  const layerIndex=file.layers().addLayer("01_PRINTABLE_SUBD_FORM",{r:124,g:205,b:194,a:255});
  const attributes=new rhino.ObjectAttributes();attributes.name="PRINTABLE_SUBD_FORM_WATERTIGHT";attributes.layerIndex=layerIndex;
  attributes.setUserString("GeometryRole","Closed 3D-printable form mesh");file.objects().addMesh(mesh,attributes);
  const faceCount=faces.count;attributes.delete();mesh.delete();return faceCount;
}

function addSectionSurfaces(file:any,rhino:any,slices:Slice[],height:number){
  const layerIndex=file.layers().addLayer("02_SECTION_SURFACES",{r:242,g:105,b:82,a:255}),nx=48,ny=36,threshold=.32;
  slices.forEach((slice,index)=>{
    const mesh=new rhino.Mesh(),vertices=mesh.vertices(),faces=mesh.faces();
    for(let y=0;y<ny-1;y++)for(let x=0;x<nx-1;x++){
      const sx=Math.round((x+.5)*(W-1)/(nx-1)),sy=Math.round((y+.5)*(H-1)/(ny-1));if(slice.field[sy*W+sx]<threshold)continue;
      const depth=exportDepth(slices,index,height),x0=(x/(nx-1)-.5)*EXPORT_TILE_INCHES,x1=((x+1)/(nx-1)-.5)*EXPORT_TILE_INCHES,y0=(.5-y/(ny-1))*EXPORT_TILE_INCHES,y1=(.5-(y+1)/(ny-1))*EXPORT_TILE_INCHES;
      const a=vertices.addPoint3d(x0,y0,depth),b=vertices.addPoint3d(x1,y0,depth),c=vertices.addPoint3d(x1,y1,depth),d=vertices.addPoint3d(x0,y1,depth);faces.addQuadFace(a,b,c,d);
    }
    mesh.normals().computeNormals();mesh.setUserString("GeometryType","Independent CT section surface");mesh.setUserString("SectionIndex",String(index+1));
    const attributes=new rhino.ObjectAttributes();attributes.name=`SECTION_SURFACE_${String(index+1).padStart(3,"0")}`;attributes.layerIndex=layerIndex;file.objects().addMesh(mesh,attributes);attributes.delete();mesh.delete();
  });
  return slices.length;
}

function addBoundaryCurves(file:any,rhino:any,slices:Slice[],fidelity:number,height:number){
  const layerIndex=file.layers().addLayer("03_BOUNDARY_LINES",{r:139,g:217,b:207,a:255});let curveCount=0;
  slices.forEach((slice,sectionIndex)=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity)).forEach((loop,boundaryIndex)=>{
    const depth=exportDepth(slices,sectionIndex,height),points=loop.map(([x,y])=>[(x/(W-1)-.5)*EXPORT_TILE_INCHES,(.5-y/(H-1))*EXPORT_TILE_INCHES,depth]),curve=new rhino.PolylineCurve(points),attributes=new rhino.ObjectAttributes();
    attributes.name=`SECTION_${String(sectionIndex+1).padStart(3,"0")}_BOUNDARY_${String(boundaryIndex+1).padStart(2,"0")}`;attributes.layerIndex=layerIndex;
    attributes.setUserString("BoundaryRole","Exterior perimeter or enclosed void");attributes.setUserString("CurveFidelity",String(fidelity));file.objects().addCurve(curve,attributes);curveCount++;attributes.delete();curve.delete();
  }));
  return curveCount;
}

async function exportLoftBundle(model:LoftModel,thickness:number,version:7|8){
  if(model.rows.length<2)return;
  const [{default:rhino3dm},{default:wasmUrl}]=await Promise.all([import("rhino3dm"),import("rhino3dm/rhino3dm.wasm?url")]);
  const rhino=await rhino3dm({locateFile:()=>wasmUrl}),file=new rhino.File3dm();
  file.settings().modelUnitSystem=rhino.UnitSystem.Inches;file.settings().pageUnitSystem=rhino.UnitSystem.Feet;
  file.applicationName="Sand Scan";
  const scale=240/2.1,base=Math.min(...model.rows.flat().map(p=>p[1]))-thickness/scale;
  const point=(p:number[])=>[p[0]*scale,p[2]*scale,(p[1]-base)*scale];
  const addMesh=(depth:number,name:string,color:{r:number;g:number;b:number;a:number})=>{
    const data=loftDisplayMesh(model.rows,depth/scale),mesh=new rhino.Mesh(),ids=new Map<string,number>();
    for(let i=0;i<data.positions.length;i+=9){const face:number[]=[];for(let j=0;j<9;j+=3){const p=point(Array.from(data.positions.slice(i+j,i+j+3))),key=p.map(v=>v.toFixed(5)).join(",");let id=ids.get(key);if(id===undefined){id=mesh.vertices().addPoint3d(p[0],p[1],p[2]);ids.set(key,id)}face.push(id)}if(new Set(face).size===3)mesh.faces().addTriFace(face[0],face[1],face[2]);}
    mesh.normals().computeNormals();const attr=new rhino.ObjectAttributes();attr.name=name;attr.layerIndex=file.layers().addLayer(name,color);file.objects().addMesh(mesh,attr);attr.delete();mesh.delete();
  };
  addMesh(thickness,"01_CIRCULATION_LOFT_FORM",{r:22,g:128,b:255,a:255});
  addMesh(0,"02_LOFT_SURFACE_MESH",{r:212,g:37,b:210,a:255});
  const layer=file.layers().addLayer("03_LOFT_CURVES",{r:255,g:38,b:140,a:255});
  const curves=[...model.rows,...model.rails,model.rows.map(row=>row[0]),model.rows.map(row=>row.at(-1)!)];
  curves.forEach((row,i)=>{const curve=new rhino.PolylineCurve(row.map(p=>point(p))),attr=new rhino.ObjectAttributes();attr.name=`LOFT_CURVE_${i+1}`;attr.layerIndex=layer;file.objects().addCurve(curve,attr);attr.delete();curve.delete()});
  file.startSectionComments="Rendered circulation loft geometry. Model units inches; layout feet. Full unclipped membrane, surface mesh, and editable curves. Mesh geometry, not native Rhino SubD. Positive membrane thickness closes the outer rim.";
  downloadRhinoFile(file,rhino,version,`sand-scan-loft-rhino${version}.3dm`);file.delete();
}

async function exportRhinoBundle(slices:Slice[],version:7|8,fidelity:number,height:number,primaryCount:number,offsetsPerPrimary:number){
  if(!slices.length)return;
  const [{default:rhino3dm},{default:wasmUrl}]=await Promise.all([import("rhino3dm"),import("rhino3dm/rhino3dm.wasm?url")]);
  const rhino=await rhino3dm({locateFile:()=>wasmUrl}),file=new rhino.File3dm();
  const settings=file.settings();settings.modelUnitSystem=rhino.UnitSystem.Inches;settings.pageUnitSystem=rhino.UnitSystem.Feet;settings.modelAbsoluteTolerance=.001;
  file.applicationName="Sand Scan";file.applicationDetails=`20ft x 20ft maximum imperial unified-space package · Rhino ${version}`;
  const faces=addPrintableForm(file,rhino,slices,height,primaryCount,offsetsPerPrimary),surfaces=addSectionSurfaces(file,rhino,slices,height),curves=addBoundaryCurves(file,rhino,slices,fidelity,height);
  file.startSectionComments=`Imperial one-file Sand Scan export for Rhino ${version}. Model units: inches; layout units: feet. All geometry is constrained to a maximum 20ft x 20ft footprint and ${height/12}ft overall height. The closed printable mesh is one ruled loft driven by ${primaryCount} primary lines with ${offsetsPerPrimary} offsets per primary through every section (${faces} faces), with ${surfaces} reference section surfaces and ${curves} editable boundary curves.`;
  downloadRhinoFile(file,rhino,version,`sand-scan-complete-rhino${version}.3dm`);file.delete();
}

function CirculationPreview({slices,fidelity,primaryCount,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,smoothing,settings,setSettings,heightInches,setHeightInches,position,onDragStart}:{slices:Slice[];fidelity:number;primaryCount:number;offsetsPerPrimary:number;bifurcationsPerPrimary:number;bifurcationDistance:number;bifurcationScale:number;smoothing:number;settings:CirculationSettings;setSettings:React.Dispatch<React.SetStateAction<CirculationSettings>>;heightInches:number;setHeightInches:(value:number)=>void;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null),[camera,setCamera]=useWorkspaceState("CirculationPreview.camera",{yaw:-.72,pitch:.38,zoom:1});
  const normalized=useMemo(()=>{if(!slices.length)return[] as Slice[];const center=(slices[0].z+slices.at(-1)!.z)/2,span=Math.max(.001,Math.abs(slices[0].z-slices.at(-1)!.z));return slices.map(slice=>({...slice,z:(slice.z-center)/span*2.1}))},[slices]);
  const loops=useMemo(()=>normalized.map(slice=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity))),[normalized,fidelity]);
  const families=useMemo(()=>longitudinalFamilies(normalized,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,[],{}),[normalized,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale]);
  const circulation=useMemo(()=>buildCirculationGeometry(families,settings,heightInches,bifurcationsPerPrimary,false),[families,settings,heightInches,bifurcationsPerPrimary]);
  useEffect(()=>{if(families.primary.length)setSettings(current=>({...current,primaryCount:Math.max(1,Math.min(current.primaryCount,families.primary.length))}))},[families.primary.length,setSettings]);
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;const width=c.width,height=c.height;ctx.fillStyle="#071014";ctx.fillRect(0,0,width,height);
    if(!circulation.ribbons.length){ctx.fillStyle="#8fa9ae";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select depth sections to generate circulation",width/2,height/2);return}
    const rotate=([x,y,z]:Vec):Vec=>{const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]},project=(point:Vec)=>{const [x,y,z]=rotate(point),perspective=5.2/(5.2+z),scale=180*camera.zoom*perspective;return{x:width*.5+x*scale,y:height*.54-y*scale,z}};
    const bottom=[[-CHUNK_HALF,circulation.floorY,-CHUNK_HALF],[CHUNK_HALF,circulation.floorY,-CHUNK_HALF],[CHUNK_HALF,circulation.floorY,CHUNK_HALF],[-CHUNK_HALF,circulation.floorY,CHUNK_HALF]] as Vec[],top=bottom.map(([x,,z])=>[x,circulation.topY,z] as Vec);ctx.strokeStyle="rgba(119,205,199,.28)";ctx.lineWidth=.8;for(const ring of [bottom,top]){ctx.beginPath();ring.forEach((p,i)=>{const s=project(p);i?ctx.lineTo(s.x,s.y):ctx.moveTo(s.x,s.y)});ctx.closePath();ctx.stroke()}for(let i=0;i<4;i++){const a=project(bottom[i]),b=project(top[i]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
    const cells=circulation.ribbons.flatMap(ribbon=>Array.from({length:Math.max(0,Math.min(ribbon.left.length,ribbon.right.length)-1)},(_,index)=>{const screen=[ribbon.left[index],ribbon.right[index],ribbon.right[index+1],ribbon.left[index+1]].map(project);return{screen,depth:screen.reduce((sum,p)=>sum+p.z,0)/4,kind:ribbon.kind}})).sort((a,b)=>b.depth-a.depth);
    cells.forEach(cell=>{ctx.fillStyle=cell.kind==="spine"?viewportColor("primary",.34):viewportColor("secondary",.3);ctx.strokeStyle=cell.kind==="spine"?viewportColor("primary",.72):viewportColor("secondary",.7);ctx.lineWidth=.65;ctx.beginPath();ctx.moveTo(cell.screen[0].x,cell.screen[0].y);for(let i=1;i<4;i++)ctx.lineTo(cell.screen[i].x,cell.screen[i].y);ctx.closePath();ctx.fill();ctx.stroke()});
    circulation.ribbons.forEach(ribbon=>{const points=ribbon.center.map(project);ctx.strokeStyle=ribbon.kind==="spine"?viewportColor("primary",.96):viewportColor("secondary",.9);ctx.lineWidth=ribbon.kind==="spine"?2:1.4;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke()});ctx.fillStyle="#789a9d";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(`20′ × 20′ ENVELOPE · ${heightInches/12}′ HEIGHT · ONE OCCUPIABLE FLOOR`,14,20);
  },[circulation,camera,heightInches]);
  return <article data-screenshot-component="circulation" className="spatial-rational-node circulation-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Route size={19}/></div><div><h2>Circulation Skeleton</h2><p>Primary cage lines → occupiable routes</p></div><span>DRAG · ORBIT</span></header>
    <div className="rational-viewport"><canvas ref={canvas} width={700} height={390} aria-label="Rotatable 3D preview of circulation routes constrained to a twenty foot square spatial chunk" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.55,Math.min(2.2,value.zoom-event.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span><button onClick={()=>setCamera({yaw:-.72,pitch:.38,zoom:1})}>Reset view</button></div>
    <div className="spatial-constraint-band"><strong>20′ × 20′</strong><span>Single occupiable floor</span><span>Outside geometry trimmed</span><label>Height <b>{heightInches/12}′</b><input type="range" min="96" max="240" step="12" value={heightInches} onChange={event=>setHeightInches(Number(event.target.value))}/></label></div>
    <div className="rational-controls">
      <label><span>Primary circulations</span><strong>{Math.min(settings.primaryCount,Math.max(1,families.primary.length))}</strong><input type="range" min="1" max={Math.max(1,families.primary.length)} step="1" value={Math.min(settings.primaryCount,Math.max(1,families.primary.length))} onChange={event=>setSettings(current=>({...current,primaryCount:Number(event.target.value)}))}/><small>Evenly distributed cage lines, preserved in XYZ</small></label>
      <label><span>Path width</span><strong>{settings.pathWidth}′</strong><input type="range" min="4" max="8" step=".5" value={settings.pathWidth} onChange={event=>setSettings(current=>({...current,pathWidth:Number(event.target.value)}))}/><small>Clear occupiable route</small></label>
      <label><span>Junction widening</span><strong>{settings.junctionWidth}%</strong><input type="range" min="0" max="100" step="5" value={settings.junctionWidth} onChange={event=>setSettings(current=>({...current,junctionWidth:Number(event.target.value)}))}/><small>Threshold expansion</small></label>
      <label><span>Route smoothing</span><strong>{settings.smoothing}%</strong><input type="range" min="0" max="100" step="5" value={settings.smoothing} onChange={event=>setSettings(current=>({...current,smoothing:Number(event.target.value)}))}/><small>Remove directional noise</small></label>
    </div>
    <div className="rational-summary"><span><strong>{Math.min(settings.primaryCount,Math.max(1,families.primary.length))}</strong>primary routes</span><span><strong>{circulation.ribbons.length}</strong>3D paths</span><span><strong>{circulation.trimmedCount}</strong>points trimmed</span><span><strong>0</strong>bifurcations</span></div>
    <p className="rational-note">Only the selected primary lines from the Unified Section Cage become circulation. Cage bifurcations remain available in the combined framework, but are excluded from this circulation skeleton.</p>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function PrimaryCirculationLoftPreview({slices,fidelity,primaryCount,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,smoothing,circulationSettings,heightInches,position,onDragStart,onMembrane}:{slices:Slice[];fidelity:number;primaryCount:number;offsetsPerPrimary:number;bifurcationsPerPrimary:number;bifurcationDistance:number;bifurcationScale:number;smoothing:number;circulationSettings:CirculationSettings;heightInches:number;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void;onMembrane:React.Dispatch<React.SetStateAction<LoftModel>>}){
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null),[camera,setCamera]=useWorkspaceState("PrimaryCirculationLoftPreview.camera",{yaw:-.72,pitch:.36,zoom:1}),[loftOpacity,setLoftOpacity]=useWorkspaceState("PrimaryCirculationLoftPreview.loftOpacity",74),[sectionDensity,setSectionDensity]=useWorkspaceState("PrimaryCirculationLoftPreview.sectionDensity",55);
  const [extension,setExtension]=useWorkspaceState("PrimaryCirculationLoftPreview.extension",55),[fairing,setFairing]=useWorkspaceState("PrimaryCirculationLoftPreview.fairing",70);
  const normalized=useMemo(()=>{if(!slices.length)return[] as Slice[];const center=(slices[0].z+slices.at(-1)!.z)/2,span=Math.max(.001,Math.abs(slices[0].z-slices.at(-1)!.z));return slices.map(slice=>({...slice,z:(slice.z-center)/span*2.1}))},[slices]);
  const loops=useMemo(()=>normalized.map(slice=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity))),[normalized,fidelity]);
  const families=useMemo(()=>longitudinalFamilies(normalized,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,[],{}),[normalized,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale]);
  const circulation=useMemo(()=>buildCirculationGeometry(families,circulationSettings,heightInches,bifurcationsPerPrimary,false),[families,circulationSettings,heightInches,bifurcationsPerPrimary]);
  const primaryCirculations=useMemo(()=>circulation.ribbons.filter(ribbon=>ribbon.kind==="spine"),[circulation]);
  const membrane=useMemo(()=>waterfall(primaryCirculations,extension,fairing),[primaryCirculations,extension,fairing]);
  useEffect(()=>onMembrane(membrane),[membrane,onMembrane]);
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;const width=c.width,height=c.height;ctx.fillStyle="#020202";ctx.fillRect(0,0,width,height);
    if(primaryCirculations.length<2){ctx.fillStyle="#8fa9ae";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select at least two primary circulations to loft between them",width/2,height/2);return}
    const rotate=([x,y,z]:Vec):Vec=>{const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]},project=(point:Vec)=>{const [x,y,z]=rotate(point),perspective=5.3/(5.3+z),scale=184*camera.zoom*perspective;return{x:width*.5+x*scale,y:height*.54-y*scale,z}};
    const draw=(line:Vec[],color:string,lineWidth:number)=>{ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.beginPath();line.map(project).forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke()};
    const faces=membrane.rows.slice(1).flatMap((row,i)=>row.slice(1).map((_,j)=>{const screen=[membrane.rows[i][j],membrane.rows[i][j+1],row[j+1],row[j]].map(project);return{screen,depth:screen.reduce((sum,p)=>sum+p.z,0)/4}})).sort((a,b)=>b.depth-a.depth);
    faces.forEach(face=>{ctx.fillStyle=viewportColor("primary",loftOpacity/100*.75);ctx.beginPath();face.screen.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.fill()});
    const sectionStep=Math.max(2,Math.round(24-sectionDensity/100*22));
    membrane.rows.forEach((row,i)=>{if(i%sectionStep===0||i===membrane.rows.length-1)draw(row,viewportColor("secondary",.85),1)});
    membrane.rails.forEach(rail=>draw(rail,VIEWPORT_PRIMARY,2));
    if(membrane.rows.length){draw(membrane.rows.map(row=>row[0]),VIEWPORT_TERTIARY,1.2);draw(membrane.rows.map(row=>row.at(-1)!),VIEWPORT_TERTIARY,1.2)}
    ctx.fillStyle="#789a9d";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText("CONTINUOUS FLOOR MEMBRANE · WATERFALL TRANSITIONS",14,20);
  },[primaryCirculations,membrane,camera,loftOpacity,sectionDensity]);
  const faceCount=Math.max(0,membrane.rows.length-1)*Math.max(0,(membrane.rows[0]?.length??0)-1);
  return <article data-screenshot-component="loft" className="spatial-rational-node primary-loft-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Sparkles size={19}/></div><div><h2>Primary Circulation Loft</h2><p>Flat circulation bands → flowing waterfall membrane</p></div><span>DRAG · ORBIT</span></header>
    <div className="rational-viewport"><canvas ref={canvas} width={700} height={390} aria-label="Rotatable 3D continuous floor membrane with curved waterfall transitions" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.55,Math.min(2.2,value.zoom-event.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span><button onClick={()=>setCamera({yaw:-.72,pitch:.36,zoom:1})}>Reset view</button></div>
    <div className="spatial-constraint-band compact"><strong>20′ × 20′</strong><span>{heightInches/12}′ height</span><span>Primary routes only</span><span>No bifurcations</span></div>
    <div className="rational-controls"><label><span>Horizontal extension</span><strong>{extension}%</strong><input type="range" min="0" max="100" value={extension} onChange={e=>setExtension(Number(e.target.value))}/><small>Protect flat zones before each bend</small></label><label><span>Longitudinal smoothing</span><strong>{fairing}%</strong><input type="range" min="0" max="100" value={fairing} onChange={e=>setFairing(Number(e.target.value))}/><small>Fair matching stations without moving route ends</small></label><label><span>Loft opacity</span><strong>{loftOpacity}%</strong><input type="range" min="20" max="100" step="5" value={loftOpacity} onChange={event=>setLoftOpacity(Number(event.target.value))}/><small>Transparent</small><small>Solid</small></label><label><span>Section density</span><strong>{sectionDensity}%</strong><input type="range" min="0" max="100" step="5" value={sectionDensity} onChange={event=>setSectionDensity(Number(event.target.value))}/><small>Few cross-sections</small><small>Dense cross-sections</small></label></div>
    <div className="rational-summary"><span><strong>{primaryCirculations.length}</strong>primary circulations</span><span><strong>{Math.max(0,primaryCirculations.length-1)}</strong>waterfall transitions</span><span><strong>{faceCount}</strong>surface cells</span><span><strong>{circulationSettings.smoothing}%</strong>route smoothing</span></div>
    <p className="rational-note">Circulation offsets retain the flat floor widths. Horizontal extensions blend tangentially into curved waterfall transitions, creating one continuous membrane. Blue: primary rails and membrane; magenta: section profiles; pink: outer edges. The connected Rhino export uses this loft geometry.</p>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function ArchitecturalAdjustmentNode({model,minimumFeet,setMinimumFeet,heightFeet,scaleFactor,position,onDragStart}:{model:LoftModel;minimumFeet:number;setMinimumFeet:(feet:number)=>void;heightFeet:number;scaleFactor:number;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),renderer=useRef<ReturnType<typeof createLoftRenderer>|null>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
  const [camera,setCamera]=useWorkspaceState("ArchitecturalAdjustmentNode.camera",{yaw:-.62,pitch:-.35,zoom:1,panX:0,panY:0});
  useEffect(()=>{const target=canvas.current;if(!target)return;try{renderer.current=createLoftRenderer(target)}catch{renderer.current=createLoftRenderer(target,true)}return()=>{renderer.current?.dispose();renderer.current=null}},[]);
  useEffect(()=>{renderer.current?.draw(model,camera,0,false,false)},[model,camera]);
  return <article className="spatial-rational-node architectural-clearance-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Ruler size={19}/></div><div><h2>Architectural Height</h2><p>Floor membrane → minimum clear height</p></div><span>DRAG · ORBIT</span></header>
    <div className="architectural-preview"><canvas ref={canvas} width={520} height={300} aria-label="Rotatable 3D view of vertically rationalized floor membrane" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.55,Math.min(2.2,value.zoom-event.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span></div>
    <div className="architectural-controls"><label><span>Minimum spatial height</span><strong>{minimumFeet}′</strong><input type="range" min="6" max="12" step=".5" value={minimumFeet} onChange={event=>setMinimumFeet(Number(event.target.value))}/><small>The floor datum stays fixed. If the loft is lower, its vertical distances scale proportionally.</small></label></div>
    <div className="architectural-summary"><span><small>Current height</small><strong>{heightFeet.toFixed(1)}′</strong></span><span><small>Vertical scale</small><strong>{scaleFactor.toFixed(2)}×</strong></span><b>{heightFeet+0.01>=minimumFeet?"Minimum met":"Height expanded to minimum"}</b></div>
    <p className="rational-note">Only vertical distances scale from the lowest floor point; horizontal dimensions stay inside the 20′ × 20′ envelope. Slopes and height changes keep their proportions. This adjusted membrane feeds the render and exports.</p>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function CirculationLoftRender({model,thickness,setThickness,position,onDragStart,onHeight,onCapture}:{onCapture:(capture:()=>string)=>void;onHeight:(height:number)=>void;model:LoftModel;thickness:number;setThickness:(value:number)=>void;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const renderNode=useRef<HTMLElement>(null);
  useEffect(()=>{const node=renderNode.current;if(!node)return;const observer=new ResizeObserver(()=>onHeight(node.offsetHeight));observer.observe(node);onHeight(node.offsetHeight);return()=>observer.disconnect()},[onHeight]);
  const canvas=useRef<HTMLCanvasElement>(null),renderer=useRef<ReturnType<typeof createLoftRenderer>|null>(null);
  const initial:LoftCamera={yaw:-.62,pitch:-.35,zoom:1.15,panX:0,panY:0};
  const [camera,setCamera]=useWorkspaceState<LoftCamera>("CirculationLoftRender.camera",initial),[view,setView]=useWorkspaceState("CirculationLoftRender.view","perspective"),[blue,setBlue]=useWorkspaceState("CirculationLoftRender.blue",false),[showRails,setShowRails]=useWorkspaceState("CirculationLoftRender.showRails",false),[renderError,setRenderError]=useState(false),[software,setSoftware]=useState(false),[clipEnabled,setClipEnabled]=useWorkspaceState("CirculationLoftRender.clipEnabled",false),[clip,setClip]=useWorkspaceState("CirculationLoftRender.clip",0);
  const [spinning,setSpinning]=useState(false);
  const gesture=useRef<{x:number;y:number;camera:LoftCamera;pan:boolean}|null>(null);
  useEffect(()=>{try{if(canvas.current)renderer.current=createLoftRenderer(canvas.current,software)}catch{if(!software)setSoftware(true);else setRenderError(true)}return()=>{renderer.current?.dispose();renderer.current=null}},[software]);
  useEffect(()=>{renderer.current?.draw(model,camera,thickness,blue,showRails,clipEnabled?clip:null)},[model,camera,thickness,blue,showRails,software,clipEnabled,clip]);
  useEffect(()=>{
    if(!spinning||model.rows.length<2||renderError)return;
    let last=performance.now();
    const timer=window.setInterval(()=>{const now=performance.now(),elapsed=Math.min((now-last)/1000,.25);last=now;if(document.hidden)return;setCamera(current=>({...current,yaw:(current.yaw+elapsed*Math.PI/10)%(Math.PI*2)}))},software?100:33);
    return()=>window.clearInterval(timer);
  },[spinning,model.rows.length,renderError,software]);
  useEffect(()=>{onCapture(()=>{if(model.rows.length<2)throw new Error("Generate a circulation loft before exporting.");const image=document.createElement("canvas");image.width=1600;image.height=1088;const output=createLoftRenderer(image,true);try{output.draw(model,camera,thickness,false,false,clipEnabled?clip:null);return image.toDataURL("image/png")}finally{output.dispose()}})},[model,camera,thickness,clipEnabled,clip,onCapture]);
  const [recording,setRecording]=useState(false),[recordedFrames,setRecordedFrames]=useState(0),[gifError,setGifError]=useState("");
  const recordingJob=useRef<{timer:number;gif:ModelGif;output:ReturnType<typeof createLoftRenderer>}|null>(null);
  useEffect(()=>()=>{const job=recordingJob.current;if(job){window.clearInterval(job.timer);job.output.dispose()}},[]);
  const stopRecording=()=>{const job=recordingJob.current;if(!job)return;window.clearInterval(job.timer);recordingJob.current=null;job.output.dispose();setRecording(false);if(job.gif.frames){void saveExport(job.gif.blob(),"sand-scan-loft-rotation.gif")}};
  const recordGif=()=>{setSpinning(false);setGifError("");const canvas=document.createElement("canvas");canvas.width=480;canvas.height=326;const output=createLoftRenderer(canvas,true),gif=new ModelGif(480,326),start={...camera};let frame=0;setRecording(true);setRecordedFrames(0);
   const tick=()=>{try{const current={...start,yaw:start.yaw+frame*Math.PI*2/160};output.draw(model,current,thickness,blue,showRails,clipEnabled?clip:null);gif.add(canvas.getContext("2d")!.getImageData(0,0,480,326).data,12);setCamera(current);frame++;setRecordedFrames(frame);if(frame>=160)stopRecording()}catch{setGifError("Could not record this view. Try again.");stopRecording()}};
   const timer=window.setInterval(tick,125);recordingJob.current={timer,gif,output};tick();
  };
  const exportClayPng=()=>{setSpinning(false);const outputCanvas=document.createElement("canvas");outputCanvas.width=1600;outputCanvas.height=1088;const output=createLoftRenderer(outputCanvas,true);try{output.draw(model,camera,thickness,false,false,clipEnabled?clip:null);outputCanvas.toBlob(blob=>{if(blob)void saveExport(blob,`sand-scan-reference-clay-${view.toLowerCase()}.png`)},"image/png")}finally{output.dispose()}};
  const chooseView=(name:string)=>{setSpinning(false);const iso:Record<string,number>={NW:-Math.PI/4,NE:Math.PI/4,SW:-3*Math.PI/4,SE:3*Math.PI/4};setView(name);setCamera({...initial,orthographic:name!=="perspective",...(name in iso?{yaw:iso[name],pitch:-Math.atan(1/Math.sqrt(2))}:name==="front"?{yaw:0,pitch:0}:name==="top"?{yaw:0,pitch:-Math.PI/2}:name==="right"?{yaw:Math.PI/2,pitch:0}:{})})};
  return <article ref={renderNode} className="spatial-rational-node loft-render-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Sparkles size={19}/></div><div><h2>Circulation Loft Render</h2><p>Primary circulation loft → smooth spatial membrane</p></div><span>DRAG · ORBIT</span></header>
    <div className="loft-render-toolbar" role="group" aria-label="Rendered loft view presets">{["perspective","front","top","right"].map(name=><button key={name} aria-pressed={view===name} onClick={()=>chooseView(name)}>{name}</button>)}</div>
    <div className="loft-render-toolbar" role="group" aria-label="30/60 isometric views"><span>30/60 isometric</span>{["NW","NE","SW","SE"].map(name=><button key={name} aria-pressed={view===name} onClick={()=>chooseView(name)}>{name}</button>)}</div>
    <div className="loft-render-toolbar" role="group" aria-label="360 degree model rotation"><button aria-pressed={spinning} disabled={model.rows.length<2||renderError} onClick={()=>{if(!spinning){setView("perspective");setCamera(current=>({...current,orthographic:false,pitch:Math.abs(current.pitch)>1.3?-.35:current.pitch}))}setSpinning(value=>!value)}}>{spinning?"Pause 360°":"Play 360°"}</button><span>Continuous turntable · 20 seconds per revolution</span></div>
    <div className="rational-viewport"><canvas key={software?"software":"gpu"} ref={canvas} width={1000} height={680} aria-label="Smooth shaded three dimensional rendering of the primary circulation loft" onContextMenu={event=>event.preventDefault()} onPointerDown={event=>{setSpinning(false);event.currentTarget.setPointerCapture(event.pointerId);gesture.current={x:event.clientX,y:event.clientY,camera:{...camera},pan:event.shiftKey||event.button!==0}}} onPointerMove={event=>{const g=gesture.current;if(!g)return;const dx=event.clientX-g.x,dy=event.clientY-g.y;if(g.pan)setCamera({...g.camera,panX:g.camera.panX+dx*.007/g.camera.zoom,panY:g.camera.panY-dy*.007/g.camera.zoom});else{setView("perspective");setCamera({...g.camera,orthographic:false,yaw:g.camera.yaw+dx*.008,pitch:Math.max(-1.57,Math.min(1.57,g.camera.pitch+dy*.008))})}}} onPointerUp={()=>{gesture.current=null}} onPointerCancel={()=>{gesture.current=null}} onWheel={event=>{event.preventDefault();setCamera(c=>({...c,zoom:Math.max(.4,Math.min(3,c.zoom-event.deltaY*.0015))}))}}/>
      <span>Drag to orbit · Shift-drag to pan · wheel to zoom</span><button onClick={()=>chooseView("perspective")}>Reset view</button>
      {(model.rows.length===0||renderError)&&<div className="loft-render-empty">{renderError?"Rendered view unavailable in this browser.":"Select images and at least two primary circulations in the loft component."}</div>}
    </div>
    <div className="loft-render-toolbar" role="group" aria-label="Animated GIF export"><button disabled={recording||model.rows.length<2||renderError} onClick={recordGif}>Record GIF</button><button disabled={!recording} onClick={stopRecording}>Stop &amp; export GIF</button><span aria-live="polite">{recording?`Recording · ${recordedFrames} frames`:"480 × 326 · one rotation · stops automatically after 160 frames"}</span>{gifError&&<p role="alert">{gifError}</p>}</div>
    <div className="loft-render-toolbar"><button disabled={model.rows.length<2||renderError} onClick={exportClayPng}><Download size={14} style={{verticalAlign:"middle",marginRight:6}}/>Export Clay PNG</button><span>Current viewport · 1600 × 1088</span></div>
    <div className="loft-render-toolbar" role="group" aria-label="Loft render material"><button aria-pressed={!blue} onClick={()=>setBlue(false)}>Reference clay</button><button aria-pressed={blue} onClick={()=>setBlue(true)}>Primary blue</button><button aria-pressed={showRails} onClick={()=>setShowRails(value=>!value)}>Primary rails {showRails?"on":"off"}</button></div>
    <div className="rational-controls"><label><span>Membrane thickness</span><strong>{thickness}″</strong><input type="range" min="0" max="12" step=".5" value={thickness} onChange={event=>setThickness(Number(event.target.value))}/><small>Connected normal extrusion · shared by render and Rhino export</small></label><label><span>View scale</span><strong>{camera.zoom.toFixed(2)}×</strong><input type="range" min=".4" max="3" step=".05" value={camera.zoom} onChange={event=>setCamera(c=>({...c,zoom:Number(event.target.value)}))}/><small>Frame the spatial chunk</small></label></div>
    <div className="loft-render-toolbar"><button aria-pressed={clipEnabled} onClick={()=>setClipEnabled(value=>!value)}>Clipping plane {clipEnabled?"on":"off"}</button></div>
    <div className="rational-controls"><label style={{gridColumn:"1/-1"}}><span>Front-to-back clipping plane</span><strong>{clip}%</strong><input aria-label="Front-to-back clipping plane" type="range" min="0" max="100" step="1" value={clip} disabled={!clipEnabled} onChange={event=>setClip(Number(event.target.value))}/><small>Front → back · pink lines show the cut · orbit to inspect inside</small></label></div>
    <p className="rational-note">The connected loft’s flat floor bands and curved transitions are rendered with overhead lighting on a black background. Clipping removes geometry from the front toward the back. Change the loft sliders to update this view. Membrane thickness is shared with Rhino Export. Clipping is view-only; the full loft is exported.</p>
    <span className="space-output-port" aria-hidden="true"/>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function ReadOnlyBoundaryLinesPreview({slices,activeIndex,fidelity,setFidelity,primaryCount,setPrimaryCount,offsetsPerPrimary,setOffsetsPerPrimary,bifurcationsPerPrimary,setBifurcationsPerPrimary,bifurcationDistance,setBifurcationDistance,bifurcationScale,setBifurcationScale,smoothing,setSmoothing,position,onDragStart}:{slices:Slice[];activeIndex:number;fidelity:number;setFidelity:(value:number)=>void;primaryCount:number;setPrimaryCount:(value:number)=>void;offsetsPerPrimary:number;setOffsetsPerPrimary:(value:number)=>void;bifurcationsPerPrimary:number;setBifurcationsPerPrimary:(value:number)=>void;bifurcationDistance:number;setBifurcationDistance:(value:number)=>void;bifurcationScale:number;setBifurcationScale:(value:number)=>void;smoothing:number;setSmoothing:(value:number)=>void;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
  const [camera,setCamera]=useWorkspaceState("ReadOnlyBoundaryLinesPreview.camera",{yaw:-.68,pitch:.34,zoom:1}),[ruledSurfaces,setRuledSurfaces]=useWorkspaceState("ReadOnlyBoundaryLinesPreview.ruledSurfaces",true);
  const loops=useMemo(()=>slices.map(slice=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity))),[slices,fidelity]);
  const families=useMemo(()=>longitudinalFamilies(slices,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,[],{}),[slices,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale]);
  const pointCount=families.anchors.length;
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;const width=c.width,height=c.height;
    const background=ctx.createLinearGradient(0,0,0,height);background.addColorStop(0,"#111a1d");background.addColorStop(1,"#080d0f");ctx.fillStyle=background;ctx.fillRect(0,0,width,height);
    if(!slices.length){ctx.fillStyle="#829da0";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select depth-map sections to trace",width/2,height/2);return}
    const zCenter=(slices[0].z+slices.at(-1)!.z)/2,rotate=([x,y,z]:Vec):Vec=>{z-=zCenter;const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]},project=(point:Vec)=>{const [x,y,z]=rotate(point),perspective=5/(5+z),scale=150*camera.zoom*perspective;return{x:width*.5+x*scale,y:height*.51-y*scale,z}};
    ctx.strokeStyle="rgba(67,98,103,.26)";ctx.lineWidth=.7;for(let i=-7;i<=7;i++){const a=project([i*.38,-1.03,zCenter-2.8]),b=project([i*.38,-1.03,zCenter+2.8]),d=project([-2.8,-1.03,zCenter+i*.38]),e=project([2.8,-1.03,zCenter+i*.38]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.beginPath();ctx.moveTo(d.x,d.y);ctx.lineTo(e.x,e.y);ctx.stroke()}
    const rendered=loops.flatMap((section,index)=>section.map(loop=>({index,points:loop.map(([x,y])=>[(x/(W-1)-.5)*2.25,(.5-y/(H-1))*1.7,slices[index].z] as Vec)}))).map(item=>({...item,depth:item.points.reduce((sum,point)=>sum+rotate(point)[2],0)/Math.max(1,item.points.length)})).sort((a,b)=>b.depth-a.depth);
    rendered.forEach(item=>{const active=item.index===activeIndex;ctx.strokeStyle=active?"rgba(155,235,224,.68)":"rgba(82,136,137,.22)";ctx.lineWidth=active?1.35:.62;ctx.beginPath();item.points.forEach((point,index)=>{const p=project(point);index?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y)});ctx.stroke()});
    if(ruledSurfaces)[...families.ruled,...families.branchPlates].forEach(strip=>{const count=Math.min(strip.a.length,strip.b.length),branch=families.branchPlates.includes(strip);for(let index=0;index<count-1;index++){const a=project(strip.a[index]),b=project(strip.b[index]),c=project(strip.b[index+1]),d=project(strip.a[index+1]);ctx.fillStyle=branch?viewportColor("tertiary",.3):viewportColor("primary",.12);ctx.strokeStyle=branch?viewportColor("tertiary",.7):viewportColor("primary",.18);ctx.lineWidth=branch ? .55 : .45;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.lineTo(c.x,c.y);ctx.lineTo(d.x,d.y);ctx.closePath();ctx.fill();ctx.stroke()}});
    const draw=(lines:Vec[][],color:string,width:number,glow=0)=>lines.forEach(line=>{const points=line.map(project);if(points.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.shadowColor=color;ctx.shadowBlur=glow;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);for(let i=1;i<points.length-1;i++){const midpoint={x:(points[i].x+points[i+1].x)/2,y:(points[i].y+points[i+1].y)/2};ctx.quadraticCurveTo(points[i].x,points[i].y,midpoint.x,midpoint.y)}ctx.lineTo(points.at(-1)!.x,points.at(-1)!.y);ctx.stroke();ctx.shadowBlur=0});
    draw(families.offset,viewportColor("secondary",.54),.85);draw(families.primary,viewportColor("primary",.92),1.55,3);draw(families.branches,viewportColor("tertiary",.85),1.15,2);
    families.anchors.forEach(({point})=>{const p=project(point);ctx.fillStyle="#d7fff8";ctx.beginPath();ctx.arc(p.x,p.y,1.35,0,Math.PI*2);ctx.fill()});
    ctx.fillStyle="#739396";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(`TRANSVERSE CAGE · SECTION ${activeIndex+1} / ${slices.length}`,14,20);ctx.textAlign="right";ctx.fillText(`${pointCount} EXTRACTED POINTS`,width-14,20);
  },[slices,loops,families,activeIndex,pointCount,camera,ruledSurfaces]);
  return <article data-screenshot-component="cage" className="boundary-lines-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><ScanLine size={19}/></div><div><h2>Unified Section Cage</h2><p>Read-only longitudinal line system</p></div><span>DRAG · ORBIT</span></header>
    <div className="boundary-viewport"><canvas ref={canvas} width={620} height={350} aria-label="Rotatable read-only 3D preview of transverse section cages and longitudinal line families" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.55,Math.min(2.2,value.zoom-event.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span><button onClick={()=>setCamera({yaw:-.68,pitch:.34,zoom:1})}>Reset view</button></div>
    <div className="geometry-mode"><span>Display</span><div><button className={!ruledSurfaces?"active":""} onClick={()=>setRuledSurfaces(false)}>Line hierarchy</button><button className={ruledSurfaces?"active":""} onClick={()=>setRuledSurfaces(true)}>Unified ruled skin</button></div><small>{families.primary.length} primary · {families.offset.length} offset · {families.branches.length} bifurcating lines</small></div>
    <div className="boundary-fidelity line-method-controls">
      <label><span>Primary lines</span><strong>{primaryCount}</strong><input type="range" min="3" max="12" step="1" value={primaryCount} onChange={event=>setPrimaryCount(Number(event.target.value))}/><small>Broad structure</small><small>Dense hierarchy</small></label>
      <label><span>Offsets per primary line</span><strong>{offsetsPerPrimary}</strong><input type="range" min="0" max="6" step="1" value={offsetsPerPrimary} onChange={event=>setOffsetsPerPrimary(Number(event.target.value))}/><small>Primary only</small><small>6 offsets each</small></label>
      <label><span>Bifurcations per primary line</span><strong>{bifurcationsPerPrimary}</strong><input type="range" min="0" max="6" step="1" value={bifurcationsPerPrimary} onChange={event=>setBifurcationsPerPrimary(Number(event.target.value))}/><small>No branches</small><small>6 nested branches</small></label>
      <label><span>Bifurcation distance</span><strong>{bifurcationDistance}%</strong><input type="range" min="10" max="100" step="5" value={bifurcationDistance} onChange={event=>setBifurcationDistance(Number(event.target.value))}/><small>Short threshold</small><small>Long traverse</small></label>
      <label><span>Bifurcation scaling</span><strong>{bifurcationScale}%</strong><input type="range" min="0" max="100" step="5" value={bifurcationScale} onChange={event=>setBifurcationScale(Number(event.target.value))}/><small>Even spacing</small><small>Expanding gaps</small></label>
      <label><span>Longitudinal smoothing</span><strong>{smoothing}%</strong><input type="range" min="0" max="100" step="2" value={smoothing} onChange={event=>setSmoothing(Number(event.target.value))}/><small>Exact section points</small><small>Fluid transition</small></label>
      <label><span>Cage fidelity</span><strong>{fidelity}%</strong><input type="range" min="10" max="100" step="2" value={fidelity} onChange={event=>setFidelity(Number(event.target.value))}/><small>Simplified + smooth</small><small>Boundary accurate</small></label>
    </div>
    <div className="architectural-anchor-key"><span><i className="extent"/>Dominant perimeter</span><span><i className="ridge"/>Primary lines</span><span><i className="void"/>Offset lines</span><span><i className="ground"/>Scaled bifurcations</span></div>
    <p>Each bifurcation belongs to its primary line, leaves and rejoins it at the shared ends, and scales progressively farther outward than the branch before it. The website's existing line colors remain unchanged.</p>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function BoundaryLinesPreview({slices,activeIndex,fidelity,setFidelity,primaryCount,setPrimaryCount,offsetsPerPrimary,setOffsetsPerPrimary,bifurcationsPerPrimary,setBifurcationsPerPrimary,bifurcationDistance,setBifurcationDistance,branchDirections,pointOffsets,setPointOffsets,smoothing,setSmoothing,position,onDragStart}:{slices:Slice[];activeIndex:number;fidelity:number;setFidelity:(value:number)=>void;primaryCount:number;setPrimaryCount:(value:number)=>void;offsetsPerPrimary:number;setOffsetsPerPrimary:(value:number)=>void;bifurcationsPerPrimary:number;setBifurcationsPerPrimary:(value:number)=>void;bifurcationDistance:number;setBifurcationDistance:(value:number)=>void;branchDirections:number[];pointOffsets:PointOffsets;setPointOffsets:React.Dispatch<React.SetStateAction<PointOffsets>>;smoothing:number;setSmoothing:(value:number)=>void;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null);
  const orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
  const pan=useRef<{x:number;y:number;panX:number;panY:number}|null>(null);
  const pointDrag=useRef<{axis:0|1|2;startX:number;startY:number;base:Vec;before:PointOffsets;screen:{x:number;y:number};worldPerPixel:number;scaleX:number;scaleY:number;key:string;live:Vec}|null>(null);
  const gumballRef=useRef<{origin:{x:number;y:number};axes:{axis:0|1|2;end:{x:number;y:number}}[]} | null>(null);
  const branchHitRef=useRef<{index:number;points:{x:number;y:number}[]}[]>([]);
  const [camera,setCamera]=useWorkspaceState("BoundaryLinesPreview.camera",{yaw:-.68,pitch:.34,zoom:1,panX:0,panY:0});
  const [ruledSurfaces,setRuledSurfaces]=useWorkspaceState("BoundaryLinesPreview.ruledSurfaces",true);
  const [selectedBranch,setSelectedBranch]=useState(0);
  const [selectedPoint,setSelectedPoint]=useState(0);
  const [expanded,setExpanded]=useState(false);
  const [activeTool,setActiveTool]=useState<"select"|"pan"|"orbit">("select");
  const [history,setHistory]=useState<{states:PointOffsets[];index:number}>({states:[{}],index:0});
  const loops=useMemo(()=>slices.map(slice=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity))),[slices,fidelity]);
  const families=useMemo(()=>longitudinalFamilies(slices,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,50,branchDirections,pointOffsets),[slices,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,branchDirections,pointOffsets]);
  const pointCount=families.anchors.length;
  useEffect(()=>{if(selectedBranch>=families.branches.length)setSelectedBranch(Math.max(0,families.branches.length-1));const length=families.branches[Math.min(selectedBranch,Math.max(0,families.branches.length-1))]?.length??0;if(selectedPoint>=length)setSelectedPoint(Math.max(0,length-1))},[families.branches,selectedBranch,selectedPoint]);
  const cloneOffsets=(value:PointOffsets)=>Object.fromEntries(Object.entries(value).map(([key,offset])=>[key,[...offset] as Vec])) as PointOffsets;
  const recordOffsets=(next:PointOffsets)=>setHistory(current=>{const base=current.states.slice(0,current.index+1),states=[...base,cloneOffsets(next)].slice(-11);return{states,index:states.length-1}});
  const setPointAxis=(axis:0|1|2,value:number)=>{const key=`${selectedBranch}:${selectedPoint}`,offset=[...(pointOffsets[key]??[0,0,0])] as Vec;offset[axis]=value;const next={...pointOffsets,[key]:offset};setPointOffsets(next);recordOffsets(next)};
  const undo=()=>setHistory(current=>{if(current.index<=0)return current;const index=current.index-1;setPointOffsets(cloneOffsets(current.states[index]));return{...current,index}});
  const redo=()=>setHistory(current=>{if(current.index>=current.states.length-1)return current;const index=current.index+1;setPointOffsets(cloneOffsets(current.states[index]));return{...current,index}});
  useEffect(()=>{if(!expanded)return;setHistory({states:[cloneOffsets(pointOffsets)],index:0});const close=(event:KeyboardEvent)=>{if(event.key==="Escape")setExpanded(false);if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="z"){event.preventDefault();event.shiftKey?redo():undo()}if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="y"){event.preventDefault();redo()}};const previous=document.body.style.overflow;document.body.style.overflow="hidden";window.addEventListener("keydown",close);return()=>{document.body.style.overflow=previous;window.removeEventListener("keydown",close)}},[expanded]);
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;const width=c.width,height=c.height;
    const background=ctx.createLinearGradient(0,0,0,height);background.addColorStop(0,"#111a1d");background.addColorStop(1,"#080d0f");ctx.fillStyle=background;ctx.fillRect(0,0,width,height);
    if(!slices.length){ctx.fillStyle="#829da0";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select depth-map sections to trace",width/2,height/2);return}
    const zCenter=(slices[0].z+slices.at(-1)!.z)/2,rotate=([x,y,z]:Vec):Vec=>{z-=zCenter;const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]};
    const project=(point:Vec)=>{const [x,y,z]=rotate(point),depth=Math.max(1.2,6.4-z),focal=(expanded?Math.min(width,height)*2.25:920),scale=focal/depth*camera.zoom;return{x:width*.5+camera.panX+x*scale,y:height*.51+camera.panY-y*scale,z}};
    ctx.strokeStyle="rgba(67,98,103,.26)";ctx.lineWidth=.7;for(let i=-7;i<=7;i++){const a=project([i*.38,-1.03,zCenter-2.8]),b=project([i*.38,-1.03,zCenter+2.8]),d=project([-2.8,-1.03,zCenter+i*.38]),e=project([2.8,-1.03,zCenter+i*.38]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.beginPath();ctx.moveTo(d.x,d.y);ctx.lineTo(e.x,e.y);ctx.stroke()}
    const rendered=loops.flatMap((section,index)=>section.map(loop=>({index,points:loop.map(([x,y])=>[(x/(W-1)-.5)*2.25,(.5-y/(H-1))*1.7,slices[index].z] as Vec)}))).map(item=>({...item,depth:item.points.reduce((sum,point)=>sum+rotate(point)[2],0)/Math.max(1,item.points.length)})).sort((a,b)=>b.depth-a.depth);
    // Transverse boundaries remain visible as a restrained reference cage.
    rendered.forEach(item=>{const active=item.index===activeIndex;ctx.strokeStyle=active?"rgba(155,235,224,.68)":"rgba(82,136,137,.22)";ctx.lineWidth=active?1.35:.62;ctx.beginPath();item.points.forEach((point,index)=>{const p=project(point);if(index===0)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y)});ctx.stroke()});
    if(ruledSurfaces)families.ruled.forEach(strip=>{const count=Math.min(strip.a.length,strip.b.length);for(let index=0;index<count-1;index++){const a=project(strip.a[index]),b=project(strip.b[index]),c=project(strip.b[index+1]),d=project(strip.a[index+1]);ctx.fillStyle=strip.kind==="void"?viewportColor("secondary",.16):viewportColor("primary",.12);ctx.strokeStyle=strip.kind==="void"?viewportColor("secondary",.24):viewportColor("primary",.18);ctx.lineWidth=.45;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.lineTo(c.x,c.y);ctx.lineTo(d.x,d.y);ctx.closePath();ctx.fill();ctx.stroke()}});
    if(ruledSurfaces)families.branchPlates.forEach(strip=>{const count=Math.min(strip.a.length,strip.b.length);for(let index=0;index<count-1;index++){const a=project(strip.a[index]),b=project(strip.b[index]),c=project(strip.b[index+1]),d=project(strip.a[index+1]);ctx.fillStyle=viewportColor("tertiary",.3);ctx.strokeStyle=viewportColor("tertiary",.7);ctx.lineWidth=.55;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.lineTo(c.x,c.y);ctx.lineTo(d.x,d.y);ctx.closePath();ctx.fill();ctx.stroke()}});
    const drawFamily=(lines:Vec[][],color:string,widthValue:number,glow=0)=>lines.forEach(line=>{
      const points=line.map(project);if(points.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=widthValue;ctx.shadowColor=color;ctx.shadowBlur=glow;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);
      for(let i=1;i<points.length-1;i++){const midpoint={x:(points[i].x+points[i+1].x)/2,y:(points[i].y+points[i+1].y)/2};ctx.quadraticCurveTo(points[i].x,points[i].y,midpoint.x,midpoint.y)}
      ctx.lineTo(points.at(-1)!.x,points.at(-1)!.y);ctx.stroke();ctx.shadowBlur=0;
    });
    drawFamily(families.tweens,viewportColor("secondary",.25),.62);drawFamily(families.offset,viewportColor("secondary",.54),.85);
    families.primary.forEach(line=>drawFamily([line],viewportColor("primary",.95),1.55,3));
    branchHitRef.current=families.branches.map((line,index)=>({index,points:line.map(project)}));
    families.branches.forEach((line,index)=>drawFamily([line],index===selectedBranch?viewportColor("tertiary",1):viewportColor("tertiary",.85),index===selectedBranch?2.1:1.15,index===selectedBranch?5:2));
    if(expanded)branchHitRef.current.forEach(branch=>branch.points.forEach((point,index)=>{const selected=branch.index===selectedBranch&&index===selectedPoint;ctx.fillStyle=selected?"#ffffff":"rgba(255,205,139,.88)";ctx.strokeStyle=selected?"#6cf4e8":"#513a20";ctx.lineWidth=selected?3:1.2;ctx.beginPath();ctx.arc(point.x,point.y,selected?7:3.5,0,Math.PI*2);ctx.fill();ctx.stroke()}));
    const selectedWorld=families.branches[selectedBranch]?.[selectedPoint],selectedScreen=selectedWorld?project(selectedWorld):null;gumballRef.current=null;
    if(expanded&&selectedWorld&&selectedScreen){const worldAxes:[0|1|2,Vec,string,string][]= [[0,[selectedWorld[0]+.42,selectedWorld[1],selectedWorld[2]],"#e96060","X"],[1,[selectedWorld[0],selectedWorld[1]+.42,selectedWorld[2]],"#65c878","Y"],[2,[selectedWorld[0],selectedWorld[1],selectedWorld[2]+.42],"#5f8ee8","Z"]],axes:{axis:0|1|2;end:{x:number;y:number}}[]=[];worldAxes.forEach(([axis,worldEnd,color,label])=>{const end=project(worldEnd),dx=end.x-selectedScreen.x,dy=end.y-selectedScreen.y,length=Math.max(1,Math.hypot(dx,dy)),ux=dx/length,uy=dy/length,px=-uy,py=ux;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(selectedScreen.x,selectedScreen.y);ctx.lineTo(end.x,end.y);ctx.stroke();ctx.beginPath();ctx.moveTo(end.x,end.y);ctx.lineTo(end.x-ux*12+px*6,end.y-uy*12+py*6);ctx.lineTo(end.x-ux*12-px*6,end.y-uy*12-py*6);ctx.closePath();ctx.fill();ctx.font="bold 12px sans-serif";ctx.fillText(label,end.x+px*8,end.y+py*8);axes.push({axis,end})});ctx.fillStyle="#f5fbff";ctx.strokeStyle="#17323a";ctx.lineWidth=2;ctx.beginPath();ctx.arc(selectedScreen.x,selectedScreen.y,8,0,Math.PI*2);ctx.fill();ctx.stroke();gumballRef.current={origin:selectedScreen,axes}}
    families.anchors.forEach(({point,role})=>{const p=project(point);ctx.fillStyle=role.startsWith("ground")?"#8bd4ff":role.startsWith("void")||role.startsWith("threshold")?"#ffc16f":role==="ridge"?"#ff89d8":"#d7fff8";ctx.beginPath();ctx.arc(p.x,p.y,1.35,0,Math.PI*2);ctx.fill()});
    const axis=([color,end,label]:[string,Vec,string])=>{const o=project([0,-1.03,zCenter]),p=project(end);ctx.strokeStyle=color;ctx.lineWidth=1.3;ctx.beginPath();ctx.moveTo(o.x,o.y);ctx.lineTo(p.x,p.y);ctx.stroke();ctx.fillStyle=color;ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(label,p.x+4,p.y)};
    axis(["#dc6666",[.5,-1.03,zCenter],"X"]);axis(["#68bf78",[0,-.53,zCenter],"Z"]);axis(["#6b91d9",[0,-1.03,zCenter+.5],"Y"]);
    ctx.fillStyle="#739396";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(`TRANSVERSE CAGE · SECTION ${activeIndex+1} / ${slices.length}`,14,20);ctx.textAlign="right";ctx.fillText(`${pointCount} EXTRACTED POINTS`,width-14,20);
  },[slices,loops,families,activeIndex,pointCount,camera,ruledSurfaces,selectedBranch,selectedPoint,expanded]);
  const selectedKey=`${selectedBranch}:${selectedPoint}`,selectedOffset=pointOffsets[selectedKey]??[0,0,0] as Vec;
  const segmentDistance=(p:{x:number;y:number},a:{x:number;y:number},b:{x:number;y:number})=>{const dx=b.x-a.x,dy=b.y-a.y,length=dx*dx+dy*dy,t=length?Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/length)):0;return Math.hypot(p.x-(a.x+dx*t),p.y-(a.y+dy*t))};
  const pointerDown=(event:React.PointerEvent<HTMLCanvasElement>)=>{
    const rightOrbit=expanded&&event.button===2&&!event.shiftKey,rightPan=expanded&&event.button===2&&event.shiftKey,middlePan=expanded&&event.button===1;
    if(rightPan||middlePan||(expanded&&activeTool==="pan"&&event.button===0)){event.currentTarget.setPointerCapture(event.pointerId);pan.current={x:event.clientX,y:event.clientY,panX:camera.panX,panY:camera.panY};return}
    if(rightOrbit||(expanded&&activeTool==="orbit"&&event.button===0)){event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch};return}
    const rect=event.currentTarget.getBoundingClientRect(),scaleX=event.currentTarget.width/rect.width,scaleY=event.currentTarget.height/rect.height,pointer={x:(event.clientX-rect.left)*scaleX,y:(event.clientY-rect.top)*scaleY};
    if(expanded&&activeTool==="select"&&gumballRef.current){
      const axis=gumballRef.current.axes.find(item=>segmentDistance(pointer,gumballRef.current!.origin,item.end)<15);
      if(axis){const dx=axis.end.x-gumballRef.current.origin.x,dy=axis.end.y-gumballRef.current.origin.y,length=Math.max(1,Math.hypot(dx,dy)),base=[...selectedOffset] as Vec;event.currentTarget.setPointerCapture(event.pointerId);pointDrag.current={axis:axis.axis,startX:event.clientX,startY:event.clientY,base,before:cloneOffsets(pointOffsets),screen:{x:dx/length,y:dy/length},worldPerPixel:.42/length,scaleX,scaleY,key:selectedKey,live:base};return}
    }
    let best={branch:-1,point:-1,distance:expanded?14:10};branchHitRef.current.forEach(branch=>branch.points.forEach((point,index)=>{const distance=Math.hypot(point.x-pointer.x,point.y-pointer.y);if(distance<best.distance)best={branch:branch.index,point:index,distance}}));
    if(best.branch>=0){setSelectedBranch(best.branch);setSelectedPoint(best.point);return}
    event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch};
  };
  const pointerMove=(event:React.PointerEvent<HTMLCanvasElement>)=>{
    const drag=pointDrag.current;if(drag){const dx=(event.clientX-drag.startX)*drag.scaleX,dy=(event.clientY-drag.startY)*drag.scaleY,amount=(dx*drag.screen.x+dy*drag.screen.y)*drag.worldPerPixel,next=[...drag.base] as Vec;next[drag.axis]=Math.max(-1.5,Math.min(1.5,drag.base[drag.axis]+amount));drag.live=next;setPointOffsets({...drag.before,[drag.key]:next});return}
    const movingPan=pan.current;if(movingPan){setCamera(value=>({...value,panX:movingPan.panX+(event.clientX-movingPan.x),panY:movingPan.panY+(event.clientY-movingPan.y)}));return}
    const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))}))};
  const pointerEnd=()=>{const drag=pointDrag.current;if(drag){const next={...drag.before,[drag.key]:drag.live};setPointOffsets(next);recordOffsets(next)}orbit.current=null;pan.current=null;pointDrag.current=null};
  return <article data-screenshot-component="cage" className={`boundary-lines-node ${expanded?"expanded":""}`} style={expanded?undefined:{left:position.x,top:position.y}} onDoubleClick={event=>{if(!(event.target as HTMLElement).closest("button,input"))setExpanded(true)}}>
    <header onPointerDown={expanded?undefined:onDragStart}><div className="model-icon"><ScanLine size={19}/></div><div><h2>Unified Section Cage</h2><p>{expanded?"Perspective control-point editor":"Primary longitudinal lines · double-click to edit full screen"}</p></div>{expanded?<button className="close-cage-editor" onClick={()=>setExpanded(false)} aria-label="Close full-screen point editor"><XCircle size={18}/>Close editor</button>:<span>DRAG · DOUBLE-CLICK TO EDIT</span>}</header>
    <div className="cage-workspace">
      <aside className="cage-toolbar" aria-label="Modeling tools">
        <button className={activeTool==="select"?"active":""} onClick={()=>setActiveTool("select")} title="Select and edit bifurcations"><MousePointer2/><span>Select</span></button>
        <button className={activeTool==="pan"?"active":""} onClick={()=>setActiveTool("pan")} title="Pan the viewport"><Hand/><span>Pan</span></button>
        <button className={activeTool==="orbit"?"active":""} onClick={()=>setActiveTool("orbit")} title="Orbit the viewport"><RotateCcw/><span>Orbit</span></button>
        <div className="tool-divider"/>
        <button onClick={undo} disabled={history.index<=0} title="Undo (Ctrl+Z)"><Undo2/><span>Undo</span></button>
        <button onClick={redo} disabled={history.index>=history.states.length-1} title="Redo (Ctrl+Y)"><Redo2/><span>Redo</span></button>
      </aside>
      <div className={`boundary-viewport tool-${activeTool}`} onContextMenu={event=>event.preventDefault()}><canvas ref={canvas} width={expanded?1280:620} height={expanded?760:350} aria-label="Rhino-style perspective editor with selectable curve control points and three-axis translation gumball" onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerEnd} onPointerCancel={pointerEnd} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.35,Math.min(3,value.zoom-event.deltaY*.0012))}))}}/><span>{expanded?"Select a control point · drag the X, Y, or Z gumball axis · Right-drag: orbit · Shift + right-drag: pan":"Double-click to edit individual control points"}</span><div className="viewport-badge">Perspective</div><button onClick={()=>setCamera({yaw:-.68,pitch:.34,zoom:1,panX:0,panY:0})}>Zoom extents</button></div>
      <aside className="cage-inspector">
        {expanded&&<div className="inspector-heading"><strong>Properties</strong><span>History {history.index} / {Math.min(10,history.states.length-1)}</span></div>}
        {expanded&&<div className="gumball-status"><div className="gumball-mini"><i className="gx"/><i className="gy"/><i className="gz"/></div><div><strong>Gumball · Point {families.branches.length?selectedPoint+1:"—"}</strong><span>Branch {families.branches.length?selectedBranch+1:"—"} · translate the selected point along X, Y, or Z.</span></div></div>}
        <div className="geometry-mode"><span>Geometry generation</span><div><button className={!ruledSurfaces?"active":""} onClick={()=>setRuledSurfaces(false)}>Line hierarchy</button><button className={ruledSurfaces?"active":""} onClick={()=>setRuledSurfaces(true)}>Unified ruled skin</button></div><small>{families.primary.length} primary · {families.offset.length} offset · {families.branches.length} bifurcating lines</small></div>
        <div className="boundary-fidelity line-method-controls">
          <label><span>Primary lines</span><strong>{primaryCount}</strong><input type="range" min="3" max="12" step="1" value={primaryCount} onChange={event=>setPrimaryCount(Number(event.target.value))}/><small>Broad structure</small><small>Dense hierarchy</small></label>
          <label><span>Offsets per primary line</span><strong>{offsetsPerPrimary}</strong><input type="range" min="0" max="6" step="1" value={offsetsPerPrimary} onChange={event=>setOffsetsPerPrimary(Number(event.target.value))}/><small>Primary only</small><small>6 offsets each</small></label>
          <label><span>Bifurcations per primary line</span><strong>{bifurcationsPerPrimary}</strong><input type="range" min="0" max="4" step="1" value={bifurcationsPerPrimary} onChange={event=>setBifurcationsPerPrimary(Number(event.target.value))}/><small>No branching</small><small>4 controlled forks</small></label>
          <label><span>Bifurcation distance</span><strong>{bifurcationDistance}%</strong><input type="range" min="10" max="100" step="5" value={bifurcationDistance} onChange={event=>setBifurcationDistance(Number(event.target.value))}/><small>Short threshold</small><small>Long traverse</small></label>
          <label><span>Selected control point</span><strong>P{selectedPoint+1} · B{selectedBranch+1}</strong><input aria-label="Select control point on active branch" type="range" min="0" max={Math.max(0,(families.branches[selectedBranch]?.length??1)-1)} step="1" value={Math.min(selectedPoint,Math.max(0,(families.branches[selectedBranch]?.length??1)-1))} disabled={!families.branches.length} onChange={event=>setSelectedPoint(Number(event.target.value))}/><small>Click a point or use slider</small><small>{families.branches[selectedBranch]?.length??0} points</small></label>
          <label><span>X translation</span><strong>{selectedOffset[0].toFixed(2)}</strong><input type="range" min="-1.5" max="1.5" step=".05" value={selectedOffset[0]} disabled={!families.branches.length} onChange={event=>setPointAxis(0,Number(event.target.value))}/><small>−X</small><small>+X</small></label>
          <label><span>Y translation</span><strong>{selectedOffset[1].toFixed(2)}</strong><input type="range" min="-1.5" max="1.5" step=".05" value={selectedOffset[1]} disabled={!families.branches.length} onChange={event=>setPointAxis(1,Number(event.target.value))}/><small>−Y</small><small>+Y</small></label>
          <label><span>Z translation</span><strong>{selectedOffset[2].toFixed(2)}</strong><input type="range" min="-1.5" max="1.5" step=".05" value={selectedOffset[2]} disabled={!families.branches.length} onChange={event=>setPointAxis(2,Number(event.target.value))}/><small>−Z</small><small>+Z</small></label>
          <label><span>Longitudinal smoothing</span><strong>{smoothing}%</strong><input type="range" min="0" max="100" step="2" value={smoothing} onChange={event=>setSmoothing(Number(event.target.value))}/><small>Exact section points</small><small>Fluid transition</small></label>
          <label><span>Cage fidelity</span><strong>{fidelity}%</strong><input type="range" min="10" max="100" step="2" value={fidelity} onChange={event=>setFidelity(Number(event.target.value))}/><small>Simplified + smooth</small><small>Boundary accurate</small></label>
        </div>
        <div className="architectural-anchor-key" aria-label="Primary and offset line logic"><span><i className="extent"/>Dominant perimeter</span><span><i className="ridge"/>Primary lines</span><span><i className="void"/>Offset lines</span><span><i className="ground"/>Shared section order</span></div>
        <p>{expanded?"Every orange control point is selectable. The red, green, and blue gumball arrows translate the active point along world X, Y, and Z; the adjacent curve and downstream ridge geometry rebuild around that edited point. Ctrl+Z and Ctrl+Y move through the last ten point translations.":"Double-click this component to open the full-screen cage editor and translate individual curve points in a standard perspective viewport."}</p>
      </aside>
    </div>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function GridRationalizationPreview({slices,gridSize,alignment,setGridSize,setAlignment,position,onDragStart}:{slices:Slice[];gridSize:number;alignment:number;setGridSize:(value:number)=>void;setAlignment:(value:number)=>void;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
  const [camera,setCamera]=useWorkspaceState("GridRationalizationPreview.camera",{yaw:-.72,pitch:.38,zoom:1});
  const loops=useMemo(()=>slices.map(slice=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,72))),[slices]);
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;const width=c.width,height=c.height;
    const bg=ctx.createLinearGradient(0,0,0,height);bg.addColorStop(0,"#172127");bg.addColorStop(1,"#080d11");ctx.fillStyle=bg;ctx.fillRect(0,0,width,height);
    if(!slices.length){ctx.fillStyle="#829da0";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select sections to align them to a grid",width/2,height/2);return}
    const zCenter=(slices[0].z+slices.at(-1)!.z)/2,rotate=([x,y,z]:Vec):Vec=>{z-=zCenter;const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]},project=(point:Vec)=>{const [x,y,z]=rotate(point),perspective=5/(5+z),scale=148*camera.zoom*perspective;return{x:width*.5+x*scale,y:height*.51-y*scale,z}};
    const unit=Math.max(.16,gridSize/(W-1)*2.25);ctx.strokeStyle="rgba(102,181,193,.24)";ctx.lineWidth=.7;
    for(let x=-1.25;x<=1.26;x+=unit){const a=project([x,-1.02,zCenter-2.7]),b=project([x,-1.02,zCenter+2.7]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
    for(let z=zCenter-2.7;z<=zCenter+2.71;z+=unit){const a=project([-1.35,-1.02,z]),b=project([1.35,-1.02,z]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
    const rendered=loops.flatMap((section,index)=>section.map(loop=>({index,points:loop.map(([x,y])=>[(x/(W-1)-.5)*2.25,(.5-y/(H-1))*1.7,slices[index].z] as Vec)}))).map(item=>({...item,depth:item.points.reduce((sum,p)=>sum+rotate(p)[2],0)/Math.max(1,item.points.length)})).sort((a,b)=>b.depth-a.depth);
    rendered.forEach(item=>{ctx.strokeStyle=item.index%4===0?VIEWPORT_PRIMARY:viewportColor("primary",.55);ctx.lineWidth=item.index%4===0?1.5:.85;ctx.beginPath();item.points.forEach((point,index)=>{const p=project(point);index?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y)});ctx.stroke()});
    ctx.fillStyle="#83aeb2";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(`GRID ${gridSize} CELLS · ALIGNMENT ${alignment}%`,14,20);
  },[slices,loops,gridSize,alignment,camera]);
  return <article data-screenshot-component="grid" className="grid-rationalization-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Grid3X3 size={19}/></div><div><h2>Grid Rationalization</h2><p>CT volume → ordered section framework</p></div><span>DRAG · GRID</span></header>
    <div className="grid-viewport"><canvas ref={canvas} width={620} height={350} aria-label="Rotatable 3D preview of grid-aligned section curves" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.55,Math.min(2.2,value.zoom-event.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span><button onClick={()=>setCamera({yaw:-.72,pitch:.38,zoom:1})}>Reset view</button></div>
    <div className="grid-controls"><label><span>Grid spacing</span><strong>{gridSize} cells</strong><input type="range" min="4" max="16" step="1" value={gridSize} onChange={event=>setGridSize(Number(event.target.value))}/><small>Fine order</small><small>Broad modules</small></label><label><span>Grid alignment</span><strong>{alignment}%</strong><input type="range" min="0" max="100" step="2" value={alignment} onChange={event=>setAlignment(Number(event.target.value))}/><small>Original curves</small><small>Orthogonal order</small></label></div>
    <p>Exterior and void boundaries are pulled toward one shared grid, then relaxed so adjacent sections blend continuously. This rationalized geometry now feeds the Unified Section Cage before its primary and offset lines guide circulation.</p>
    <span className="port port-left port-ct-in" aria-hidden="true"/><span className="port port-right port-cage-out" aria-hidden="true"/>
  </article>;
}

function ExportNode({model,thickness,slices,lineFidelity,primaryCount,offsetsPerPrimary,exportHeight,setExportHeight,position,onDragStart}:{model:LoftModel;thickness:number;slices:Slice[];lineFidelity:number;primaryCount:number;offsetsPerPrimary:number;exportHeight:number;setExportHeight:(value:number)=>void;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const [version,setVersion]=useWorkspaceState<7|8>("ExportNode.version",8),[busy,setBusy]=useState(false);
  const run=async()=>{setBusy(true);try{await exportLoftBundle(model,thickness,version)}finally{setBusy(false)}};
  return <article className="export-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Download size={19}/></div><div><h2>Rhino Export</h2><p>One coordinated file · three editable geometry layers</p></div><span>DRAG · 3DM</span></header>
    <div className="rhino-version"><span>File version</span><div role="group" aria-label="Rhino file version"><button className={version===7?"active":""} onClick={()=>setVersion(7)}>Rhino 7</button><button className={version===8?"active":""} onClick={()=>setVersion(8)}>Rhino 8</button></div><small>Imperial model units: inches · layout units: feet.</small></div>
    <div className="export-scale"><div><span>Maximum footprint</span><strong>20′ × 20′</strong><small>240″ × 240″ architectural envelope</small></div><label><span>Overall height</span><strong>{exportHeight/12}′ <small>({exportHeight}″)</small></strong><input type="range" min="96" max="240" step="12" value={exportHeight} onChange={event=>setExportHeight(Number(event.target.value))}/><small>8′</small><small>20′</small></label></div>
    <div className="export-options export-bundle">
      <section><Box size={22}/><div><strong>01 · Unified Printable Form</strong><p>The displayed circulation loft, with {thickness}″ membrane thickness. A positive thickness closes its perimeter.</p></div></section>
      <section><ScanLine size={22}/><div><strong>02 · Section Surfaces</strong><p>The same loft membrane exported as an unthickened surface mesh on a separate layer.</p></div></section>
      <section><ScanLine size={22}/><div><strong>03 · Boundary Lines</strong><p>Editable loft section curves, perimeter curves, and primary circulation rails.</p></div></section>
    </div>
    <div className="export-secondary export-all"><span>{slices.length} sections · one floor · max 20′ × 20′ × {exportHeight/12}′ · outside geometry trimmed</span><button onClick={()=>void run()} disabled={model.rows.length<2||busy}><Download size={14}/>{busy?"Building complete package…":`Export all 3 · R${version}`}</button></div>
    <span className="port port-left port-mesh-in" aria-hidden="true"/><span className="port port-right port-preview-out" aria-hidden="true"/>
  </article>;
}

const SPACE_CRITERIA=["curvature","collision","deflation","centrality","rhythm","scale"] as const;
type SpaceCriterion=typeof SPACE_CRITERIA[number];
type SpaceKind="lobby"|"office"|"gathering";
const SPACE_TYPES=[{id:"lobby",title:"Lobby",number:"01"},{id:"office",title:"Workspaces/Office Spaces",number:"02"},{id:"gathering",title:"Gathering Spaces",number:"03"}] as const;
const SPACE_COLORS:Record<SpaceKind,string>={lobby:"#d924e8",office:"#1ec1f2",gathering:"#ff268c"};
const SPACE_PRESETS:Record<SpaceKind,{name:string;ratings:number[]}[]>={
 lobby:[{name:"Topographic / Ground-field",ratings:[7,2,6,2,2,3]},{name:"Compressed Sequential",ratings:[4,6,9,2,8,6]},{name:"Vertical Void",ratings:[1,7,5,9,3,5]},{name:"Continuous Hall",ratings:[1,2,8,2,1,7]},{name:"Linear Gallery",ratings:[3,6,5,2,2,3]}],
 office:[{name:"Cascaded / Terrace Plates",ratings:[4,4,2,1,6,5]},{name:"Open Hall",ratings:[1,2,6,2,1,7]},{name:"Flat Deep-plate",ratings:[1,2,2,1,1,2]},{name:"Void-edge",ratings:[1,7,5,9,3,5]},{name:"Folded / Undulating",ratings:[8,7,9,3,8,7]}],
 gathering:[{name:"Stepped Amphitheatre",ratings:[5,3,3,7,3,4]},{name:"Void-field",ratings:[9,6,5,8,2,2]},{name:"Inserted Horizontal Plate",ratings:[7,8,8,4,5,6]},{name:"Contained Room Within Volume",ratings:[9,9,5,2,3,5]},{name:"Linear Edge Gallery",ratings:[7,7,5,2,2,3]}]
};
const presetValues=(space:SpaceKind,index:number)=>Object.fromEntries(SPACE_CRITERIA.map((criterion,i)=>[criterion,SPACE_PRESETS[space][index].ratings[i]])) as Record<SpaceCriterion,number>;
function SpaceQualitiesNode({autoScores,setAutoScores,manualLabel,presetIndex,onPreset,activeSpace,onSpaceChange,title,number,images,active,onActivate,values,onValue,model,position,onDragStart}:{autoScores:boolean;setAutoScores:(value:boolean)=>void;manualLabel:string;presetIndex:number;onPreset:(index:number)=>void;activeSpace:SpaceKind;onSpaceChange:(space:SpaceKind)=>void;title:string;number:string;images:StoredImage[];active:boolean;onActivate:()=>void;values:Record<SpaceCriterion,number>;onValue:(criterion:SpaceCriterion,value:number,base?:Record<SpaceCriterion,number>)=>void;model:LoftModel|null;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const measured=useMemo(()=>model?analyzeLoft(model):null,[model]);
  const matches=useMemo(()=>measured?rankTypologies(measured,SPACE_PRESETS[activeSpace]):[],[measured,activeSpace]);
  const recommendations=useMemo(()=>measured?recommendSpace(measured,SPACE_PRESETS):[],[measured]);
  const recommended=recommendations[0],intent=psychologicalIntent[activeSpace],categoryResult=recommendations.find(result=>result.category===activeSpace);
  const displayed=autoScores&&measured?measured:values;
  return <article className={`spatial-rational-node space-qualities-node ${active?"space-active":""}`} style={{left:position.x,top:position.y,"--space-color":SPACE_COLORS[activeSpace]} as React.CSSProperties} onClick={onActivate} aria-label={`${title} spatial qualities`}>
    <span className="space-input-port" aria-hidden="true"/>
    <header onPointerDown={onDragStart}><div className="model-icon">{number}</div><div><h2>Spatial Qualities</h2><p>Circulation loft → {title}</p></div><span>DRAG</span></header>
    <div className="loft-render-toolbar space-tabs" role="tablist" aria-label="Space type">{SPACE_TYPES.map(space=><button key={space.id} id={`space-tab-${space.id}`} role="tab" aria-selected={activeSpace===space.id} aria-controls="space-qualities-panel" aria-pressed={activeSpace===space.id} style={{"--tab-color":SPACE_COLORS[space.id]} as React.CSSProperties} onClick={()=>onSpaceChange(space.id)} onKeyDown={event=>{if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;event.preventDefault();const index=SPACE_TYPES.findIndex(space=>space.id===activeSpace),next=event.key==="Home"?0:event.key==="End"?2:(index+(event.key==="ArrowRight"?1:2))%3;onSpaceChange(SPACE_TYPES[next].id);document.getElementById(`space-tab-${SPACE_TYPES[next].id}`)?.focus()}}>{space.title}</button>)}</div>
    <div id="space-qualities-panel" role="tabpanel" aria-labelledby={`space-tab-${activeSpace}`}>
    <button className="space-route-button" aria-pressed={active} onClick={onActivate}>{active?"Active · receiving circulation loft":"Select as active destination"}</button>
    <div className="space-source-images">{Array.from({length:3},(_,i)=>images[i]?<figure key={images[i].key}><img src={source(images[i].key)} alt={images[i].name}/><figcaption title={images[i].name}>{images[i].name}</figcaption></figure>:<div className="space-source-placeholder" key={i}>Source image {i+1}<small>Select images in CT Volume</small></div>)}</div>
    {images.length>3&&<p className="space-source-note">First three selected source images · {images.length} images in the loft sequence</p>}
    <section className="space-recommendation" aria-live="polite"><span>Recommended use · psychological design intent</span>{recommended?<><h3>{SPACE_TYPES.find(space=>space.id===recommended.category)!.title}</h3><p>{recommended.typology} · {recommended.fit}% heuristic fit · {recommended.fit<65||recommended.fit-recommendations[1].fit<5?"Tentative recommendation":"Leading recommendation"}</p><button onClick={()=>onSpaceChange(recommended.category)}>View recommended space</button><table><thead><tr><th>Space</th><th>Reference fit</th><th>Design intent</th><th>Combined</th></tr></thead><tbody>{recommendations.map(result=><tr key={result.category}><td>{SPACE_TYPES.find(space=>space.id===result.category)!.title}</td><td>{result.referenceFit}%</td><td>{result.intentFit}%</td><td>{result.fit}%</td></tr>)}</tbody></table></>:<p>Generate a loft to compare lobby, workspace, and gathering suitability.</p>}
    <h4>How this space is intended to feel</h4><ul>{intent.goals.map(goal=><li key={goal}>{goal}</li>)}</ul>
    {categoryResult&&<><h4>Geometry signals · {title}</h4>{categoryResult.checks.map(check=><p key={check.name}><strong>{check.name}: {Math.round(check.value*100)}%</strong><br/>{check.reason}</p>)}</>}
    <details><summary>Potential concerns and missing context</summary><ul>{intent.risks.map(risk=><li key={risk}>{risk}</li>)}</ul><p>Not established by this loft:</p><ul>{intent.context.map(item=><li key={item}>{item}</li>)}</ul></details>
    <p className="psy-method">Based on your psychology slides. Combined fit uses 60% reference descriptor similarity and 40% geometric design-intent proxies, with equal weight within each group. These are design suggestions, not predictions of how a person will feel. Daylight, privacy, and neighboring spaces are unmeasured and do not contribute to the score.</p><a href={`/space-references/${activeSpace}-psychology.png`} target="_blank" rel="noreferrer">View psychology reference</a></section>
    <section className="space-classification" aria-live="polite"><span>Geometry classification · {title}</span>{matches.length?<><h3>{autoScores?matches[0].name:manualLabel}</h3>{!autoScores&&<p>Manual classification · criteria loaded from the reference, with any slider adjustments.</p>}<p>{matches[0].similarity}% descriptor similarity · {matches[0].similarity<65||matches[1].distance-matches[0].distance<.5?"Tentative match":"Closest reference match"}</p><table><thead><tr><th>Descriptor</th><th>Loft</th><th>Reference</th></tr></thead><tbody>{SPACE_CRITERIA.map((key,i)=><tr key={key}><td>{key}</td><td>{measured![key]}</td><td>{matches[0].ratings[i]}</td></tr>)}</tbody></table><h4>Closest alternatives</h4>{matches.slice(1,3).map(match=><p key={match.name}>{match.name} · {match.similarity}% descriptor similarity</p>)}<details><summary>All matches and measurement logic</summary>{matches.slice(3).map(match=><p key={match.name}>{match.name} · {match.similarity}%</p>)}<p>Curvature: accumulated profile and rail bending. Collision: proximity between primary rails. Deflation: vertical compression within the envelope. Centrality: rail concentration around the plan center. Rhythm: repeated rails and consistent profile widths. Scale: occupied geometric extent. Measurements use the loft’s dimensions for proximity and vertical compression. Matching compares absolute ratings and the relative descriptor pattern, giving curvature, convergence, centrality, and rhythm more influence than generic size. No random labels are assigned. This is a diagrammatic estimate, not a test of occupancy or structural performance.</p></details></>:<p>Select source images and generate a circulation loft to assign a typology.</p>}</section>
    <div className="space-preset"><label>Reference space type<select value={presetIndex} onChange={event=>{setAutoScores(false);onPreset(Number(event.target.value))}}>{SPACE_PRESETS[activeSpace].map((preset,i)=><option key={preset.name} value={i}>{preset.name}</option>)}</select></label><button onClick={()=>{setAutoScores(false);onPreset(presetIndex)}}>Restore reference ratings</button><a href={`/space-references/${activeSpace}.png`} target="_blank" rel="noreferrer">View reference sheet</a><p>Choosing a type loads its six reference ratings. Adjust the sliders to customize this space.</p></div>
    <div className="loft-render-toolbar"><button aria-pressed={autoScores} onClick={()=>setAutoScores(true)}>Live extracted scores</button><span>{autoScores?"Scores follow loft geometry":"Reference ratings · used in Illustrator export"}</span></div>
    <div className="space-criteria">{SPACE_CRITERIA.map(criterion=><label key={criterion}><span>{criterion}</span><output>{displayed[criterion]} / 10</output><input aria-label={`${title} ${criterion}`} type="range" min="1" max="10" step=".1" value={displayed[criterion]} onChange={event=>{setAutoScores(false);onValue(criterion,Number(event.target.value),displayed)}}/><small>1</small><small>10</small></label>)}</div>
    <p className="space-source-note" aria-live="polite">{active?(model?.rows.length?`Live loft connected · ${model.rows.length} transverse profiles · ${model.rails.length} primary rails`:"Active destination · select source images to generate the loft"):"Inactive destination · click this component to connect the live loft"}. Criteria are independent ratings for this space.</p>
    </div>
  </article>;
}

function IllustratorExportNode({classificationMode,setClassificationMode,manualLabel,onManualLabel,criteria,images,model,space,capture,position,onDragStart}:{classificationMode:"automatic"|"manual";setClassificationMode:(mode:"automatic"|"manual")=>void;manualLabel:string;onManualLabel:(label:string)=>void;criteria:Record<SpaceCriterion,number>;images:StoredImage[];model:LoftModel;space:SpaceKind;capture:()=>string;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
 const scores=useMemo(()=>analyzeLoft(model),[model]),match=useMemo(()=>scores?rankTypologies(scores,SPACE_PRESETS[space])[0]:null,[scores,space]);
 const [preview,setPreview]=useState<string|null>(null),[error,setError]=useState(""),[busy,setBusy]=useState(false);
 const exportScores=classificationMode==="automatic"?scores:criteria;
 const exportLabel=classificationMode==="automatic"?(match?.name||""):manualLabel.trim();
 useEffect(()=>{setPreview(null)},[model,space,classificationMode,exportLabel,criteria]);
 useEffect(()=>()=>{if(preview)URL.revokeObjectURL(preview)},[preview]);
 const make=async()=>{if(!scores||!exportLabel)return null;const clay=capture();const sourceImages=await Promise.all(images.slice(0,3).map(async image=>{const response=await fetch(source(image.key));if(!response.ok)throw new Error("Could not load a source image.");const blob=await response.blob();const data=await new Promise<string>((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=()=>reject(new Error("Could not embed a source image."));reader.readAsDataURL(blob)});return {data,name:image.name}}));return illustratorSheet(clay,exportLabel,SPACE_TYPES.find(s=>s.id===space)!.title,exportScores!,SPACE_COLORS[space],sourceImages)};
 const run=async(download:boolean)=>{setBusy(true);setError("");try{const svg=await make();if(!svg)return;const blob=new Blob([svg],{type:"image/svg+xml"});if(download){await saveExport(blob,`sand-scan-${space}-typology.svg`)}else setPreview(URL.createObjectURL(blob))}catch(e){setError(e instanceof Error?e.message:"Could not export the sheet.")}finally{setBusy(false)}};
 return <article className="spatial-rational-node illustrator-export-node" style={{left:position.x,top:position.y}}><header onPointerDown={onDragStart}><div className="model-icon"><Download size={20}/></div><div><h2>Illustrator Export</h2><p>Current clay view · assigned typology · descriptors</p></div><span>DRAG · SVG</span></header><div className="illustrator-content"><p>{SPACE_TYPES.find(s=>s.id===space)!.title} · {exportLabel||"Enter a classification"}</p><fieldset className="classification-export-controls"><legend>Geometric classification</legend><div className="loft-render-toolbar" role="group" aria-label="Classification source"><button aria-pressed={classificationMode==="automatic"} onClick={()=>setClassificationMode("automatic")}>Automatic match</button><button aria-pressed={classificationMode==="manual"} onClick={()=>{if(classificationMode!=="manual"&&match)onManualLabel(match.name);setClassificationMode("manual")}}>Choose manually</button></div>{classificationMode==="manual"&&<><label>Reference typology<select value={SPACE_PRESETS[space].some(p=>p.name===manualLabel)?manualLabel:"custom"} onChange={event=>onManualLabel(event.target.value==="custom"?"":event.target.value)}>{SPACE_PRESETS[space].map(preset=><option key={preset.name} value={preset.name}>{preset.name}</option>)}<option value="custom">Custom classification</option></select></label><label>Exported classification<input type="text" maxLength={100} value={manualLabel} onChange={event=>onManualLabel(event.target.value)} placeholder="Enter your typology label"/></label><p>Your label is used in the preview and Illustrator export. Selecting a reference loads its six saved criteria in Spatial Qualities and this export. Custom labels keep the current criteria.</p></>}<p>Automatic suggestion: {match?.name||"Generate a loft first"}</p></fieldset><p>Matches the reference layout: clay above, typology below, then three source images and six gradient rated bars. Exports the current camera, zoom, pan, thickness, and clipping view in clay.</p><div className="loft-render-toolbar"><button disabled={!scores||!exportLabel||busy} onClick={()=>run(false)}>Preview current sheet</button><button disabled={!scores||!exportLabel||busy} onClick={()=>run(true)}>Export for Illustrator</button></div>{error&&<p role="alert">{error}</p>}{preview&&<img className="illustrator-sheet-preview" src={preview} alt="Clay view, assigned typology and six measured descriptor bars"/>}<p>Illustrator-compatible SVG. Text and bars are editable vectors; the clay view is an embedded image. Open in Illustrator and save as .ai. Pause 360° rotation to choose a specific angle.</p></div><span className="port port-left" aria-hidden="true"/></article>;
}

export default function PhotoModel({images,position,onDragStart}:{images:StoredImage[];position:{x:number;y:number};onDragStart:(e:React.PointerEvent)=>void}){
  const [selected,setSelected]=useWorkspaceState<string[]>("PhotoModel.selected",[]);
  const [fields,setFields]=useState<Float32Array[]>([]),[busy,setBusy]=useState(true);
  const [interval,setInterval]=useWorkspaceState("PhotoModel.interval",.68),[elongation,setElongation]=useWorkspaceState("PhotoModel.elongation",165),[continuity,setContinuity]=useWorkspaceState("PhotoModel.continuity",72),[scan,setScan]=useWorkspaceState("PhotoModel.scan",50),[density,setDensity]=useWorkspaceState("PhotoModel.density",42),[depthGain,setDepthGain]=useWorkspaceState("PhotoModel.depthGain",68),[voidTransform,setVoidTransform]=useWorkspaceState("PhotoModel.voidTransform",0);
  const [angle,setAngle]=useWorkspaceState("PhotoModel.angle",.62),[tilt,setTilt]=useWorkspaceState("PhotoModel.tilt",.3),[lineFidelity,setLineFidelity]=useWorkspaceState("PhotoModel.lineFidelity",76),[primaryLineCount,setPrimaryLineCount]=useWorkspaceState("PhotoModel.primaryLineCount",5),[offsetsPerPrimary,setOffsetsPerPrimary]=useWorkspaceState("PhotoModel.offsetsPerPrimary",2),[bifurcationsPerPrimary,setBifurcationsPerPrimary]=useWorkspaceState("PhotoModel.bifurcationsPerPrimary",3),[bifurcationDistance,setBifurcationDistance]=useWorkspaceState("PhotoModel.bifurcationDistance",65),[bifurcationScale,setBifurcationScale]=useWorkspaceState("PhotoModel.bifurcationScale",60),[primarySmoothing,setPrimarySmoothing]=useWorkspaceState("PhotoModel.primarySmoothing",72),[gridSize,setGridSize]=useWorkspaceState("PhotoModel.gridSize",8),[gridAlignment,setGridAlignment]=useWorkspaceState("PhotoModel.gridAlignment",64),[exportHeight,setExportHeight]=useWorkspaceState("PhotoModel.exportHeight",144);
  const [circulationSettings,setCirculationSettings]=useWorkspaceState<CirculationSettings>("PhotoModel.circulationSettings",{primaryCount:2,pathWidth:5.5,activeBranches:2,junctionWidth:55,smoothing:70});
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;a:number;t:number}|null>(null);
  const captureClay=useRef<(()=>string)|null>(null);
  const [loftThickness,setLoftThickness]=useWorkspaceState("PhotoModel.loftThickness",3);
  const [loftMembrane,setLoftMembrane]=useState<LoftModel>({rows:[],rails:[]});
  const [minimumArchitecturalHeight,setMinimumArchitecturalHeight]=useWorkspaceState("PhotoModel.minimumArchitecturalHeight",6);
  const architecturalResult=useMemo(()=>enforceMinimumArchitecturalHeight(loftMembrane,minimumArchitecturalHeight),[loftMembrane,minimumArchitecturalHeight]);
  const [renderHeight,setRenderHeight]=useState(1070);
  const [activeSpace,setActiveSpace]=useWorkspaceState<SpaceKind>("PhotoModel.activeSpace","lobby");
  const [spaceValues,setSpaceValues]=useWorkspaceState<Record<SpaceKind,Record<SpaceCriterion,number>>>("PhotoModel.spaceValues",()=>Object.fromEntries(SPACE_TYPES.map(space=>[space.id,presetValues(space.id,0)])) as Record<SpaceKind,Record<SpaceCriterion,number>>);
  const [spacePresets,setSpacePresets]=useWorkspaceState<Record<SpaceKind,number>>("PhotoModel.spacePresets",{lobby:0,office:0,gathering:0});
  const [manualSpaces,setManualSpaces]=useWorkspaceState<Record<SpaceKind,boolean>>("PhotoModel.manualSpaces",{lobby:false,office:false,gathering:false});
  const [manualLabels,setManualLabels]=useWorkspaceState<Record<SpaceKind,string>>("PhotoModel.manualLabels",{lobby:SPACE_PRESETS.lobby[0].name,office:SPACE_PRESETS.office[0].name,gathering:SPACE_PRESETS.gathering[0].name});
  const chooseClassification=(label:string)=>{
    setManualLabels(current=>({...current,[activeSpace]:label}));
    setManualSpaces(current=>({...current,[activeSpace]:true}));
    const index=SPACE_PRESETS[activeSpace].findIndex(preset=>preset.name===label);
    if(index>=0){setSpacePresets(current=>({...current,[activeSpace]:index}));setSpaceValues(current=>({...current,[activeSpace]:presetValues(activeSpace,index)}));}
  };
  const loftSourceImages=selected.map(key=>images.find(image=>image.key===key)).filter((image):image is StoredImage=>Boolean(image));
  type ChildNode="export"|"boundary"|"grid"|"circulation"|"loft"|"architecture"|"loftRender"|"spaces"|"illustrator"|"screenshots";
  const [nodePositions,setNodePositions]=useWorkspaceState<Record<ChildNode,{x:number;y:number}>>("PhotoModel.nodePositions",()=>({screenshots:{x:position.x+4580,y:position.y+2000},grid:{x:position.x+760,y:position.y},boundary:{x:position.x+1520,y:position.y},circulation:{x:position.x+2280,y:position.y},loft:{x:position.x+3060,y:position.y},architecture:{x:position.x+3440,y:position.y},loftRender:{x:position.x+3820,y:position.y},export:{x:position.x+4580,y:position.y},spaces:{x:position.x+3820,y:position.y+1200},illustrator:{x:position.x+4580,y:position.y+850}}));
  useEffect(()=>{if(!nodePositions.architecture)setNodePositions(current=>current.architecture?current:{...current,architecture:{x:current.loft.x+380,y:current.loft.y}})},[nodePositions,setNodePositions]);
  const nodeDrag=useRef<{node:ChildNode;startX:number;startY:number;origin:{x:number;y:number}}|null>(null);
  useEffect(()=>{
    const move=(event:PointerEvent)=>{const active=nodeDrag.current;if(!active)return;setNodePositions(current=>({...current,[active.node]:{x:Math.max(12,active.origin.x+event.clientX-active.startX),y:Math.max(20,active.origin.y+event.clientY-active.startY)}}))};
    const up=()=>{nodeDrag.current=null};
    window.addEventListener("pointermove",move);window.addEventListener("pointerup",up);window.addEventListener("pointercancel",up);
    return()=>{window.removeEventListener("pointermove",move);window.removeEventListener("pointerup",up);window.removeEventListener("pointercancel",up)};
  },[]);
  const startChildDrag=(node:ChildNode)=>(event:React.PointerEvent)=>{if(event.button!==0||(event.target as HTMLElement).closest("button,input,canvas"))return;event.preventDefault();nodeDrag.current={node,startX:event.clientX,startY:event.clientY,origin:nodePositions[node]}};
  useEffect(()=>{if(!images.length)return;setSelected(current=>{const available=new Set(images.map(image=>image.key));return current.filter(key=>available.has(key))})},[images]);
  useEffect(()=>{
    let active=true;setBusy(true);
    Promise.all(selected.map(key=>new Promise<Float32Array>(resolve=>{
      const img=new Image();img.onload=()=>{const c=document.createElement("canvas");c.width=W;c.height=H;const ctx=c.getContext("2d",{willReadFrequently:true});if(!ctx){resolve(new Float32Array(W*H));return}ctx.fillStyle="#000";ctx.fillRect(0,0,W,H);ctx.drawImage(img,0,0,W,H);resolve(fieldFromImage(ctx.getImageData(0,0,W,H).data,depthGain))};
      img.onerror=()=>resolve(new Float32Array(W*H));img.src=source(key);
    }))).then(result=>{if(active){setFields(result);setBusy(false)}});
    return()=>{active=false};
  },[selected,depthGain]);
  const transformedFields=useMemo(()=>fields.map(field=>transformVoids(field,voidTransform)),[fields,voidTransform]);
  const slices=useMemo(()=>interpolate(transformedFields,interval*elongation/100,continuity),[transformedFields,interval,elongation,continuity]);
  const gridSlices=useMemo(()=>gridAlignSlices(slices,gridSize,gridAlignment).map(slice=>({...slice,field:keepLargestConnectedField(slice.field)})),[slices,gridSize,gridAlignment]);
  const scanIndex=Math.round(scan/100*Math.max(0,slices.length-1));
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;
    const width=c.width,height=c.height;ctx.fillStyle="#0c1114";ctx.fillRect(0,0,width,height);
    ctx.strokeStyle="#203036";ctx.lineWidth=1;
    for(let x=20;x<width;x+=34){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,height);ctx.stroke()}
    for(let y=20;y<height;y+=34){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(width,y);ctx.stroke()}
    if(!slices.length){ctx.fillStyle="#b8c8ca";ctx.font="16px sans-serif";ctx.textAlign="center";ctx.fillText(busy?"Reconstructing slices…":"Select photos to scan",width/2,height/2);return}
    const ca=Math.cos(angle),sa=Math.sin(angle),ct=Math.cos(tilt),st=Math.sin(tilt);
    const span=Math.abs(slices[0].z-slices.at(-1)!.z),pW=2.25*Math.abs(ca)+span*Math.abs(sa),pH=1.7*Math.abs(ct)+(2.25*Math.abs(sa)+span*Math.abs(ca))*Math.abs(st);
    const unit=Math.min(190,490*.8/Math.max(1.8,pW),height*.76/Math.max(1.7,pH));
    const project=(x:number,y:number,z:number)=>({x:260+(x*ca-z*sa)*unit,y:height/2-(y*ct-(x*sa+z*ca)*st)*unit});
    for(const {slice,i} of slices.map((slice,i)=>({slice,i})).sort((a,b)=>a.slice.z*ca-b.slice.z*ca)){
      const p=project(-1.125,.85,slice.z),px=project(1.125,.85,slice.z),py=project(-1.125,-.85,slice.z);
      ctx.save();ctx.setTransform((px.x-p.x)/W,(px.y-p.y)/W,(py.x-p.x)/H,(py.y-p.y)/H,p.x,p.y);
      ctx.globalAlpha=i===scanIndex?.98:density/100*.43;ctx.imageSmoothingEnabled=true;ctx.drawImage(slice.image,0,0);
      ctx.globalAlpha=i===scanIndex?.95:.14;ctx.strokeStyle=i===scanIndex?"#bde5dc":"#b6c9c9";ctx.lineWidth=i===scanIndex?.22:.11;contours(ctx,slice.field);
      if(i===scanIndex){ctx.globalAlpha=1;ctx.strokeStyle=VIEWPORT_TERTIARY;ctx.lineWidth=.18;drawVoidPerimeters(ctx,slice.field)}
      if(i===scanIndex){ctx.strokeStyle=VIEWPORT_SECONDARY;ctx.lineWidth=.14;ctx.strokeRect(0,0,W,H)}ctx.restore();
    }
    const active=slices[scanIndex],bx=535,by=85,bw=165,bh=165*H/W;
    ctx.fillStyle="#10191c";ctx.fillRect(520,48,190,240);ctx.strokeStyle="#55747a";ctx.lineWidth=1;ctx.strokeRect(520.5,48.5,189,239);
    ctx.fillStyle="#9fc6c7";ctx.font="12px sans-serif";ctx.textAlign="left";ctx.fillText("ACTIVE SECTION",bx,72);
    ctx.globalAlpha=1;ctx.drawImage(active.image,bx,by,bw,bh);ctx.save();ctx.setTransform(bw/W,0,0,bh/H,bx,by);ctx.strokeStyle=VIEWPORT_TERTIARY;ctx.lineWidth=.22;drawVoidPerimeters(ctx,active.field);ctx.restore();ctx.strokeStyle=VIEWPORT_SECONDARY;ctx.strokeRect(bx+.5,by+.5,bw,bh);
    ctx.strokeStyle="#659494";ctx.beginPath();ctx.moveTo(bx+bw/2,by);ctx.lineTo(bx+bw/2,by+bh);ctx.moveTo(bx,by+bh/2);ctx.lineTo(bx+bw,by+bh/2);ctx.stroke();
    ctx.fillStyle="#a7bec0";ctx.font="12px sans-serif";ctx.fillText(`${scan}% THROUGH VOLUME`,bx,by+bh+23);
  },[slices,scanIndex,scan,density,angle,tilt,busy]);
  const toggle=(key:string)=>setSelected(current=>current.includes(key)?current.filter(item=>item!==key):[...current,key]);
  const deselectSelection=()=>setSelected([]);
  const exportPosition=nodePositions.export;
  const boundaryPosition=nodePositions.boundary;
  const gridPosition=nodePositions.grid;
  const circulationPosition=nodePositions.circulation;
  const loftPosition=nodePositions.loft;
  const loftRenderPosition=nodePositions.loftRender;
  const horizontalWire=(start:{x:number;y:number},end:{x:number;y:number})=>`M ${start.x} ${start.y} C ${start.x+(end.x-start.x)*.48} ${start.y}, ${end.x-(end.x-start.x)*.48} ${end.y}, ${end.x} ${end.y}`;
  const verticalWire=(start:{x:number;y:number},end:{x:number;y:number})=>`M ${start.x} ${start.y} C ${start.x} ${start.y+(end.y-start.y)*.48}, ${end.x} ${end.y-(end.y-start.y)*.48}, ${end.x} ${end.y}`;
  return <>
    <svg className="workflow-wires" aria-hidden="true">
      <path d={horizontalWire({x:loftRenderPosition.x+700,y:loftRenderPosition.y+500},{x:nodePositions.illustrator.x,y:nodePositions.illustrator.y+286})}/>
      <path style={{stroke:SPACE_COLORS[activeSpace]}} className="space-connection space-connection-active" d={verticalWire({x:loftRenderPosition.x+350,y:loftRenderPosition.y+renderHeight},{x:nodePositions.spaces.x+350,y:nodePositions.spaces.y})}/>
      <path d={horizontalWire({x:loftRenderPosition.x+700,y:loftRenderPosition.y+286},{x:exportPosition.x,y:exportPosition.y+286})}/>
      <path d={horizontalWire({x:position.x+620,y:position.y+286},{x:gridPosition.x,y:gridPosition.y+286})}/>
      <path d={horizontalWire({x:gridPosition.x+620,y:gridPosition.y+286},{x:boundaryPosition.x,y:boundaryPosition.y+286})}/>
      <path d={horizontalWire({x:boundaryPosition.x+620,y:boundaryPosition.y+286},{x:circulationPosition.x,y:circulationPosition.y+286})}/>
      <path d={horizontalWire({x:circulationPosition.x+700,y:circulationPosition.y+286},{x:loftPosition.x,y:loftPosition.y+286})}/>
      <path d={horizontalWire({x:loftPosition.x+700,y:loftPosition.y+286},{x:loftRenderPosition.x,y:loftRenderPosition.y+286})}/>
    </svg>
    <article className="model-node ct-node" data-node="model" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><ScanLine size={21}/></div><div><h2>Depth Maps → CT Volume</h2><p>Cubic depth blending · continuous massing</p></div><span>DRAG</span></header>
    <div className="model-stage"><canvas ref={canvas} width={720} height={380} aria-label="Rotatable CT style volume with highlighted scan section" onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);orbit.current={x:e.clientX,y:e.clientY,a:angle,t:tilt}}} onPointerMove={e=>{const start=orbit.current;if(start){setAngle(start.a+(e.clientX-start.x)*.009);setTilt(Math.max(-1.2,Math.min(1.2,start.t+(e.clientY-start.y)*.006)))}}} onPointerUp={()=>{orbit.current=null}}/><span>Drag to orbit</span></div>
    <div className="ct-controls"><label>Scan position <strong>{scan}%</strong><input type="range" min="0" max="100" value={scan} onChange={e=>setScan(Number(e.target.value))}/></label><label>Volume density <strong>{density}%</strong><input type="range" min="10" max="100" value={density} onChange={e=>setDensity(Number(e.target.value))}/></label><label className="depth-gain">Depth separation <strong>{depthGain}%</strong><input type="range" min="10" max="100" value={depthGain} onChange={e=>setDepthGain(Number(e.target.value))}/></label><label className="massing-shape">Massing elongation <strong>{elongation}%</strong><input type="range" min="75" max="350" step="5" value={elongation} onChange={e=>setElongation(Number(e.target.value))}/><span className="range-ends"><i>Compact</i><i>Elongated</i></span></label><label className="massing-shape">Surface continuity <strong>{continuity}%</strong><input type="range" min="0" max="100" step="2" value={continuity} onChange={e=>setContinuity(Number(e.target.value))}/><span className="range-ends"><i>Sectional</i><i>Smooth + curved</i></span></label><label className="void-transform">Void / open-area transform <strong>{voidTransform===0?"Unchanged":voidTransform>0?`Expand +${voidTransform}%`:`Compress ${Math.abs(voidTransform)}%`}</strong><input type="range" min="-100" max="100" step="5" value={voidTransform} onChange={e=>setVoidTransform(Number(e.target.value))}/><span className="range-ends"><i>Compress openings</i><i>Expand openings</i></span></label></div>
    <div className="model-controls"><label>Slice interval <input type="range" min=".4" max="1.2" step=".02" value={interval} onChange={e=>setInterval(Number(e.target.value))}/></label><span className="export-routed">Exports routed to the connected Rhino component</span></div>
    <div className="model-sources"><div className="model-caption"><strong>Imported depth sequence</strong><div><span>{selected.length} of {images.length} slices</span><button className="reset-selection" onClick={deselectSelection} disabled={!selected.length} title="Deselect every image from the CT sequence"><XCircle size={13}/>Deselect selection</button></div></div>{([{id:"existing",name:"Model Photos"},{id:"new",name:"Midjourney Photos"}] as const).map(folder=><div className="depth-folder" key={folder.id}><h3>{folder.name} <small>{images.filter(image=>(image.folder||"existing")===folder.id).length} photos</small></h3><div className="model-thumbs">{images.filter(image=>(image.folder||"existing")===folder.id).map(image=><button key={image.key} className={selected.includes(image.key)?"active":""} onClick={()=>toggle(image.key)} aria-pressed={selected.includes(image.key)} title={image.name}><img src={source(image.key)} alt=""/><span>{image.name}</span></button>)}</div>{!images.some(image=>(image.folder||"existing")===folder.id)&&<p className="depth-folder-empty">No photos yet. Add images to this folder in Image Storage.</p>}</div>)}{!images.length&&<div className="depth-empty">Import images in Image Storage to build the CT sequence.</div>}</div>
    <p className="model-note">Each photo is read as a depth map with perimeter-aware cavity detection. The CT result is sent horizontally to Grid Rationalization, establishing the ordered framework that the Unified Section Cage uses to generate primary, offset, and bifurcating lines.</p><span className="port port-left port-storage-in" aria-hidden="true"/><span className="port port-right port-grid-out" aria-hidden="true"/>
    </article>
    <GridRationalizationPreview slices={gridSlices} gridSize={gridSize} alignment={gridAlignment} setGridSize={setGridSize} setAlignment={setGridAlignment} position={gridPosition} onDragStart={startChildDrag("grid")}/>
    <ReadOnlyBoundaryLinesPreview slices={gridSlices} activeIndex={scanIndex} fidelity={lineFidelity} setFidelity={setLineFidelity} primaryCount={primaryLineCount} setPrimaryCount={setPrimaryLineCount} offsetsPerPrimary={offsetsPerPrimary} setOffsetsPerPrimary={setOffsetsPerPrimary} bifurcationsPerPrimary={bifurcationsPerPrimary} setBifurcationsPerPrimary={setBifurcationsPerPrimary} bifurcationDistance={bifurcationDistance} setBifurcationDistance={setBifurcationDistance} bifurcationScale={bifurcationScale} setBifurcationScale={setBifurcationScale} smoothing={primarySmoothing} setSmoothing={setPrimarySmoothing} position={boundaryPosition} onDragStart={startChildDrag("boundary")}/>
    <CirculationPreview slices={gridSlices} fidelity={lineFidelity} primaryCount={primaryLineCount} offsetsPerPrimary={offsetsPerPrimary} bifurcationsPerPrimary={bifurcationsPerPrimary} bifurcationDistance={bifurcationDistance} bifurcationScale={bifurcationScale} smoothing={primarySmoothing} settings={circulationSettings} setSettings={setCirculationSettings} heightInches={exportHeight} setHeightInches={setExportHeight} position={circulationPosition} onDragStart={startChildDrag("circulation")}/>
    <PrimaryCirculationLoftPreview slices={gridSlices} fidelity={lineFidelity} primaryCount={primaryLineCount} offsetsPerPrimary={offsetsPerPrimary} bifurcationsPerPrimary={bifurcationsPerPrimary} bifurcationDistance={bifurcationDistance} bifurcationScale={bifurcationScale} smoothing={primarySmoothing} circulationSettings={circulationSettings} heightInches={exportHeight} position={loftPosition} onDragStart={startChildDrag("loft")} onMembrane={setLoftMembrane}/>
    <ArchitecturalAdjustmentNode model={architecturalResult.model} minimumFeet={minimumArchitecturalHeight} setMinimumFeet={setMinimumArchitecturalHeight} heightFeet={architecturalResult.heightFeet} scaleFactor={architecturalResult.factor} position={nodePositions.architecture??{x:loftPosition.x+380,y:loftPosition.y}} onDragStart={startChildDrag("architecture")}/>
    <CirculationLoftRender onCapture={capture=>{captureClay.current=capture}} onHeight={setRenderHeight} model={architecturalResult.model} thickness={loftThickness} setThickness={setLoftThickness} position={loftRenderPosition} onDragStart={startChildDrag("loftRender")}/>
    <ComponentScreenshotExport position={nodePositions.screenshots} onDragStart={startChildDrag("screenshots")}/>
    <IllustratorExportNode classificationMode={manualSpaces[activeSpace]?"manual":"automatic"} setClassificationMode={mode=>setManualSpaces(current=>({...current,[activeSpace]:mode==="manual"}))} manualLabel={manualLabels[activeSpace]} onManualLabel={chooseClassification} criteria={spaceValues[activeSpace]} images={loftSourceImages} model={architecturalResult.model} space={activeSpace} capture={()=>{if(!captureClay.current)throw new Error("The render is not ready yet.");return captureClay.current()}} position={nodePositions.illustrator} onDragStart={startChildDrag("illustrator")}/>
    <SpaceQualitiesNode autoScores={!manualSpaces[activeSpace]} setAutoScores={value=>setManualSpaces(current=>({...current,[activeSpace]:!value}))} manualLabel={manualLabels[activeSpace]} presetIndex={spacePresets[activeSpace]} onPreset={index=>chooseClassification(SPACE_PRESETS[activeSpace][index].name)} activeSpace={activeSpace} onSpaceChange={setActiveSpace} title={SPACE_TYPES.find(space=>space.id===activeSpace)!.title} number={SPACE_TYPES.find(space=>space.id===activeSpace)!.number} images={loftSourceImages} active={true} onActivate={()=>{}} values={spaceValues[activeSpace]} onValue={(criterion,value,base)=>setSpaceValues(current=>({...current,[activeSpace]:{...(base||current[activeSpace]),[criterion]:value}}))} model={architecturalResult.model} position={nodePositions.spaces} onDragStart={startChildDrag("spaces")}/>
    <ExportNode model={architecturalResult.model} thickness={loftThickness} slices={gridSlices} lineFidelity={lineFidelity} primaryCount={primaryLineCount} offsetsPerPrimary={offsetsPerPrimary} exportHeight={exportHeight} setExportHeight={setExportHeight} position={exportPosition} onDragStart={startChildDrag("export")}/>
  </>;
}

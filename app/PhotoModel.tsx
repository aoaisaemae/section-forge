"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Box, Download, Grid3X3, Hand, Layers3, MousePointer2, Redo2, RotateCcw, Route, ScanLine, Sparkles, Undo2, XCircle } from "lucide-react";

type StoredImage={key:string;name:string;size:number;uploadedAt:string};
const source=(key:string)=>`/api/images/content?key=${encodeURIComponent(key)}`;
const W=64,H=48,steps=8;
type Slice={field:Float32Array;z:number;image:HTMLCanvasElement};
type Vec=[number,number,number];
type RealitySettings={supportSpacing:number;clearance:number;maxSlope:number};

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
  const url=URL.createObjectURL(new Blob([obj],{type:"text/plain"})),link=document.createElement("a");link.href=url;link.download="ct-volume-study.obj";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
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
  const bytes=file.toByteArrayOptions(options),url=URL.createObjectURL(new Blob([bytes],{type:"application/octet-stream"})),link=document.createElement("a");
  link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);options.delete();
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
  const file=new rhino.File3dm();file.applicationName="Section Forge";file.applicationDetails=`Photo to CT Volume · Rhino ${version} SubD massing export`;
  file.startSectionComments=`Editable multi-boundary CT loft written for Rhino ${version}. Outer and internal section boundaries are preserved as quad surfaces. Select CT_SUBD_MASSING and run ToSubD with Creases=No.`;
  const layerIndex=file.layers().addLayer("CT_SUBD_MASSING",{r:124,g:205,b:194,a:255});
  const attributes=new rhino.ObjectAttributes();attributes.name="CT_SUBD_MASSING";attributes.layerIndex=layerIndex;
  attributes.setUserString("NextCommand","_ToSubD _Creases=_No");
  file.objects().addMesh(mesh,attributes);
  downloadRhinoFile(file,rhino,version,`section-forge-subd-rhino${version}.3dm`);
  attributes.delete();mesh.delete();file.delete();
}

async function exportRhinoSections(slices:Slice[],version:7|8){
  if(!slices.length)return;
  const [{default:rhino3dm},{default:wasmUrl}]=await Promise.all([import("rhino3dm"),import("rhino3dm/rhino3dm.wasm?url")]);
  const rhino=await rhino3dm({locateFile:()=>wasmUrl}),file=new rhino.File3dm();
  file.applicationName="Section Forge";file.applicationDetails=`CT section surfaces · Rhino ${version}`;
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
  downloadRhinoFile(file,rhino,version,`section-forge-surfaces-rhino${version}.3dm`);file.delete();
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
type FloorPlateSettings={noiseReduction:number;plateThickness:number;surfaceFlow:number};
type CirculationRibbon={center:Vec[];left:Vec[];right:Vec[];kind:"spine"|"branch"};
type CirculationGeometry={ribbons:CirculationRibbon[];floorY:number;topY:number;trimmedCount:number};
type SpatialPlate={profiles:Vec[][][];cellCount:number};
const CHUNK_HALF=1.05;

function smoothSpatialRoute(line:Vec[],amount:number){
  let result=line.filter((point,index)=>index===0||Math.hypot(point[0]-line[index-1][0],point[1]-line[index-1][1],point[2]-line[index-1][2])>.025);
  const passes=Math.round(Math.max(0,Math.min(100,amount))/20),weight=.34;
  for(let pass=0;pass<passes;pass++)result=result.map((point,index)=>index===0||index===result.length-1?point:[point[0]*(1-weight)+(result[index-1][0]+result[index+1][0])*.5*weight,point[1]*(1-weight)+(result[index-1][1]+result[index+1][1])*.5*weight,point[2]*(1-weight)+(result[index-1][2]+result[index+1][2])*.5*weight] as Vec);
  return result;
}

function buildCirculationGeometry(families:LongitudinalFamily,settings:CirculationSettings,heightInches:number,bifurcationsPerPrimary:number):CirculationGeometry{
  const floorY=-CHUNK_HALF+.08,heightFeet=heightInches/12,topY=Math.min(CHUNK_HALF,floorY+heightFeet/20*(CHUNK_HALF*2));
  if(!families.primary.length)return{ribbons:[],floorY,topY,trimmedCount:0};
  const selectedCount=Math.max(1,Math.min(families.primary.length,settings.primaryCount));
  const selectedIndices=Array.from({length:selectedCount},(_,index)=>selectedCount===1?0:Math.round(index*(families.primary.length-1)/(selectedCount-1)));
  const sources:[Vec[],"spine"|"branch"][]=[];
  selectedIndices.forEach(primaryIndex=>{
    sources.push([families.primary[primaryIndex],"spine"]);
    const branchStart=primaryIndex*Math.max(0,bifurcationsPerPrimary);
    families.branches.slice(branchStart,branchStart+Math.max(0,settings.activeBranches)).forEach(line=>sources.push([line,"branch"]));
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

function buildUnifiedFloorPlate(circulation:CirculationGeometry,settings:FloorPlateSettings):SpatialPlate{
  const sample=(line:Vec[],count:number)=>Array.from({length:count},(_,index)=>{const position=index/Math.max(1,count-1)*Math.max(0,line.length-1),a=Math.floor(position),b=Math.min(line.length-1,a+1),t=position-a;return[line[a][0]+(line[b][0]-line[a][0])*t,line[a][1]+(line[b][1]-line[a][1])*t,line[a][2]+(line[b][2]-line[a][2])*t] as Vec});
  const rails=circulation.ribbons.map(ribbon=>smoothSpatialRoute(ribbon.center,settings.noiseReduction)).filter(line=>line.length>1),spaceExpansion=Math.max(0,Math.min(100,settings.surfaceFlow))/100;
  const pairs=rails.length>1?rails.slice(0,-1).map((rail,index)=>[rail,rails[index+1]] as [Vec[],Vec[]]):circulation.ribbons.slice(0,1).map(ribbon=>[smoothSpatialRoute(ribbon.left,settings.noiseReduction),smoothSpatialRoute(ribbon.right,settings.noiseReduction)] as [Vec[],Vec[]]);
  const profiles=pairs.map(([railA,railB])=>{
    const count=Math.max(2,Math.max(railA.length,railB.length)),a=sample(railA,count),b=sample(railB,count);
    return Array.from({length:count},(_,index)=>{const progress=index/Math.max(1,count-1),node=Math.pow(Math.sin(Math.PI*progress),4),expansion=1+spaceExpansion*(.08+node*.35),mid:[number,number,number]=[(a[index][0]+b[index][0])/2,(a[index][1]+b[index][1])/2,(a[index][2]+b[index][2])/2],expand=(point:Vec):Vec=>[Math.max(-CHUNK_HALF,Math.min(CHUNK_HALF,mid[0]+(point[0]-mid[0])*expansion)),Math.max(circulation.floorY,Math.min(circulation.topY,point[1])),Math.max(-CHUNK_HALF,Math.min(CHUNK_HALF,mid[2]+(point[2]-mid[2])*expansion))];return[expand(a[index]),expand(b[index])]});
  });
  const cellCount=profiles.reduce((sum,route)=>sum+Math.max(0,route.length-1)*Math.max(0,(route[0]?.length??1)-1),0);
  return{profiles,cellCount};
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
  file.applicationName="Section Forge";file.applicationDetails=`CT section boundary curves · Rhino ${version}`;
  file.startSectionComments=`Joined section contours written for Rhino ${version}. Exterior perimeters and internal void boundaries are editable curve objects; no pixels or meshes are included.`;
  const layerIndex=file.layers().addLayer("CT_SECTION_BOUNDARY_LINES",{r:139,g:217,b:207,a:255});let curveCount=0;
  slices.forEach((slice,sectionIndex)=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity)).forEach((loop,boundaryIndex)=>{
    const points=loop.map(([x,y])=>[(x/(W-1)-.5)*2.25,(.5-y/(H-1))*1.7,slice.z]),curve=new rhino.PolylineCurve(points),attributes=new rhino.ObjectAttributes();
    attributes.name=`SECTION_${String(sectionIndex+1).padStart(3,"0")}_BOUNDARY_${String(boundaryIndex+1).padStart(2,"0")}`;attributes.layerIndex=layerIndex;
    attributes.setUserString("GeometryType","Joined CT section boundary curve");attributes.setUserString("SectionIndex",String(sectionIndex+1));attributes.setUserString("Depth",slice.z.toFixed(5));attributes.setUserString("BoundaryRole","Exterior perimeter or enclosed void");attributes.setUserString("CurveFidelity",String(fidelity));
    file.objects().addCurve(curve,attributes);curveCount++;attributes.delete();curve.delete();
  }));
  file.startSectionComments+=` ${curveCount} joined boundary curves exported from ${slices.length} sections.`;
  downloadRhinoFile(file,rhino,version,`section-forge-boundary-lines-rhino${version}.3dm`);file.delete();
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

async function exportRhinoBundle(slices:Slice[],version:7|8,fidelity:number,height:number,primaryCount:number,offsetsPerPrimary:number){
  if(!slices.length)return;
  const [{default:rhino3dm},{default:wasmUrl}]=await Promise.all([import("rhino3dm"),import("rhino3dm/rhino3dm.wasm?url")]);
  const rhino=await rhino3dm({locateFile:()=>wasmUrl}),file=new rhino.File3dm();
  const settings=file.settings();settings.modelUnitSystem=rhino.UnitSystem.Inches;settings.pageUnitSystem=rhino.UnitSystem.Feet;settings.modelAbsoluteTolerance=.001;
  file.applicationName="Section Forge";file.applicationDetails=`20ft x 20ft maximum imperial unified-space package · Rhino ${version}`;
  const faces=addPrintableForm(file,rhino,slices,height,primaryCount,offsetsPerPrimary),surfaces=addSectionSurfaces(file,rhino,slices,height),curves=addBoundaryCurves(file,rhino,slices,fidelity,height);
  file.startSectionComments=`Imperial one-file Section Forge export for Rhino ${version}. Model units: inches; layout units: feet. All geometry is constrained to a maximum 20ft x 20ft footprint and ${height/12}ft overall height. The closed printable mesh is one ruled loft driven by ${primaryCount} primary lines with ${offsetsPerPrimary} offsets per primary through every section (${faces} faces), with ${surfaces} reference section surfaces and ${curves} editable boundary curves.`;
  downloadRhinoFile(file,rhino,version,`section-forge-complete-rhino${version}.3dm`);file.delete();
}

function CirculationPreview({slices,fidelity,primaryCount,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,smoothing,settings,setSettings,heightInches,setHeightInches,position,onDragStart}:{slices:Slice[];fidelity:number;primaryCount:number;offsetsPerPrimary:number;bifurcationsPerPrimary:number;bifurcationDistance:number;bifurcationScale:number;smoothing:number;settings:CirculationSettings;setSettings:React.Dispatch<React.SetStateAction<CirculationSettings>>;heightInches:number;setHeightInches:(value:number)=>void;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null),[camera,setCamera]=useState({yaw:-.72,pitch:.38,zoom:1});
  const normalized=useMemo(()=>{if(!slices.length)return[] as Slice[];const center=(slices[0].z+slices.at(-1)!.z)/2,span=Math.max(.001,Math.abs(slices[0].z-slices.at(-1)!.z));return slices.map(slice=>({...slice,z:(slice.z-center)/span*2.1}))},[slices]);
  const loops=useMemo(()=>normalized.map(slice=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity))),[normalized,fidelity]);
  const families=useMemo(()=>longitudinalFamilies(normalized,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,[],{}),[normalized,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale]);
  const circulation=useMemo(()=>buildCirculationGeometry(families,settings,heightInches,bifurcationsPerPrimary),[families,settings,heightInches,bifurcationsPerPrimary]);
  useEffect(()=>{setSettings(current=>({...current,primaryCount:Math.max(1,Math.min(current.primaryCount,Math.max(1,families.primary.length))),activeBranches:Math.min(current.activeBranches,bifurcationsPerPrimary)}))},[families.primary.length,bifurcationsPerPrimary,setSettings]);
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;const width=c.width,height=c.height;ctx.fillStyle="#071014";ctx.fillRect(0,0,width,height);
    if(!circulation.ribbons.length){ctx.fillStyle="#8fa9ae";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select depth sections to generate circulation",width/2,height/2);return}
    const rotate=([x,y,z]:Vec):Vec=>{const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]},project=(point:Vec)=>{const [x,y,z]=rotate(point),perspective=5.2/(5.2+z),scale=180*camera.zoom*perspective;return{x:width*.5+x*scale,y:height*.54-y*scale,z}};
    const bottom=[[-CHUNK_HALF,circulation.floorY,-CHUNK_HALF],[CHUNK_HALF,circulation.floorY,-CHUNK_HALF],[CHUNK_HALF,circulation.floorY,CHUNK_HALF],[-CHUNK_HALF,circulation.floorY,CHUNK_HALF]] as Vec[],top=bottom.map(([x,,z])=>[x,circulation.topY,z] as Vec);ctx.strokeStyle="rgba(119,205,199,.28)";ctx.lineWidth=.8;for(const ring of [bottom,top]){ctx.beginPath();ring.forEach((p,i)=>{const s=project(p);i?ctx.lineTo(s.x,s.y):ctx.moveTo(s.x,s.y)});ctx.closePath();ctx.stroke()}for(let i=0;i<4;i++){const a=project(bottom[i]),b=project(top[i]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
    const cells=circulation.ribbons.flatMap(ribbon=>Array.from({length:Math.max(0,Math.min(ribbon.left.length,ribbon.right.length)-1)},(_,index)=>{const screen=[ribbon.left[index],ribbon.right[index],ribbon.right[index+1],ribbon.left[index+1]].map(project);return{screen,depth:screen.reduce((sum,p)=>sum+p.z,0)/4,kind:ribbon.kind}})).sort((a,b)=>b.depth-a.depth);
    cells.forEach(cell=>{ctx.fillStyle=cell.kind==="spine"?"rgba(95,231,218,.34)":"rgba(242,151,72,.3)";ctx.strokeStyle=cell.kind==="spine"?"rgba(166,255,245,.72)":"rgba(255,203,127,.7)";ctx.lineWidth=.65;ctx.beginPath();ctx.moveTo(cell.screen[0].x,cell.screen[0].y);for(let i=1;i<4;i++)ctx.lineTo(cell.screen[i].x,cell.screen[i].y);ctx.closePath();ctx.fill();ctx.stroke()});
    circulation.ribbons.forEach(ribbon=>{const points=ribbon.center.map(project);ctx.strokeStyle=ribbon.kind==="spine"?"rgba(95,231,218,.96)":"rgba(255,184,97,.9)";ctx.lineWidth=ribbon.kind==="spine"?2:1.4;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke()});ctx.fillStyle="#789a9d";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(`20′ × 20′ ENVELOPE · ${heightInches/12}′ HEIGHT · ONE OCCUPIABLE FLOOR`,14,20);
  },[circulation,camera,heightInches]);
  return <article className="spatial-rational-node circulation-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Route size={19}/></div><div><h2>Circulation Skeleton</h2><p>True 3D cage lines → multiple occupiable routes</p></div><span>DRAG · ORBIT</span></header>
    <div className="rational-viewport"><canvas ref={canvas} width={700} height={390} aria-label="Rotatable 3D preview of circulation routes constrained to a twenty foot square spatial chunk" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.55,Math.min(2.2,value.zoom-event.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span><button onClick={()=>setCamera({yaw:-.72,pitch:.38,zoom:1})}>Reset view</button></div>
    <div className="spatial-constraint-band"><strong>20′ × 20′</strong><span>Single occupiable floor</span><span>Outside geometry trimmed</span><label>Height <b>{heightInches/12}′</b><input type="range" min="96" max="240" step="12" value={heightInches} onChange={event=>setHeightInches(Number(event.target.value))}/></label></div>
    <div className="rational-controls">
      <label><span>Primary circulations</span><strong>{Math.min(settings.primaryCount,Math.max(1,families.primary.length))}</strong><input type="range" min="1" max={Math.max(1,families.primary.length)} step="1" value={Math.min(settings.primaryCount,Math.max(1,families.primary.length))} onChange={event=>setSettings(current=>({...current,primaryCount:Number(event.target.value)}))}/><small>Evenly distributed cage lines, preserved in XYZ</small></label>
      <label><span>Path width</span><strong>{settings.pathWidth}′</strong><input type="range" min="4" max="8" step=".5" value={settings.pathWidth} onChange={event=>setSettings(current=>({...current,pathWidth:Number(event.target.value)}))}/><small>Clear occupiable route</small></label>
      <label><span>Bifurcations per primary</span><strong>{settings.activeBranches}</strong><input type="range" min="0" max={Math.max(0,bifurcationsPerPrimary)} step="1" value={Math.min(settings.activeBranches,bifurcationsPerPrimary)} onChange={event=>setSettings(current=>({...current,activeBranches:Number(event.target.value)}))}/><small>Repeated for every selected primary route</small></label>
      <label><span>Junction widening</span><strong>{settings.junctionWidth}%</strong><input type="range" min="0" max="100" step="5" value={settings.junctionWidth} onChange={event=>setSettings(current=>({...current,junctionWidth:Number(event.target.value)}))}/><small>Threshold expansion</small></label>
      <label><span>Route smoothing</span><strong>{settings.smoothing}%</strong><input type="range" min="0" max="100" step="5" value={settings.smoothing} onChange={event=>setSettings(current=>({...current,smoothing:Number(event.target.value)}))}/><small>Remove directional noise</small></label>
    </div>
    <div className="rational-summary"><span><strong>{Math.min(settings.primaryCount,Math.max(1,families.primary.length))}</strong>primary routes</span><span><strong>{circulation.ribbons.length}</strong>3D paths</span><span><strong>{circulation.trimmedCount}</strong>points trimmed</span><span><strong>XYZ</strong>cage preserved</span></div>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function CageCirculationPreview({slices,fidelity,primaryCount,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,smoothing,circulationSettings,heightInches,position,onDragStart}:{slices:Slice[];fidelity:number;primaryCount:number;offsetsPerPrimary:number;bifurcationsPerPrimary:number;bifurcationDistance:number;bifurcationScale:number;smoothing:number;circulationSettings:CirculationSettings;heightInches:number;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  type FrameworkView="perspective"|"top"|"front"|"back"|"left"|"right";
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null),[view,setView]=useState<FrameworkView>("perspective"),[camera,setCamera]=useState({yaw:-.72,pitch:.36}),[zoom,setZoom]=useState(1),[pathEmphasis,setPathEmphasis]=useState(72);
  const normalized=useMemo(()=>{if(!slices.length)return[] as Slice[];const center=(slices[0].z+slices.at(-1)!.z)/2,span=Math.max(.001,Math.abs(slices[0].z-slices.at(-1)!.z));return slices.map(slice=>({...slice,z:(slice.z-center)/span*2.1}))},[slices]);
  const loops=useMemo(()=>normalized.map(slice=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity))),[normalized,fidelity]);
  const families=useMemo(()=>longitudinalFamilies(normalized,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,[],{}),[normalized,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale]);
  const circulation=useMemo(()=>buildCirculationGeometry(families,circulationSettings,heightInches,bifurcationsPerPrimary),[families,circulationSettings,heightInches,bifurcationsPerPrimary]);
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;const width=c.width,height=c.height;ctx.fillStyle="#071014";ctx.fillRect(0,0,width,height);
    if(!normalized.length){ctx.fillStyle="#8fa9ae";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select depth sections to combine cage and circulation",width/2,height/2);return}
    const orient=([x,y,z]:Vec):Vec=>{if(view==="perspective"){const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]}return view==="top"?[x,-z,y]:view==="front"?[x,y,z]:view==="back"?[-x,y,-z]:view==="left"?[z,y,x]:[-z,y,-x]},project=(point:Vec)=>{const [x,y,depth]=orient(point),perspective=view==="perspective"?5.25/(5.25+depth):1,scale=166*zoom*perspective;return{x:width*.5+x*scale,y:height*.53-y*scale,z:depth}};
    const transverse=loops.flatMap((section,index)=>section.map(loop=>loop.map(([x,y])=>[(x/(W-1)-.5)*2.25,(.5-y/(H-1))*1.7,normalized[index].z] as Vec))).map(line=>({line,depth:line.reduce((sum,p)=>sum+orient(p)[2],0)/Math.max(1,line.length)})).sort((a,b)=>b.depth-a.depth);
    transverse.forEach(({line})=>{const points=line.map(project);ctx.strokeStyle="rgba(94,139,144,.25)";ctx.lineWidth=.55;ctx.beginPath();points.forEach((p,index)=>index?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.stroke()});
    const draw=(lines:Vec[][],color:string,lineWidth:number)=>lines.forEach(line=>{const points=line.map(project);if(points.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);for(let index=1;index<points.length-1;index++){const next=points[index+1],mid={x:(points[index].x+next.x)/2,y:(points[index].y+next.y)/2};ctx.quadraticCurveTo(points[index].x,points[index].y,mid.x,mid.y)}ctx.lineTo(points.at(-1)!.x,points.at(-1)!.y);ctx.stroke()});
    draw(families.offset,"rgba(211,109,224,.46)",.75);draw(families.primary,"rgba(95,231,218,.84)",1.25);draw(families.branches,"rgba(255,184,97,.72)",1);
    const alpha=.2+pathEmphasis/100*.45,cells=circulation.ribbons.flatMap(ribbon=>Array.from({length:Math.max(0,Math.min(ribbon.left.length,ribbon.right.length)-1)},(_,index)=>{const screen=[ribbon.left[index],ribbon.right[index],ribbon.right[index+1],ribbon.left[index+1]].map(project);return{screen,depth:screen.reduce((sum,p)=>sum+p.z,0)/4,kind:ribbon.kind}})).sort((a,b)=>b.depth-a.depth);
    cells.forEach(cell=>{ctx.fillStyle=cell.kind==="spine"?`rgba(95,231,218,${alpha})`:`rgba(242,151,72,${alpha*.9})`;ctx.strokeStyle=cell.kind==="spine"?"rgba(198,255,248,.7)":"rgba(255,211,145,.66)";ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(cell.screen[0].x,cell.screen[0].y);for(let index=1;index<4;index++)ctx.lineTo(cell.screen[index].x,cell.screen[index].y);ctx.closePath();ctx.fill();ctx.stroke()});
    circulation.ribbons.forEach(ribbon=>draw([ribbon.center],ribbon.kind==="spine"?"rgba(225,255,251,.96)":"rgba(255,222,172,.92)",ribbon.kind==="spine"?2.2:1.6));ctx.fillStyle="#789a9d";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(`${view.toUpperCase()} ${view==="perspective"?"VIEW":"ORTHOGRAPHIC"} · SECTION CAGE + OCCUPIABLE CIRCULATION`,14,20);
  },[normalized,loops,families,circulation,view,camera,zoom,pathEmphasis]);
  return <article className="spatial-rational-node combined-framework-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Layers3 size={19}/></div><div><h2>Cage + Circulation Framework</h2><p>Unified section lines and occupiable paths</p></div><span>DRAG · MULTIVIEW</span></header>
    <div className="geometry-mode framework-view-switcher"><span>Viewport</span><div>{(["perspective","top","front","back","left","right"] as FrameworkView[]).map(option=><button key={option} className={view===option?"active":""} onClick={()=>setView(option)}>{option[0].toUpperCase()+option.slice(1)}</button>)}</div><small>Perspective supports orbiting; architectural elevations and plan use undistorted parallel projection.</small></div>
    <div className="rational-viewport"><canvas ref={canvas} width={700} height={390} aria-label={`${view} view of the unified section cage and circulation skeleton`} onPointerDown={event=>{if(view!=="perspective")return;event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(!start||view!=="perspective")return;setCamera({yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))})}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setZoom(value=>Math.max(.55,Math.min(2.2,value-event.deltaY*.0012)))}}/><span>{view==="perspective"?"Drag to orbit · wheel to zoom":`Wheel to zoom · ${view} orthographic`}</span><button onClick={()=>{setZoom(1);setCamera({yaw:-.72,pitch:.36})}}>Reset view</button></div>
    <div className="spatial-constraint-band"><strong>20′ × 20′</strong><span>{heightInches/12}′ adjustable height</span><span>Cage preserved in XYZ</span><span>One circulation system</span></div>
    <div className="rational-controls"><label><span>Circulation emphasis</span><strong>{pathEmphasis}%</strong><input type="range" min="10" max="100" step="5" value={pathEmphasis} onChange={event=>setPathEmphasis(Number(event.target.value))}/><small>Cage dominant</small><small>Paths dominant</small></label></div>
    <div className="rational-summary"><span><strong>{families.primary.length}</strong>primary cage lines</span><span><strong>{families.offset.length}</strong>offset lines</span><span><strong>{families.branches.length}</strong>bifurcations</span><span><strong>{circulation.ribbons.length}</strong>circulation paths</span></div>
    <p className="rational-note">The section cage remains the geometric reference system. Circulation is overlaid on selected primary and bifurcating lines, so the occupiable paths can be evaluated without losing the original longitudinal hierarchy.</p>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function UnifiedFloorPlatePreview({slices,fidelity,primaryCount,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,smoothing,circulationSettings,settings,setSettings,heightInches,position,onDragStart}:{slices:Slice[];fidelity:number;primaryCount:number;offsetsPerPrimary:number;bifurcationsPerPrimary:number;bifurcationDistance:number;bifurcationScale:number;smoothing:number;circulationSettings:CirculationSettings;settings:FloorPlateSettings;setSettings:React.Dispatch<React.SetStateAction<FloorPlateSettings>>;heightInches:number;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null),[camera,setCamera]=useState({yaw:-.72,pitch:.34,zoom:1});
  const normalized=useMemo(()=>{if(!slices.length)return[] as Slice[];const center=(slices[0].z+slices.at(-1)!.z)/2,span=Math.max(.001,Math.abs(slices[0].z-slices.at(-1)!.z));return slices.map(slice=>({...slice,z:(slice.z-center)/span*2.1}))},[slices]);
  const loops=useMemo(()=>normalized.map(slice=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity))),[normalized,fidelity]);
  const families=useMemo(()=>longitudinalFamilies(normalized,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,[],{}),[normalized,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale]);
  const circulation=useMemo(()=>buildCirculationGeometry(families,circulationSettings,heightInches,bifurcationsPerPrimary),[families,circulationSettings,heightInches,bifurcationsPerPrimary]),plate=useMemo(()=>buildUnifiedFloorPlate(circulation,settings),[circulation,settings]);
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;const width=c.width,height=c.height;ctx.fillStyle="#071014";ctx.fillRect(0,0,width,height);
    if(!plate.profiles.length){ctx.fillStyle="#8fa9ae";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Generate circulation to form paths and occupiable spaces",width/2,height/2);return}
    const rotate=([x,y,z]:Vec):Vec=>{const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]},project=(point:Vec)=>{const [x,y,z]=rotate(point),perspective=5.3/(5.3+z),scale=180*camera.zoom*perspective;return{x:width*.5+x*scale,y:height*.53-y*scale,z}};
    const cells=plate.profiles.flatMap(route=>route.slice(0,-1).map((profile,index)=>{const screen=[profile[0],route[index+1][0],route[index+1][1],profile[1]].map(project);return{screen,depth:screen.reduce((sum,p)=>sum+p.z,0)/4}})).sort((a,b)=>b.depth-a.depth);
    cells.forEach(cell=>{ctx.fillStyle="rgba(71,177,168,.72)";ctx.strokeStyle="rgba(185,245,238,.38)";ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(cell.screen[0].x,cell.screen[0].y);for(let i=1;i<4;i++)ctx.lineTo(cell.screen[i].x,cell.screen[i].y);ctx.closePath();ctx.fill();ctx.stroke()});
    plate.profiles.forEach(route=>route.forEach((profile,index)=>{if(index%Math.max(1,Math.floor(route.length/8))!==0&&index!==route.length-1)return;const points=profile.map(project);ctx.strokeStyle="rgba(222,255,251,.55)";ctx.lineWidth=.6;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke()}));ctx.fillStyle="#789a9d";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(`3D RULED FLOOR SURFACES · ${heightInches/12}′ MAX ENVELOPE HEIGHT`,14,20);
  },[plate,camera,heightInches]);
  return <article className="spatial-rational-node floor-plate-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Layers3 size={19}/></div><div><h2>Unified Floor Plate</h2><p>3D circulation rails → ruled paths and spaces</p></div><span>DRAG · ORBIT</span></header>
    <div className="rational-viewport"><canvas ref={canvas} width={760} height={420} aria-label="Rotatable preview of three-dimensional ruled floor surfaces derived from circulation lines" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.55,Math.min(2.2,value.zoom-event.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span><button onClick={()=>setCamera({yaw:-.72,pitch:.34,zoom:1})}>Reset view</button></div>
    <div className="spatial-constraint-band"><strong>20′ × 20′</strong><span>{heightInches/12}′ adjustable height</span><span>One floor only</span><span>Envelope trim active</span></div>
    <div className="rational-controls floor-controls">
      <label><span>Noise removal</span><strong>{settings.noiseReduction}%</strong><input type="range" min="0" max="100" step="5" value={settings.noiseReduction} onChange={event=>setSettings(current=>({...current,noiseReduction:Number(event.target.value)}))}/><small>Simplifies hooks and short deviations</small></label>
      <label><span>Floor plate thickness</span><strong>{settings.plateThickness}″</strong><input type="range" min="6" max="18" step="1" value={settings.plateThickness} onChange={event=>setSettings(current=>({...current,plateThickness:Number(event.target.value)}))}/><small>Single occupiable datum</small></label>
      <label><span>Path-to-space expansion</span><strong>{settings.surfaceFlow}%</strong><input type="range" min="0" max="100" step="5" value={settings.surfaceFlow} onChange={event=>setSettings(current=>({...current,surfaceFlow:Number(event.target.value)}))}/><small>Direct circulation band</small><small>Expanded occupiable nodes</small></label>
    </div>
    <div className="rational-summary"><span><strong>1</strong>continuous floor system</span><span><strong>{plate.profiles.length}</strong>rail pairs</span><span><strong>{plate.cellCount}</strong>ruled faces</span><span><strong>{settings.surfaceFlow}%</strong>space expansion</span></div>
    <p className="rational-note">Adjacent circulation lines are now the actual surface rails. Their corresponding section points are connected by straight rulings, producing floor plates that span between the cage-derived paths while retaining both rails’ height and curvature. No offset ribbon substitutes for the circulation lines, and no enclosure is added.</p>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function UnifiedRuledMeshPreview({slices,fidelity,primaryCount,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,smoothing,circulationSettings,floorSettings,heightInches,position,onDragStart}:{slices:Slice[];fidelity:number;primaryCount:number;offsetsPerPrimary:number;bifurcationsPerPrimary:number;bifurcationDistance:number;bifurcationScale:number;smoothing:number;circulationSettings:CirculationSettings;floorSettings:FloorPlateSettings;heightInches:number;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
  const [camera,setCamera]=useState({yaw:-.68,pitch:.32,zoom:1}),[meshOpacity,setMeshOpacity]=useState(88),[wireframe,setWireframe]=useState(true);
  const meshSlices=useMemo(()=>{
    if(!slices.length)return[] as Slice[];const center=(slices[0].z+slices.at(-1)!.z)/2,span=Math.max(.001,Math.abs(slices[0].z-slices.at(-1)!.z));
    return slices.map(slice=>({...slice,z:(slice.z-center)/span*2.1}));
  },[slices]);
  const loops=useMemo(()=>meshSlices.map(slice=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,fidelity))),[meshSlices,fidelity]);
  const families=useMemo(()=>longitudinalFamilies(meshSlices,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale,[],{}),[meshSlices,loops,primaryCount,smoothing,offsetsPerPrimary,bifurcationsPerPrimary,bifurcationDistance,bifurcationScale]);
  const circulation=useMemo(()=>buildCirculationGeometry(families,circulationSettings,heightInches,bifurcationsPerPrimary),[families,circulationSettings,heightInches,bifurcationsPerPrimary]),plate=useMemo(()=>buildUnifiedFloorPlate(circulation,floorSettings),[circulation,floorSettings]);
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;const width=c.width,height=c.height;ctx.fillStyle="#071014";ctx.fillRect(0,0,width,height);
    if(!meshSlices.length){ctx.fillStyle="#8fa9ae";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select images to generate the unified ruled mesh",width/2,height/2);return}
    const rotate=([x,y,z]:Vec):Vec=>{const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]},project=(point:Vec)=>{const [x,y,z]=rotate(point),perspective=5.4/(5.4+z),scale=205*camera.zoom*perspective;return{x:width*.5+x*scale,y:height*.54-y*scale,z}};
    ctx.strokeStyle="rgba(55,90,105,.28)";ctx.lineWidth=.7;for(let i=-8;i<=8;i++){const a=project([i*.25,circulation.floorY,-CHUNK_HALF]),b=project([i*.25,circulation.floorY,CHUNK_HALF]),d=project([-CHUNK_HALF,circulation.floorY,i*.25]),e=project([CHUNK_HALF,circulation.floorY,i*.25]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.beginPath();ctx.moveTo(d.x,d.y);ctx.lineTo(e.x,e.y);ctx.stroke()}
    const cells=plate.profiles.flatMap(route=>route.slice(0,-1).map((profile,index)=>{const screen=[profile[0],route[index+1][0],route[index+1][1],profile[1]].map(project);return{screen,depth:screen.reduce((sum,p)=>sum+p.z,0)/4}})).sort((a,b)=>b.depth-a.depth);
    cells.forEach(cell=>{const depthTone=Math.max(0,Math.min(1,(cell.depth+2.2)/4.4)),alpha=.22+meshOpacity/100*.62;ctx.fillStyle=`rgba(${Math.round(40+depthTone*32)},${Math.round(148+depthTone*76)},${Math.round(150+depthTone*67)},${alpha})`;ctx.strokeStyle=wireframe?`rgba(181,255,246,${.18+meshOpacity/100*.5})`:"transparent";ctx.lineWidth=.55;ctx.beginPath();ctx.moveTo(cell.screen[0].x,cell.screen[0].y);for(let i=1;i<4;i++)ctx.lineTo(cell.screen[i].x,cell.screen[i].y);ctx.closePath();ctx.fill();if(wireframe)ctx.stroke()});
    const draw=(line:Vec[],color:string,widthValue:number)=>{const pts=line.map(project);if(pts.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=widthValue;ctx.beginPath();ctx.moveTo(pts[0].x,pts[0].y);for(let i=1;i<pts.length-1;i++){const mid={x:(pts[i].x+pts[i+1].x)/2,y:(pts[i].y+pts[i+1].y)/2};ctx.quadraticCurveTo(pts[i].x,pts[i].y,mid.x,mid.y)}ctx.lineTo(pts.at(-1)!.x,pts.at(-1)!.y);ctx.stroke()};
    if(wireframe)circulation.ribbons.forEach(ribbon=>draw(ribbon.center,ribbon.kind==="spine"?"rgba(202,255,247,.9)":"rgba(255,178,91,.92)",ribbon.kind==="spine"?1.25:1));
    ctx.fillStyle="#8aa9ae";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(`20′ × 20′ · ${heightInches/12}′ HIGH · ONE FLOOR · ${cells.length} RATIONALIZED FACES`,14,20);
  },[meshSlices,plate,circulation,camera,meshOpacity,wireframe,heightInches]);
  return <article className="spatial-chunk-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Box size={19}/></div><div><h2>Unified Ruled Mesh</h2><p>3D ruled floor plates → continuous quad mesh</p></div><span>DRAG · 3D MESH</span></header>
    <div className="chunk-viewport"><canvas ref={canvas} width={760} height={430} aria-label="Rotatable 3D preview of the unified ruled surface mesh" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.55,Math.min(2.2,value.zoom-event.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span><button onClick={()=>setCamera({yaw:-.68,pitch:.32,zoom:1})}>Reset view</button><label><span>Mesh opacity</span><input type="range" min="10" max="100" value={meshOpacity} onChange={event=>setMeshOpacity(Number(event.target.value))}/><strong>{meshOpacity}%</strong></label></div>
    <div className="spatial-constraint-band mesh-constraint"><strong>20′ × 20′</strong><span>{heightInches/12}′ adjustable height</span><span>1 occupiable floor</span><span>Trimmed envelope</span></div>
    <div className="geometry-mode"><span>Mesh display</span><div><button className={!wireframe?"active":""} onClick={()=>setWireframe(false)}>Shaded</button><button className={wireframe?"active":""} onClick={()=>setWireframe(true)}>Shaded + edges</button></div><small>Ruled floor surfaces preserve circulation height · no enclosure</small></div>
    <div className="chunk-summary"><span><strong>1</strong>floor plate</span><span><strong>{circulation.ribbons.length}</strong>circulation bands</span><span><strong>{plate.profiles.length}</strong>rail pairs</span><span><strong>{plate.cellCount}</strong>plate faces</span></div>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function ReadOnlyBoundaryLinesPreview({slices,activeIndex,fidelity,setFidelity,primaryCount,setPrimaryCount,offsetsPerPrimary,setOffsetsPerPrimary,bifurcationsPerPrimary,setBifurcationsPerPrimary,bifurcationDistance,setBifurcationDistance,bifurcationScale,setBifurcationScale,smoothing,setSmoothing,position,onDragStart}:{slices:Slice[];activeIndex:number;fidelity:number;setFidelity:(value:number)=>void;primaryCount:number;setPrimaryCount:(value:number)=>void;offsetsPerPrimary:number;setOffsetsPerPrimary:(value:number)=>void;bifurcationsPerPrimary:number;setBifurcationsPerPrimary:(value:number)=>void;bifurcationDistance:number;setBifurcationDistance:(value:number)=>void;bifurcationScale:number;setBifurcationScale:(value:number)=>void;smoothing:number;setSmoothing:(value:number)=>void;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
  const [camera,setCamera]=useState({yaw:-.68,pitch:.34,zoom:1}),[ruledSurfaces,setRuledSurfaces]=useState(true);
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
    if(ruledSurfaces)[...families.ruled,...families.branchPlates].forEach(strip=>{const count=Math.min(strip.a.length,strip.b.length),branch=families.branchPlates.includes(strip);for(let index=0;index<count-1;index++){const a=project(strip.a[index]),b=project(strip.b[index]),c=project(strip.b[index+1]),d=project(strip.a[index+1]);ctx.fillStyle=branch?"rgba(242,151,72,.3)":"rgba(95,231,218,.12)";ctx.strokeStyle=branch?"rgba(255,203,127,.7)":"rgba(95,231,218,.18)";ctx.lineWidth=branch ? .55 : .45;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.lineTo(c.x,c.y);ctx.lineTo(d.x,d.y);ctx.closePath();ctx.fill();ctx.stroke()}});
    const draw=(lines:Vec[][],color:string,width:number,glow=0)=>lines.forEach(line=>{const points=line.map(project);if(points.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.shadowColor=color;ctx.shadowBlur=glow;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);for(let i=1;i<points.length-1;i++){const midpoint={x:(points[i].x+points[i+1].x)/2,y:(points[i].y+points[i+1].y)/2};ctx.quadraticCurveTo(points[i].x,points[i].y,midpoint.x,midpoint.y)}ctx.lineTo(points.at(-1)!.x,points.at(-1)!.y);ctx.stroke();ctx.shadowBlur=0});
    draw(families.offset,"rgba(211,109,224,.54)",.85);draw(families.primary,"rgba(95,231,218,.92)",1.55,3);draw(families.branches,"rgba(255,184,97,.85)",1.15,2);
    families.anchors.forEach(({point})=>{const p=project(point);ctx.fillStyle="#d7fff8";ctx.beginPath();ctx.arc(p.x,p.y,1.35,0,Math.PI*2);ctx.fill()});
    ctx.fillStyle="#739396";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(`TRANSVERSE CAGE · SECTION ${activeIndex+1} / ${slices.length}`,14,20);ctx.textAlign="right";ctx.fillText(`${pointCount} EXTRACTED POINTS`,width-14,20);
  },[slices,loops,families,activeIndex,pointCount,camera,ruledSurfaces]);
  return <article className="boundary-lines-node" style={{left:position.x,top:position.y}}>
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
  const [camera,setCamera]=useState({yaw:-.68,pitch:.34,zoom:1,panX:0,panY:0});
  const [ruledSurfaces,setRuledSurfaces]=useState(true);
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
    if(ruledSurfaces)families.ruled.forEach(strip=>{const count=Math.min(strip.a.length,strip.b.length);for(let index=0;index<count-1;index++){const a=project(strip.a[index]),b=project(strip.b[index]),c=project(strip.b[index+1]),d=project(strip.a[index+1]);ctx.fillStyle=strip.kind==="void"?"rgba(255,181,91,.16)":"rgba(95,231,218,.12)";ctx.strokeStyle=strip.kind==="void"?"rgba(255,181,91,.24)":"rgba(95,231,218,.18)";ctx.lineWidth=.45;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.lineTo(c.x,c.y);ctx.lineTo(d.x,d.y);ctx.closePath();ctx.fill();ctx.stroke()}});
    if(ruledSurfaces)families.branchPlates.forEach(strip=>{const count=Math.min(strip.a.length,strip.b.length);for(let index=0;index<count-1;index++){const a=project(strip.a[index]),b=project(strip.b[index]),c=project(strip.b[index+1]),d=project(strip.a[index+1]);ctx.fillStyle="rgba(242,151,72,.3)";ctx.strokeStyle="rgba(255,203,127,.7)";ctx.lineWidth=.55;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.lineTo(c.x,c.y);ctx.lineTo(d.x,d.y);ctx.closePath();ctx.fill();ctx.stroke()}});
    const drawFamily=(lines:Vec[][],color:string,widthValue:number,glow=0)=>lines.forEach(line=>{
      const points=line.map(project);if(points.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=widthValue;ctx.shadowColor=color;ctx.shadowBlur=glow;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);
      for(let i=1;i<points.length-1;i++){const midpoint={x:(points[i].x+points[i+1].x)/2,y:(points[i].y+points[i+1].y)/2};ctx.quadraticCurveTo(points[i].x,points[i].y,midpoint.x,midpoint.y)}
      ctx.lineTo(points.at(-1)!.x,points.at(-1)!.y);ctx.stroke();ctx.shadowBlur=0;
    });
    drawFamily(families.tweens,"rgba(118,153,166,.25)",.62);drawFamily(families.offset,"rgba(211,109,224,.54)",.85);
    families.primary.forEach((line,index)=>{const role=families.primaryRoles[index],color=role.startsWith("ground")?"rgba(114,203,255,.95)":role.startsWith("void")||role.startsWith("threshold")?"rgba(255,181,91,.96)":role==="ridge"?"rgba(255,110,207,.96)":"rgba(95,231,218,.92)";drawFamily([line],color,1.55,3)});
    branchHitRef.current=families.branches.map((line,index)=>({index,points:line.map(project)}));
    families.branches.forEach((line,index)=>drawFamily([line],index===selectedBranch?"rgba(108,244,232,.98)":"rgba(255,184,97,.85)",index===selectedBranch?2.1:1.15,index===selectedBranch?5:2));
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
  return <article className={`boundary-lines-node ${expanded?"expanded":""}`} style={expanded?undefined:{left:position.x,top:position.y}} onDoubleClick={event=>{if(!(event.target as HTMLElement).closest("button,input"))setExpanded(true)}}>
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
        <p>{expanded?"Every orange control point is selectable. The red, green, and blue gumball arrows translate the active point along world X, Y, and Z; the adjacent curve and floor plate rebuild around that edited point. Ctrl+Z and Ctrl+Y move through the last ten point translations.":"Double-click this component to open the full-screen cage editor and translate individual curve points in a standard perspective viewport."}</p>
      </aside>
    </div>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function GridRationalizationPreview({slices,gridSize,alignment,setGridSize,setAlignment,position,onDragStart}:{slices:Slice[];gridSize:number;alignment:number;setGridSize:(value:number)=>void;setAlignment:(value:number)=>void;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
  const [camera,setCamera]=useState({yaw:-.72,pitch:.38,zoom:1});
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
    rendered.forEach(item=>{ctx.strokeStyle=item.index%4===0?"#aaf7eb":"rgba(112,205,197,.55)";ctx.lineWidth=item.index%4===0?1.5:.85;ctx.beginPath();item.points.forEach((point,index)=>{const p=project(point);index?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y)});ctx.stroke()});
    ctx.fillStyle="#83aeb2";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText(`GRID ${gridSize} CELLS · ALIGNMENT ${alignment}%`,14,20);
  },[slices,loops,gridSize,alignment,camera]);
  return <article className="grid-rationalization-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Grid3X3 size={19}/></div><div><h2>Grid Rationalization</h2><p>CT volume → ordered section framework</p></div><span>DRAG · GRID</span></header>
    <div className="grid-viewport"><canvas ref={canvas} width={620} height={350} aria-label="Rotatable 3D preview of grid-aligned section curves" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(event.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.55,Math.min(2.2,value.zoom-event.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span><button onClick={()=>setCamera({yaw:-.72,pitch:.38,zoom:1})}>Reset view</button></div>
    <div className="grid-controls"><label><span>Grid spacing</span><strong>{gridSize} cells</strong><input type="range" min="4" max="16" step="1" value={gridSize} onChange={event=>setGridSize(Number(event.target.value))}/><small>Fine order</small><small>Broad modules</small></label><label><span>Grid alignment</span><strong>{alignment}%</strong><input type="range" min="0" max="100" step="2" value={alignment} onChange={event=>setAlignment(Number(event.target.value))}/><small>Original curves</small><small>Orthogonal order</small></label></div>
    <p>Exterior and void boundaries are pulled toward one shared grid, then relaxed so adjacent sections blend continuously. This rationalized geometry now feeds the Unified Section Cage before its primary and offset lines generate the ruled mesh.</p>
    <span className="port port-left port-ct-in" aria-hidden="true"/><span className="port port-right port-cage-out" aria-hidden="true"/>
  </article>;
}

function RealityNode({slices,settings,setSettings,position,onDragStart}:{slices:Slice[];settings:RealitySettings;setSettings:React.Dispatch<React.SetStateAction<RealitySettings>>;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
  const [camera,setCamera]=useState({yaw:-.62,pitch:.34,zoom:1});
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;const width=c.width,height=c.height;
    const bg=ctx.createLinearGradient(0,0,0,height);bg.addColorStop(0,"#172022");bg.addColorStop(1,"#090e10");ctx.fillStyle=bg;ctx.fillRect(0,0,width,height);
    if(!slices.length){ctx.fillStyle="#88a2a3";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select sections to run reality checks",width/2,height/2);return}
    const zCenter=(slices[0].z+slices.at(-1)!.z)/2,rotate=([x,y,z]:Vec):Vec=>{z-=zCenter;const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]};
    const project=(p:Vec)=>{const [x,y,z]=rotate(p),perspective=5/(5+z),scale=145*camera.zoom*perspective;return{x:width*.5+x*scale,y:height*.54-y*scale,z}};
    // Ground datum and one continuous occupiable floor.
    const ground:[Vec,Vec,Vec,Vec]=[[-2.4,-.77,zCenter-2.4],[2.4,-.77,zCenter-2.4],[2.4,-.77,zCenter+2.4],[-2.4,-.77,zCenter+2.4]];
    ctx.fillStyle="rgba(92,126,108,.16)";ctx.strokeStyle="rgba(137,190,160,.55)";ctx.lineWidth=1.1;ctx.beginPath();ground.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y)});ctx.closePath();ctx.fill();ctx.stroke();
    const floorY=-.73,a=project([-1.18,floorY,zCenter-1.65]),b=project([1.18,floorY,zCenter-1.65]),d=project([1.18,floorY,zCenter+1.65]),e=project([-1.18,floorY,zCenter+1.65]);ctx.fillStyle="rgba(113,170,160,.13)";ctx.strokeStyle="rgba(130,221,204,.78)";ctx.lineWidth=1.5;ctx.beginPath();[a,b,d,e].forEach((q,i)=>i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y));ctx.closePath();ctx.fill();ctx.stroke();
    const rendered=slices.flatMap((slice,index)=>sectionContourLoops(slice.field).map(loop=>({index,points:contourFidelity(loop,58).map(([x,y])=>[(x/(W-1)-.5)*2.25,(.5-y/(H-1))*1.7,slice.z] as Vec)}))).map(item=>({...item,depth:item.points.reduce((sum,p)=>sum+rotate(p)[2],0)/Math.max(1,item.points.length)})).sort((a,b)=>b.depth-a.depth);
    rendered.forEach(item=>{ctx.strokeStyle="rgba(175,220,214,.28)";ctx.lineWidth=.8;ctx.beginPath();item.points.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y)});ctx.stroke()});
    // Supports read as a structural frame tied directly to the ground plane.
    const span=Math.max(.35,settings.supportSpacing/28);ctx.strokeStyle="#e3b96f";ctx.lineWidth=1.5;for(let x=-1.05;x<=1.06;x+=span){const a=project([x,-.77,zCenter-1.25]),b=project([x,.42,zCenter-1.25]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
    // A continuous circulation trace passes through the threshold sequence.
    ctx.strokeStyle="#ff785e";ctx.lineWidth=2.4;ctx.shadowColor="#ff785e";ctx.shadowBlur=6;ctx.beginPath();slices.forEach((slice,index)=>{const q=project([0,-.48,slice.z]);index?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y)});ctx.stroke();ctx.shadowBlur=0;
    ctx.fillStyle="#8ca9aa";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText("GROUND DATUM · SINGLE FLOOR · SUPPORTS · FLOW",14,20);
  },[slices,settings,camera]);
  const update=(key:keyof RealitySettings,value:number)=>setSettings(current=>({...current,[key]:value}));
  return <article className="reality-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Box size={19}/></div><div><h2>Reality Constraints</h2><p>Grounded, supportable single-floor space</p></div><span>DRAG · VALIDATE</span></header>
    <div className="reality-viewport"><canvas ref={canvas} width={560} height={330} aria-label="Rotatable preview of grounded structure, one floor and circulation" onPointerDown={event=>{event.currentTarget.setPointerCapture(event.pointerId);orbit.current={x:event.clientX,y:event.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={event=>{const start=orbit.current;if(start)setCamera(value=>({...value,yaw:start.yaw+(event.clientX-start.x)*.01,pitch:Math.max(-1.2,Math.min(1.2,start.pitch+(event.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={event=>{event.preventDefault();setCamera(value=>({...value,zoom:Math.max(.6,Math.min(2.1,value.zoom-event.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span><button onClick={()=>setCamera({yaw:-.62,pitch:.34,zoom:1})}>Reset view</button></div>
    <div className="reality-groups">
      <section><strong>Structural rationalization</strong><p>Ground-bearing piers resolve unsupported areas.</p><label>Maximum support spacing <b>{settings.supportSpacing} ft</b><input type="range" min="12" max="32" step="2" value={settings.supportSpacing} onChange={e=>update("supportSpacing",Number(e.target.value))}/></label></section>
      <section><strong>Occupancy + circulation</strong><p>Detected voids become continuous human thresholds.</p><label>Minimum clear opening <b>{settings.clearance} ft</b><input type="range" min="4" max="12" step="1" value={settings.clearance} onChange={e=>update("clearance",Number(e.target.value))}/></label></section>
      <section><strong>Site conditions</strong><p>One continuous floor follows the ground; grade changes stay walkable.</p><label>Maximum slope <b>1:{settings.maxSlope}</b><input type="range" min="12" max="24" step="1" value={settings.maxSlope} onChange={e=>update("maxSlope",Number(e.target.value))}/></label></section>
    </div>
    <div className="reality-status"><span>GROUND CONTACT</span><span>STRUCTURAL SUPPORT</span><span>SINGLE OCCUPIABLE FLOOR</span><span>CONNECTED FLOW</span></div>
    <span className="port port-left" aria-hidden="true"/>
  </article>;
}

function ExportGeometryPreview({slices,exportHeight,position,onDragStart}:{slices:Slice[];exportHeight:number;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null);
  const orbit=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
  const [camera,setCamera]=useState({yaw:-.72,pitch:.38,zoom:1});
  const [layers,setLayers]=useState({mesh:true,sections:true,boundaries:true});
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;
    const width=c.width,height=c.height;ctx.clearRect(0,0,width,height);
    const background=ctx.createLinearGradient(0,0,0,height);background.addColorStop(0,"#565c60");background.addColorStop(.55,"#343a3e");background.addColorStop(1,"#202529");ctx.fillStyle=background;ctx.fillRect(0,0,width,height);
    if(!slices.length){ctx.fillStyle="#8fa4a6";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select depth sections to preview the Rhino export",width/2,height/2);return}
    const nx=44,ny=34,nz=slices.length,threshold=.32,zCenter=exportHeight*.5;
    const filled=Array.from({length:nz},(_,z)=>Array.from({length:ny},(_,y)=>Array.from({length:nx},(_,x)=>slices[z].field[Math.round((y+.5)*(H-1)/ny)*W+Math.round((x+.5)*(W-1)/nx)]>=threshold)));
    const centers=Array.from({length:nz},(_,index)=>exportDepth(slices,index,exportHeight)),zBounds=Array.from({length:nz+1},(_,index)=>nz===1?index*exportHeight:index===0?0:index===nz?exportHeight:(centers[index-1]+centers[index])*.5),world=(x:number,y:number,z:number):Vec=>[(x/nx-.5)*EXPORT_TILE_INCHES,(.5-y/ny)*EXPORT_TILE_INCHES,zBounds[z]];
    const rotate=([x,y,z]:Vec):Vec=>{z-=zCenter;const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch),rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]};
    const extent=Math.max(EXPORT_TILE_INCHES,exportHeight),project=(point:Vec)=>{const [x,y,z]=rotate(point),distance=extent*2.5,perspective=distance/(distance+z),scale=Math.min(width,height)*.72/extent*camera.zoom*perspective;return{x:width*.5+x*scale,y:height*.52-y*scale,z}};
    ctx.strokeStyle="rgba(196,205,209,.18)";ctx.lineWidth=.7;for(let i=0;i<=10;i++){const v=-10+i*2,a=project([v,-10,0]),b=project([v,10,0]),d=project([-10,v,0]),e=project([10,v,0]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.beginPath();ctx.moveTo(d.x,d.y);ctx.lineTo(e.x,e.y);ctx.stroke()}
    type Face={points:Vec[];tone:"outer"|"void"|"cap"};const faces:Face[]=[],isFilled=(x:number,y:number,z:number)=>z>=0&&z<nz&&y>=0&&y<ny&&x>=0&&x<nx&&filled[z][y][x],push=(points:[number,number,number][],tone:Face["tone"])=>faces.push({points:points.map(([x,y,z])=>world(x,y,z)),tone});
    for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){if(!filled[z][y][x])continue;
      if(!isFilled(x-1,y,z))push([[x,y,z],[x,y+1,z],[x,y+1,z+1],[x,y,z+1]],x===0?"outer":"void");
      if(!isFilled(x+1,y,z))push([[x+1,y,z],[x+1,y,z+1],[x+1,y+1,z+1],[x+1,y+1,z]],x===nx-1?"outer":"void");
      if(!isFilled(x,y-1,z))push([[x,y,z],[x,y,z+1],[x+1,y,z+1],[x+1,y,z]],y===0?"outer":"void");
      if(!isFilled(x,y+1,z))push([[x,y+1,z],[x+1,y+1,z],[x+1,y+1,z+1],[x,y+1,z+1]],y===ny-1?"outer":"void");
      if(!isFilled(x,y,z-1))push([[x,y,z],[x+1,y,z],[x+1,y+1,z],[x,y+1,z]],"cap");
      if(!isFilled(x,y,z+1))push([[x,y,z+1],[x,y+1,z+1],[x+1,y+1,z+1],[x+1,y,z+1]],"cap");
    }
    const rendered=faces.map(face=>({face,rotated:face.points.map(rotate),screen:face.points.map(project)})).sort((a,b)=>b.rotated.reduce((n,p)=>n+p[2],0)-a.rotated.reduce((n,p)=>n+p[2],0));
    if(layers.mesh)rendered.forEach(item=>{const a=item.rotated[0],b=item.rotated[1],d=item.rotated[3],ux=b[0]-a[0],uy=b[1]-a[1],uz=b[2]-a[2],vx=d[0]-a[0],vy=d[1]-a[1],vz=d[2]-a[2],nxv=uy*vz-uz*vy,nyv=uz*vx-ux*vz,nzv=ux*vy-uy*vx,len=Math.hypot(nxv,nyv,nzv)||1,light=Math.max(.08,Math.abs(nxv*-.35+nyv*.82+nzv*-.45)/len),shade=Math.round(76+light*150);ctx.globalAlpha=.78;ctx.fillStyle=item.face.tone==="void"?`rgb(${20+Math.round(light*22)},${31+Math.round(light*30)},${35+Math.round(light*34)})`:item.face.tone==="cap"?`rgb(${Math.round(shade*.68)},${Math.round(shade*.74)},${Math.round(shade*.76)})`:`rgb(${Math.min(226,shade)},${Math.min(234,shade+7)},${Math.min(236,shade+10)})`;ctx.beginPath();ctx.moveTo(item.screen[0].x,item.screen[0].y);for(let i=1;i<4;i++)ctx.lineTo(item.screen[i].x,item.screen[i].y);ctx.closePath();ctx.fill()});ctx.globalAlpha=1;
    // Layer 02: every exported planar section surface, shown as a translucent
    // baked object at its exact Rhino depth.
    let sectionFaces=0;
    if(layers.sections){ctx.fillStyle="rgba(242,105,82,.055)";ctx.strokeStyle="rgba(255,139,105,.14)";ctx.lineWidth=.32;for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){if(!filled[z][y][x])continue;sectionFaces++;const depth=centers[z],points:[Vec,Vec,Vec,Vec]=[[(x/nx-.5)*EXPORT_TILE_INCHES,(.5-y/ny)*EXPORT_TILE_INCHES,depth],[((x+1)/nx-.5)*EXPORT_TILE_INCHES,(.5-y/ny)*EXPORT_TILE_INCHES,depth],[((x+1)/nx-.5)*EXPORT_TILE_INCHES,(.5-(y+1)/ny)*EXPORT_TILE_INCHES,depth],[(x/nx-.5)*EXPORT_TILE_INCHES,(.5-(y+1)/ny)*EXPORT_TILE_INCHES,depth]],screen=points.map(project);ctx.beginPath();ctx.moveTo(screen[0].x,screen[0].y);for(let i=1;i<4;i++)ctx.lineTo(screen[i].x,screen[i].y);ctx.closePath();ctx.fill();ctx.stroke()}}
    // Layer 03: all joined exterior and internal-void boundaries are drawn as
    // baked Rhino curves over the composite surface set.
    let boundaryCount=0;
    if(layers.boundaries){ctx.strokeStyle="rgba(109,244,226,.72)";ctx.lineWidth=.72;slices.forEach((slice,index)=>sectionContourLoops(slice.field).map(loop=>contourFidelity(loop,76)).forEach(loop=>{boundaryCount++;ctx.beginPath();loop.forEach(([x,y],pointIndex)=>{const point=project([(x/(W-1)-.5)*EXPORT_TILE_INCHES,(.5-y/(H-1))*EXPORT_TILE_INCHES,centers[index]]);pointIndex?ctx.lineTo(point.x,point.y):ctx.moveTo(point.x,point.y)});ctx.stroke()}))}
    ctx.strokeStyle="#8bd9cf";ctx.lineWidth=1.1;const tile=[project([-10,-10,0]),project([10,-10,0]),project([10,10,0]),project([-10,10,0])];ctx.beginPath();tile.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.stroke();
    ctx.fillStyle="#d8e0e2";ctx.font="11px sans-serif";ctx.textAlign="left";ctx.fillText(`${faces.length} mesh faces · ${sectionFaces} section cells · ${boundaryCount} curves`,16,height-16);
  },[slices,camera,exportHeight,layers]);
  const toggleLayer=(key:keyof typeof layers)=>setLayers(current=>({...current,[key]:!current[key]}));
  return <article className="subd-preview-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Sparkles size={19}/></div><div><h2>Baked Rhino Geometry Preview</h2><p>All exported layers · combined Rhino viewport</p></div><span>DRAG · LIVE</span></header>
    <div className="subd-viewport"><canvas ref={canvas} width={500} height={360} aria-label="Interactive preview showing the watertight mesh, all section surfaces, and all boundary curves baked together in Rhino. Drag to orbit and use the mouse wheel to zoom." onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);orbit.current={x:e.clientX,y:e.clientY,yaw:camera.yaw,pitch:camera.pitch}}} onPointerMove={e=>{const start=orbit.current;if(start)setCamera(v=>({...v,yaw:start.yaw+(e.clientX-start.x)*.01,pitch:Math.max(-1.25,Math.min(1.25,start.pitch+(e.clientY-start.y)*.008))}))}} onPointerUp={()=>{orbit.current=null}} onPointerCancel={()=>{orbit.current=null}} onWheel={e=>{e.preventDefault();setCamera(v=>({...v,zoom:Math.max(.55,Math.min(2.2,v.zoom-e.deltaY*.0012))}))}}/><span>Drag to orbit · wheel to zoom</span><button onClick={()=>setCamera({yaw:-.72,pitch:.38,zoom:1})}>Reset view</button></div>
    <div className="spatial-constraint-band compact"><strong>20′ × 20′</strong><span>{exportHeight/12}′ high</span><span>1 floor</span><span>Trimmed</span></div>
    <div className="baked-layer-controls"><button className={layers.mesh?"active":""} onClick={()=>toggleLayer("mesh")} aria-pressed={layers.mesh}><i className="mesh"/>Printable SubD mesh</button><button className={layers.sections?"active":""} onClick={()=>toggleLayer("sections")} aria-pressed={layers.sections}><i className="sections"/>Section surfaces</button><button className={layers.boundaries?"active":""} onClick={()=>toggleLayer("boundaries")} aria-pressed={layers.boundaries}><i className="boundaries"/>Boundary curves</button></div>
    <div className="subd-preview-meta"><span>RHINO LAYERS 01–03 · ALL GEOMETRY BAKED</span><strong>Composite export preview</strong><p>The single-floor ruled form, reference sections, and boundary curves are displayed inside the 20′ × 20′ × {exportHeight/12}′ envelope; geometry beyond it is removed.</p></div>
    <span className="port port-left" aria-hidden="true"/><span className="port port-right" aria-hidden="true"/>
  </article>;
}

function FrontClipPreview({slices,heightInches,position,onDragStart}:{slices:Slice[];heightInches:number;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const canvas=useRef<HTMLCanvasElement>(null);
  const [clip,setClip]=useState(50);
  const clipIndex=Math.round(clip/100*Math.max(0,slices.length-1));
  useEffect(()=>{
    const c=canvas.current,ctx=c?.getContext("2d");if(!c||!ctx)return;
    const width=c.width,height=c.height;ctx.clearRect(0,0,width,height);
    const bg=ctx.createLinearGradient(0,0,0,height);bg.addColorStop(0,"#4b5155");bg.addColorStop(1,"#22272a");ctx.fillStyle=bg;ctx.fillRect(0,0,width,height);
    ctx.strokeStyle="rgba(205,215,218,.14)";ctx.lineWidth=1;
    for(let x=18;x<width;x+=24){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,height);ctx.stroke()}
    for(let y=18;y<height;y+=24){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(width,y);ctx.stroke()}
    if(!slices.length){ctx.fillStyle="#a7b3b6";ctx.font="14px sans-serif";ctx.textAlign="center";ctx.fillText("Select depth sections to clip",width/2,height/2);return}
    const pad=30,viewW=width-90,viewH=height-58,scale=Math.min(viewW/W,viewH/H),ox=pad+(viewW-W*scale)/2,oy=24+(viewH-H*scale)/2;
    // The accumulated front projection stays visible behind the active clipping cut.
    ctx.save();ctx.globalAlpha=.055;ctx.globalCompositeOperation="lighter";
    for(const slice of slices)ctx.drawImage(slice.image,ox,oy,W*scale,H*scale);
    ctx.restore();
    const active=slices[clipIndex];
    ctx.save();ctx.globalAlpha=.2;ctx.drawImage(active.image,ox,oy,W*scale,H*scale);ctx.restore();
    ctx.fillStyle="rgba(222,232,233,.72)";
    for(let y=0;y<H;y++)for(let x=0;x<W;x++)if(active.field[y*W+x]>=.32)ctx.fillRect(ox+x*scale,oy+y*scale,scale+.35,scale+.35);
    ctx.save();ctx.setTransform(scale,0,0,scale,ox,oy);ctx.strokeStyle="#e4eeee";ctx.lineWidth=.28;contours(ctx,active.field);ctx.strokeStyle="#ff6b52";ctx.lineWidth=.34;drawVoidPerimeters(ctx,active.field);ctx.restore();
    ctx.strokeStyle="#8bd9cf";ctx.lineWidth=1.25;ctx.strokeRect(ox+.5,oy+.5,W*scale,H*scale);
    // Rhino-like side depth rail showing the clipping plane position through the loft.
    const railX=width-37,railTop=31,railBottom=height-30,planeY=railTop+(railBottom-railTop)*clip/100;
    ctx.fillStyle="rgba(10,16,18,.52)";ctx.fillRect(railX-10,railTop,20,railBottom-railTop);
    ctx.strokeStyle="#718388";ctx.strokeRect(railX-10+.5,railTop+.5,19,railBottom-railTop-1);
    ctx.fillStyle="#8fa1a5";for(let i=0;i<7;i++)ctx.fillRect(railX-4,railTop+i*(railBottom-railTop)/6,8,1);
    ctx.shadowColor="#7cf0df";ctx.shadowBlur=9;ctx.strokeStyle="#8ff4e5";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(railX-15,planeY);ctx.lineTo(railX+15,planeY);ctx.stroke();ctx.shadowBlur=0;
    ctx.fillStyle="#dbe6e8";ctx.font="10px sans-serif";ctx.textAlign="left";ctx.fillText("FRONT",12,17);ctx.fillStyle="#91acae";ctx.fillText(`CUT ${clipIndex+1} / ${slices.length}`,12,height-12);
  },[slices,clip,clipIndex]);
  return <article className="clip-preview-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><ScanLine size={18}/></div><div><h2>SubD Section Cut</h2><p>Front orthographic · section from smoothed massing</p></div><span>DRAG · CLIP</span></header>
    <canvas ref={canvas} width={430} height={300} aria-label="Front orthographic view with movable clipping plane"/>
    <div className="spatial-constraint-band compact"><strong>20′ × 20′</strong><span>{heightInches/12}′ high</span><span>1 floor</span><span>Trimmed</span></div>
    <div className="clip-control"><label><span>Clipping plane</span><strong>{clip}%</strong><input type="range" min="0" max="100" step="1" value={clip} onChange={e=>setClip(Number(e.target.value))}/><small>Front section</small><small>Back section</small></label></div>
    <p className="clip-note">The filled cut is sampled directly from the smoothed SubD field. Move the plane through the massing; orange lines retain its internal void boundaries.</p>
    <span className="port port-left" aria-hidden="true"/>
  </article>;
}

function ExportNode({slices,lineFidelity,primaryCount,offsetsPerPrimary,exportHeight,setExportHeight,position,onDragStart}:{slices:Slice[];lineFidelity:number;primaryCount:number;offsetsPerPrimary:number;exportHeight:number;setExportHeight:(value:number)=>void;position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
  const [version,setVersion]=useState<7|8>(8),[busy,setBusy]=useState(false);
  const run=async()=>{setBusy(true);try{await exportRhinoBundle(slices,version,lineFidelity,exportHeight,primaryCount,offsetsPerPrimary)}finally{setBusy(false)}};
  return <article className="export-node" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><Download size={19}/></div><div><h2>Rhino Export</h2><p>One coordinated file · three editable geometry layers</p></div><span>DRAG · 3DM</span></header>
    <div className="rhino-version"><span>File version</span><div role="group" aria-label="Rhino file version"><button className={version===7?"active":""} onClick={()=>setVersion(7)}>Rhino 7</button><button className={version===8?"active":""} onClick={()=>setVersion(8)}>Rhino 8</button></div><small>Imperial model units: inches · layout units: feet.</small></div>
    <div className="export-scale"><div><span>Maximum footprint</span><strong>20′ × 20′</strong><small>240″ × 240″ architectural envelope</small></div><label><span>Overall height</span><strong>{exportHeight/12}′ <small>({exportHeight}″)</small></strong><input type="range" min="96" max="240" step="12" value={exportHeight} onChange={event=>setExportHeight(Number(event.target.value))}/><small>8′</small><small>20′</small></label></div>
    <div className="export-options export-bundle">
      <section><Box size={22}/><div><strong>01 · Unified Printable Form</strong><p>One single-floor watertight mesh inside the 20′ × 20′ envelope, organized by {primaryCount} primary lines and {offsetsPerPrimary} offsets per primary.</p></div></section>
      <section><ScanLine size={22}/><div><strong>02 · Section Surfaces</strong><p>Every CT section remains an independent named surface on its own Rhino layer.</p></div></section>
      <section><ScanLine size={22}/><div><strong>03 · Boundary Lines</strong><p>Editable exterior and void curves exported at {lineFidelity}% fidelity.</p></div></section>
    </div>
    <div className="export-secondary export-all"><span>{slices.length} sections · one floor · max 20′ × 20′ × {exportHeight/12}′ · outside geometry trimmed</span><button onClick={()=>void run()} disabled={!slices.length||busy}><Download size={14}/>{busy?"Building complete package…":`Export all 3 · R${version}`}</button></div>
    <span className="port port-left port-mesh-in" aria-hidden="true"/><span className="port port-right port-preview-out" aria-hidden="true"/>
  </article>;
}

export default function PhotoModel({images,position,onDragStart}:{images:StoredImage[];position:{x:number;y:number};onDragStart:(e:React.PointerEvent)=>void}){
  const [selected,setSelected]=useState<string[]>([]);
  const [fields,setFields]=useState<Float32Array[]>([]),[busy,setBusy]=useState(true);
  const [interval,setInterval]=useState(.68),[elongation,setElongation]=useState(165),[continuity,setContinuity]=useState(72),[scan,setScan]=useState(50),[density,setDensity]=useState(42),[depthGain,setDepthGain]=useState(68),[voidTransform,setVoidTransform]=useState(0);
  const [angle,setAngle]=useState(.62),[tilt,setTilt]=useState(.3),[lineFidelity,setLineFidelity]=useState(76),[primaryLineCount,setPrimaryLineCount]=useState(5),[offsetsPerPrimary,setOffsetsPerPrimary]=useState(2),[bifurcationsPerPrimary,setBifurcationsPerPrimary]=useState(3),[bifurcationDistance,setBifurcationDistance]=useState(65),[bifurcationScale,setBifurcationScale]=useState(60),[primarySmoothing,setPrimarySmoothing]=useState(72),[gridSize,setGridSize]=useState(8),[gridAlignment,setGridAlignment]=useState(64),[exportHeight,setExportHeight]=useState(144);
  const [circulationSettings,setCirculationSettings]=useState<CirculationSettings>({primaryCount:2,pathWidth:5.5,activeBranches:2,junctionWidth:55,smoothing:70}),[floorPlateSettings,setFloorPlateSettings]=useState<FloorPlateSettings>({noiseReduction:72,plateThickness:10,surfaceFlow:82});
  const [realitySettings,setRealitySettings]=useState<RealitySettings>({supportSpacing:24,clearance:6,maxSlope:12});
  const canvas=useRef<HTMLCanvasElement>(null),orbit=useRef<{x:number;y:number;a:number;t:number}|null>(null);
  type ChildNode="reality"|"preview"|"clip"|"export"|"boundary"|"grid"|"circulation"|"framework"|"floor"|"mesh";
  const [nodePositions,setNodePositions]=useState<Record<ChildNode,{x:number;y:number}>>(()=>({grid:{x:position.x+760,y:position.y},boundary:{x:position.x+1520,y:position.y},circulation:{x:position.x+2280,y:position.y},framework:{x:position.x+3060,y:position.y},floor:{x:position.x+3820,y:position.y},mesh:{x:position.x+4660,y:position.y},export:{x:position.x+5500,y:position.y},preview:{x:position.x+6500,y:position.y},clip:{x:position.x+7080,y:position.y},reality:{x:position.x+760,y:position.y+900}}));
  const nodeDrag=useRef<{node:ChildNode;startX:number;startY:number;origin:{x:number;y:number}}|null>(null);
  useEffect(()=>{
    const move=(event:PointerEvent)=>{const active=nodeDrag.current;if(!active)return;setNodePositions(current=>({...current,[active.node]:{x:Math.max(12,active.origin.x+event.clientX-active.startX),y:Math.max(20,active.origin.y+event.clientY-active.startY)}}))};
    const up=()=>{nodeDrag.current=null};
    window.addEventListener("pointermove",move);window.addEventListener("pointerup",up);window.addEventListener("pointercancel",up);
    return()=>{window.removeEventListener("pointermove",move);window.removeEventListener("pointerup",up);window.removeEventListener("pointercancel",up)};
  },[]);
  const startChildDrag=(node:ChildNode)=>(event:React.PointerEvent)=>{if((event.target as HTMLElement).closest("button,input,canvas"))return;event.preventDefault();nodeDrag.current={node,startX:event.clientX,startY:event.clientY,origin:nodePositions[node]}};
  useEffect(()=>{setSelected(current=>{const available=new Set(images.map(image=>image.key));return current.filter(key=>available.has(key))})},[images]);
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
  const groundedSlices=useMemo(()=>rationalizeSlices(slices,realitySettings).map(slice=>({...slice,field:keepLargestConnectedField(slice.field)})),[slices,realitySettings]);
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
      if(i===scanIndex){ctx.globalAlpha=1;ctx.strokeStyle="#ff6b52";ctx.lineWidth=.18;drawVoidPerimeters(ctx,slice.field)}
      if(i===scanIndex){ctx.strokeStyle="#81d9cd";ctx.lineWidth=.14;ctx.strokeRect(0,0,W,H)}ctx.restore();
    }
    const active=slices[scanIndex],bx=535,by=85,bw=165,bh=165*H/W;
    ctx.fillStyle="#10191c";ctx.fillRect(520,48,190,240);ctx.strokeStyle="#55747a";ctx.lineWidth=1;ctx.strokeRect(520.5,48.5,189,239);
    ctx.fillStyle="#9fc6c7";ctx.font="12px sans-serif";ctx.textAlign="left";ctx.fillText("ACTIVE SECTION",bx,72);
    ctx.globalAlpha=1;ctx.drawImage(active.image,bx,by,bw,bh);ctx.save();ctx.setTransform(bw/W,0,0,bh/H,bx,by);ctx.strokeStyle="#ff6b52";ctx.lineWidth=.22;drawVoidPerimeters(ctx,active.field);ctx.restore();ctx.strokeStyle="#8cd5cb";ctx.strokeRect(bx+.5,by+.5,bw,bh);
    ctx.strokeStyle="#659494";ctx.beginPath();ctx.moveTo(bx+bw/2,by);ctx.lineTo(bx+bw/2,by+bh);ctx.moveTo(bx,by+bh/2);ctx.lineTo(bx+bw,by+bh/2);ctx.stroke();
    ctx.fillStyle="#a7bec0";ctx.font="12px sans-serif";ctx.fillText(`${scan}% THROUGH VOLUME`,bx,by+bh+23);
  },[slices,scanIndex,scan,density,angle,tilt,busy]);
  const toggle=(key:string)=>setSelected(current=>current.includes(key)?current.filter(item=>item!==key):[...current,key]);
  const deselectSelection=()=>setSelected([]);
  const realityPosition=nodePositions.reality;
  const previewPosition=nodePositions.preview;
  const clipPosition=nodePositions.clip;
  const exportPosition=nodePositions.export;
  const boundaryPosition=nodePositions.boundary;
  const gridPosition=nodePositions.grid;
  const circulationPosition=nodePositions.circulation;
  const frameworkPosition=nodePositions.framework;
  const floorPosition=nodePositions.floor;
  const meshPosition=nodePositions.mesh;
  const horizontalWire=(start:{x:number;y:number},end:{x:number;y:number})=>`M ${start.x} ${start.y} C ${start.x+(end.x-start.x)*.48} ${start.y}, ${end.x-(end.x-start.x)*.48} ${end.y}, ${end.x} ${end.y}`;
  const verticalWire=(start:{x:number;y:number},end:{x:number;y:number})=>`M ${start.x} ${start.y} C ${start.x} ${start.y+(end.y-start.y)*.48}, ${end.x} ${end.y-(end.y-start.y)*.48}, ${end.x} ${end.y}`;
  return <>
    <svg className="workflow-wires" aria-hidden="true">
      <path d={horizontalWire({x:position.x+620,y:position.y+286},{x:gridPosition.x,y:gridPosition.y+286})}/>
      <path d={horizontalWire({x:gridPosition.x+620,y:gridPosition.y+286},{x:boundaryPosition.x,y:boundaryPosition.y+286})}/>
      <path d={horizontalWire({x:boundaryPosition.x+620,y:boundaryPosition.y+286},{x:circulationPosition.x,y:circulationPosition.y+286})}/>
      <path d={horizontalWire({x:circulationPosition.x+700,y:circulationPosition.y+286},{x:frameworkPosition.x,y:frameworkPosition.y+286})}/>
      <path d={horizontalWire({x:frameworkPosition.x+700,y:frameworkPosition.y+286},{x:floorPosition.x,y:floorPosition.y+286})}/>
      <path d={horizontalWire({x:floorPosition.x+760,y:floorPosition.y+286},{x:meshPosition.x,y:meshPosition.y+286})}/>
      <path d={horizontalWire({x:meshPosition.x+760,y:meshPosition.y+286},{x:exportPosition.x,y:exportPosition.y+286})}/>
      <path d={horizontalWire({x:exportPosition.x+980,y:exportPosition.y+286},{x:previewPosition.x,y:previewPosition.y+287})}/>
      <path d={horizontalWire({x:previewPosition.x+500,y:previewPosition.y+287},{x:clipPosition.x,y:clipPosition.y+285})}/>
    </svg>
    <article className="model-node ct-node" data-node="model" style={{left:position.x,top:position.y}}>
    <header onPointerDown={onDragStart}><div className="model-icon"><ScanLine size={21}/></div><div><h2>Depth Maps → CT Volume</h2><p>Cubic depth blending · continuous massing</p></div><span>DRAG</span></header>
    <div className="model-stage"><canvas ref={canvas} width={720} height={380} aria-label="Rotatable CT style volume with highlighted scan section" onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);orbit.current={x:e.clientX,y:e.clientY,a:angle,t:tilt}}} onPointerMove={e=>{const start=orbit.current;if(start){setAngle(start.a+(e.clientX-start.x)*.009);setTilt(Math.max(-1.2,Math.min(1.2,start.t+(e.clientY-start.y)*.006)))}}} onPointerUp={()=>{orbit.current=null}}/><span>Drag to orbit</span></div>
    <div className="ct-controls"><label>Scan position <strong>{scan}%</strong><input type="range" min="0" max="100" value={scan} onChange={e=>setScan(Number(e.target.value))}/></label><label>Volume density <strong>{density}%</strong><input type="range" min="10" max="100" value={density} onChange={e=>setDensity(Number(e.target.value))}/></label><label className="depth-gain">Depth separation <strong>{depthGain}%</strong><input type="range" min="10" max="100" value={depthGain} onChange={e=>setDepthGain(Number(e.target.value))}/></label><label className="massing-shape">Massing elongation <strong>{elongation}%</strong><input type="range" min="75" max="350" step="5" value={elongation} onChange={e=>setElongation(Number(e.target.value))}/><span className="range-ends"><i>Compact</i><i>Elongated</i></span></label><label className="massing-shape">Surface continuity <strong>{continuity}%</strong><input type="range" min="0" max="100" step="2" value={continuity} onChange={e=>setContinuity(Number(e.target.value))}/><span className="range-ends"><i>Sectional</i><i>Smooth + curved</i></span></label><label className="void-transform">Void / open-area transform <strong>{voidTransform===0?"Unchanged":voidTransform>0?`Expand +${voidTransform}%`:`Compress ${Math.abs(voidTransform)}%`}</strong><input type="range" min="-100" max="100" step="5" value={voidTransform} onChange={e=>setVoidTransform(Number(e.target.value))}/><span className="range-ends"><i>Compress openings</i><i>Expand openings</i></span></label></div>
    <div className="model-controls"><label>Slice interval <input type="range" min=".4" max="1.2" step=".02" value={interval} onChange={e=>setInterval(Number(e.target.value))}/></label><span className="export-routed">Exports routed to the connected Rhino component</span></div>
    <div className="model-sources"><div className="model-caption"><strong>Imported depth sequence</strong><div><span>{selected.length} of {images.length} slices</span><button className="reset-selection" onClick={deselectSelection} disabled={!selected.length} title="Deselect every image from the CT sequence"><XCircle size={13}/>Deselect selection</button></div></div><div className="model-thumbs">{images.map(image=><button key={image.key} className={selected.includes(image.key)?"active":""} onClick={()=>toggle(image.key)} aria-pressed={selected.includes(image.key)} title={image.name}><img src={source(image.key)} alt=""/><span>{image.name}</span></button>)}</div>{!images.length&&<div className="depth-empty">Import images in Image Storage to build the CT sequence.</div>}</div>
    <p className="model-note">Each photo is read as a depth map with perimeter-aware cavity detection. The CT result is sent horizontally to Grid Rationalization, establishing the ordered framework that the Unified Section Cage uses to generate primary, offset, and bifurcating lines.</p><span className="port port-left port-storage-in" aria-hidden="true"/><span className="port port-right port-grid-out" aria-hidden="true"/>
    </article>
    <RealityNode slices={groundedSlices} settings={realitySettings} setSettings={setRealitySettings} position={realityPosition} onDragStart={startChildDrag("reality")}/>
    <ExportGeometryPreview slices={gridSlices} exportHeight={exportHeight} position={previewPosition} onDragStart={startChildDrag("preview")}/>
    <FrontClipPreview slices={gridSlices} heightInches={exportHeight} position={clipPosition} onDragStart={startChildDrag("clip")}/>
    <GridRationalizationPreview slices={gridSlices} gridSize={gridSize} alignment={gridAlignment} setGridSize={setGridSize} setAlignment={setGridAlignment} position={gridPosition} onDragStart={startChildDrag("grid")}/>
    <ReadOnlyBoundaryLinesPreview slices={gridSlices} activeIndex={scanIndex} fidelity={lineFidelity} setFidelity={setLineFidelity} primaryCount={primaryLineCount} setPrimaryCount={setPrimaryLineCount} offsetsPerPrimary={offsetsPerPrimary} setOffsetsPerPrimary={setOffsetsPerPrimary} bifurcationsPerPrimary={bifurcationsPerPrimary} setBifurcationsPerPrimary={setBifurcationsPerPrimary} bifurcationDistance={bifurcationDistance} setBifurcationDistance={setBifurcationDistance} bifurcationScale={bifurcationScale} setBifurcationScale={setBifurcationScale} smoothing={primarySmoothing} setSmoothing={setPrimarySmoothing} position={boundaryPosition} onDragStart={startChildDrag("boundary")}/>
    <CirculationPreview slices={gridSlices} fidelity={lineFidelity} primaryCount={primaryLineCount} offsetsPerPrimary={offsetsPerPrimary} bifurcationsPerPrimary={bifurcationsPerPrimary} bifurcationDistance={bifurcationDistance} bifurcationScale={bifurcationScale} smoothing={primarySmoothing} settings={circulationSettings} setSettings={setCirculationSettings} heightInches={exportHeight} setHeightInches={setExportHeight} position={circulationPosition} onDragStart={startChildDrag("circulation")}/>
    <CageCirculationPreview slices={gridSlices} fidelity={lineFidelity} primaryCount={primaryLineCount} offsetsPerPrimary={offsetsPerPrimary} bifurcationsPerPrimary={bifurcationsPerPrimary} bifurcationDistance={bifurcationDistance} bifurcationScale={bifurcationScale} smoothing={primarySmoothing} circulationSettings={circulationSettings} heightInches={exportHeight} position={frameworkPosition} onDragStart={startChildDrag("framework")}/>
    <UnifiedFloorPlatePreview slices={gridSlices} fidelity={lineFidelity} primaryCount={primaryLineCount} offsetsPerPrimary={offsetsPerPrimary} bifurcationsPerPrimary={bifurcationsPerPrimary} bifurcationDistance={bifurcationDistance} bifurcationScale={bifurcationScale} smoothing={primarySmoothing} circulationSettings={circulationSettings} settings={floorPlateSettings} setSettings={setFloorPlateSettings} heightInches={exportHeight} position={floorPosition} onDragStart={startChildDrag("floor")}/>
    <UnifiedRuledMeshPreview slices={gridSlices} fidelity={lineFidelity} primaryCount={primaryLineCount} offsetsPerPrimary={offsetsPerPrimary} bifurcationsPerPrimary={bifurcationsPerPrimary} bifurcationDistance={bifurcationDistance} bifurcationScale={bifurcationScale} smoothing={primarySmoothing} circulationSettings={circulationSettings} floorSettings={floorPlateSettings} heightInches={exportHeight} position={meshPosition} onDragStart={startChildDrag("mesh")}/>
    <ExportNode slices={gridSlices} lineFidelity={lineFidelity} primaryCount={primaryLineCount} offsetsPerPrimary={offsetsPerPrimary} exportHeight={exportHeight} setExportHeight={setExportHeight} position={exportPosition} onDragStart={startChildDrag("export")}/>
  </>;
}

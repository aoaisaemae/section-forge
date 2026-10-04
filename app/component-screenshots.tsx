"use client";
import {useState} from 'react';
import {saveExportFolder} from './export-destination';
export type ExportFile={name:string;blob:Blob};
const targets=[['01-circulation-skeleton.png','circulation','Circulation Skeleton'],['02-unified-section-cage.png','cage','Unified Section Cage'],['03-grid-rationalization.png','grid','Grid Rationalization'],['04-primary-circulation-loft.png','loft','Primary Circulation Loft']] as const;
/** Freeze the current DOM and canvas buffers, independent of board zoom or pan. */
function freeze(node:HTMLElement){
 const clone=node.cloneNode(true) as HTMLElement,originals=[node,...node.querySelectorAll('*')],copies=[clone,...clone.querySelectorAll('*')];
 originals.forEach((original,i)=>{
  const copy=copies[i] as HTMLElement,style=getComputedStyle(original);
  for(const property of Array.from(style))copy.style.setProperty(property,style.getPropertyValue(property),style.getPropertyPriority(property));
  copy.style.animation='none';copy.style.transition='none';
  if(original instanceof HTMLCanvasElement){const image=document.createElement('img');image.src=original.toDataURL('image/png');image.style.cssText=copy.style.cssText;image.width=original.width;image.height=original.height;copy.replaceWith(image);}
  if(original instanceof HTMLInputElement){
   if(original.type==='range'){
    const slider=document.createElement('div'),track=document.createElement('div'),thumb=document.createElement('div');slider.style.cssText=copy.style.cssText;slider.style.position='relative';slider.style.height='20px';slider.style.background='transparent';slider.style.border='0';slider.style.padding='0';const fraction=(Number(original.value)-Number(original.min||0))/(Number(original.max||100)-Number(original.min||0));
    track.style.cssText='position:absolute;left:0;right:0;top:8px;height:4px;background:#454545;border-radius:3px';track.style.background=`linear-gradient(to right, ${style.getPropertyValue('--space-color')||'#1ec1f2'} ${fraction*100}%, #454545 ${fraction*100}%)`;thumb.style.cssText=`position:absolute;left:calc(${fraction*100}% - 6px);top:4px;width:12px;height:12px;border-radius:50%;background:white`;slider.append(track,thumb);copy.replaceWith(slider);
   }else (copy as HTMLInputElement).setAttribute('value',original.value);
  }
 });
 const width=node.offsetWidth,height=Math.max(node.offsetHeight,node.scrollHeight);
 Object.assign(clone.style,{position:'relative',left:'0',top:'0',right:'auto',bottom:'auto',margin:'0',transform:'none',width:`${width}px`,height:`${height}px`,maxHeight:'none',overflow:'visible'});
 return {clone,width,height};
}
async function renderSnapshot(snapshot:ReturnType<typeof freeze>){
 await Promise.all(Array.from(snapshot.clone.querySelectorAll('img')).map(async img=>{if(img.src.startsWith('data:'))return;const response=await fetch(img.src);if(!response.ok)throw Error('A component image could not be captured.');const blob=await response.blob();img.src=await new Promise<string>((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=()=>reject(Error('Image capture failed'));reader.readAsDataURL(blob);});}));
 const {width,height,clone}=snapshot;clone.setAttribute('xmlns','http://www.w3.org/1999/xhtml');
 const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><foreignObject x="0" y="0" width="100%" height="100%">${new XMLSerializer().serializeToString(clone)}</foreignObject></svg>`;
 const image=new Image();await new Promise<void>((resolve,reject)=>{image.onload=()=>resolve();image.onerror=()=>reject(Error('This browser could not capture the component screenshot.'));image.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svg);});
 const canvas=document.createElement('canvas');canvas.width=width*2;canvas.height=height*2;const context=canvas.getContext('2d');if(!context)throw Error('Screenshot canvas unavailable');context.fillStyle='#000';context.fillRect(0,0,canvas.width,canvas.height);context.drawImage(image,0,0,canvas.width,canvas.height);
 return new Promise<Blob>((resolve,reject)=>{try{canvas.toBlob(blob=>blob?resolve(blob):reject(Error('PNG capture failed')),'image/png');}catch{reject(Error('The browser blocked screenshot capture.'));}});
}
export async function zipFolder(files:ExportFile[],folder:string){
 const parts:Uint8Array[]=[],central:Uint8Array[]=[],encoder=new TextEncoder();let offset=0;
 const crc=(bytes:Uint8Array)=>{let c=0xffffffff;for(const byte of bytes){c^=byte;for(let i=0;i<8;i++)c=(c>>>1)^((c&1)?0xedb88320:0);}return (c^0xffffffff)>>>0;};
 for(const file of files){const bytes=new Uint8Array(await file.blob.arrayBuffer()),name=encoder.encode(`${folder}/${file.name}`),checksum=crc(bytes),header=new Uint8Array(30+name.length),h=new DataView(header.buffer);h.setUint32(0,0x04034b50,true);h.setUint16(4,20,true);h.setUint16(6,0x800,true);h.setUint32(14,checksum,true);h.setUint32(18,bytes.length,true);h.setUint32(22,bytes.length,true);h.setUint16(26,name.length,true);header.set(name,30);parts.push(header,bytes);
 const entry=new Uint8Array(46+name.length),e=new DataView(entry.buffer);e.setUint32(0,0x02014b50,true);e.setUint16(4,20,true);e.setUint16(6,20,true);e.setUint16(8,0x800,true);e.setUint32(16,checksum,true);e.setUint32(20,bytes.length,true);e.setUint32(24,bytes.length,true);e.setUint16(28,name.length,true);e.setUint32(42,offset,true);entry.set(name,46);central.push(entry);offset+=header.length+bytes.length;
 }
 const size=central.reduce((sum,item)=>sum+item.length,0),end=new Uint8Array(22),e=new DataView(end.buffer);e.setUint32(0,0x06054b50,true);e.setUint16(8,files.length,true);e.setUint16(10,files.length,true);e.setUint32(12,size,true);e.setUint32(16,offset,true);return new Blob([...parts,...central,end],{type:'application/zip'});
}
export function ComponentScreenshotExport({position,onDragStart}:{position:{x:number;y:number};onDragStart:(event:React.PointerEvent)=>void}){
 const [busy,setBusy]=useState(false),[status,setStatus]=useState('');
 const run=async()=>{setBusy(true);setStatus('Capturing four components…');try{await document.fonts.ready;const snapshots=targets.map(([name,id,title])=>{const node=document.querySelector<HTMLElement>(`[data-screenshot-component="${id}"]`);if(!node)throw Error(`${title} is unavailable.`);return {name,...freeze(node)};});const files=await Promise.all(snapshots.map(async snapshot=>({name:snapshot.name,blob:await renderSnapshot(snapshot)})));const folder=`sand-scan-components-${new Date().toISOString().replace(/[:.]/g,'-')}`;const zip=await zipFolder(files,folder);const result=await saveExportFolder(files,folder,zip);setStatus(result);}catch(error){setStatus(error instanceof Error?error.message:'Screenshot export failed.');}finally{setBusy(false);}};
 return <article className="spatial-rational-node component-screenshot-export" style={{left:position.x,top:position.y}}><header onPointerDown={onDragStart}><div className="model-icon">PNG</div><div><h2>Component Screenshot Export</h2><p>Four component screenshots · one folder</p></div><span>DRAG</span></header><div className="illustrator-content"><ul>{targets.map(([name,id,title])=><li key={id}>{title}</li>)}</ul><p>Captures the current view, controls, values, and labels of each whole component at 2× resolution.</p><p>Choose an export folder in the top bar to save a new subfolder. Browser downloads deliver a ZIP containing the same folder.</p><button onClick={()=>void run()} disabled={busy}>{busy?'Capturing…':'Export screenshot folder'}</button><p role="status">{status}</p></div></article>;
}

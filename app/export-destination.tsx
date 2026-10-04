"use client";
import {useEffect,useState} from "react";
type Directory={getDirectoryHandle:(name:string,options:{create:boolean})=>Promise<Directory>;name:string;getFileHandle:(name:string,options?:{create:boolean})=>Promise<{createWritable:()=>Promise<{write:(blob:Blob)=>Promise<void>;close:()=>Promise<void>}>}>};
let directory:Directory|null=null;
let pending:{blob:Blob;name:string}|null=null;
let message="Browser downloads";
let queue=Promise.resolve();
const announce=(text:string)=>{message=text;window.dispatchEvent(new Event("sand-scan-export-status"));};
function download(blob:Blob,name:string){const url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
export function saveExport(blob:Blob,name:string){
 const target=directory;
 queue=queue.then(async()=>{
  if(!target){download(blob,name);announce(`Downloaded ${name}`);return;}
  try{
   // Preserve existing exports instead of overwriting files in the chosen folder.
   const dot=name.lastIndexOf('.'),stem=dot<0?name:name.slice(0,dot),extension=dot<0?'':name.slice(dot);
   let filename=name;
   for(let i=0;;i++){filename=i?`${stem} (${i})${extension}`:name;try{await target.getFileHandle(filename);}catch(error){if((error as DOMException).name==='NotFoundError')break;throw error;}}
   const handle=await target.getFileHandle(filename,{create:true}),writer=await handle.createWritable();await writer.write(blob);await writer.close();pending=null;announce(`Saved ${filename} in ${target.name}`);
  }catch(error){pending={blob,name};announce(`Could not save ${name}. Choose the folder again or download this export.`);}
 });return queue;
}
export async function saveExportFolder(files:{name:string;blob:Blob}[],folderName:string,zip:Blob){
 const target=directory;
 if(!target){await saveExport(zip,`${folderName}.zip`);return 'Downloaded ZIP containing all four screenshots.';}
 let result='';
 queue=queue.then(async()=>{try{const folder=await target.getDirectoryHandle(folderName,{create:true});for(const file of files){const handle=await folder.getFileHandle(file.name,{create:true}),writer=await handle.createWritable();await writer.write(file.blob);await writer.close();}result=`Saved four screenshots in ${target.name}/${folderName}`;announce(result);}catch{pending={blob:zip,name:`${folderName}.zip`};result='Folder export could not finish. Use Download unsaved export in the top bar for the complete ZIP.';announce(result);}});
 await queue;return result;
}
export function ExportDestination(){
 const [folder,setFolder]=useState<string|null>(null),[status,setStatus]=useState(message),[supported,setSupported]=useState(false),[failed,setFailed]=useState(false);
 useEffect(()=>{setSupported('showDirectoryPicker' in window);const update=()=>{setStatus(message);setFailed(!!pending);};window.addEventListener('sand-scan-export-status',update);return()=>window.removeEventListener('sand-scan-export-status',update);},[]);
 const choose=async()=>{try{const selected=await (window as unknown as {showDirectoryPicker:(options:object)=>Promise<Directory>}).showDirectoryPicker({id:'sand-scan-exports',mode:'readwrite'});directory=selected;setFolder(selected.name);announce(`All exports will save in ${selected.name}`);if(pending){const file=pending;await saveExport(file.blob,file.name);}}catch(error){if((error as DOMException).name!=='AbortError')announce('Folder access is unavailable here. Open Sand Scan directly in a supported desktop browser, or use browser downloads.');}};
 return <div className="export-destination"><span>Export destination: <strong>{folder||'Browser downloads'}</strong></span><button onClick={()=>void choose()} disabled={!supported}>{folder?'Change folder':'Choose export folder'}</button>{folder&&<button onClick={()=>{directory=null;setFolder(null);announce('Browser downloads selected');}}>Use browser downloads</button>}{failed&&<button onClick={()=>{if(pending){download(pending.blob,pending.name);pending=null;announce('Export downloaded');}}}>Download unsaved export</button>}<small role="status">{supported?status:'Folder selection is unavailable in this browser. Exports use browser downloads.'}</small></div>;
}

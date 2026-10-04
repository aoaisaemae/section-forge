"use client";
import {useCallback,useEffect,useRef,useState,type SetStateAction} from "react";

const STORAGE="sand-scan-workspace-v1";
type Snapshot=Record<string,unknown>;
let values:Snapshot={},loaded=false,restoring=false,gesture=false;
const listeners=new Map<string,(value:unknown)=>void>();
let past:Snapshot[]=[],future:Snapshot[]=[],pending:Snapshot|null=null;
let timer:ReturnType<typeof setTimeout>|undefined,saveTimer:ReturnType<typeof setTimeout>|undefined;
const copy=(value:Snapshot):Snapshot=>JSON.parse(JSON.stringify(value));
function load(){if(loaded||typeof window==="undefined")return;loaded=true;try{const saved=JSON.parse(localStorage.getItem(STORAGE)||"null");if(saved&&saved.version===1&&saved.values&&typeof saved.values==="object")values=saved.values}catch{}}
function save(){try{localStorage.setItem(STORAGE,JSON.stringify({version:1,values}))}catch{/* Storage may be unavailable; editing still works. */}}
function flush(){if(timer)clearTimeout(timer);timer=undefined;if(pending){past.push(pending);past=past.slice(-10);future=[];pending=null}}
function change(key:string,value:unknown){if(JSON.stringify(values[key])===JSON.stringify(value))return;if(!restoring&&!pending)pending=copy(values);values[key]=value;if(saveTimer)clearTimeout(saveTimer);saveTimer=setTimeout(save,150);if(!gesture){if(timer)clearTimeout(timer);timer=setTimeout(flush,250)}}
function apply(snapshot:Snapshot){restoring=true;values=copy(snapshot);listeners.forEach((notify,key)=>{if(key in values)notify(values[key])});restoring=false;save()}
export function undoWorkspace(){flush();const previous=past.pop();if(previous){future.push(copy(values));apply(previous)}}
export function redoWorkspace(){flush();const next=future.pop();if(next){past.push(copy(values));apply(next)}}
export function useWorkspaceState<T>(key:string,initial:T|(()=>T)){
 const [value,setValue]=useState<T>(initial),current=useRef(value);
 useEffect(()=>{load();const notify=(next:unknown)=>{current.current=next as T;setValue(next as T)};listeners.set(key,notify);if(key in values)notify(values[key]);else values[key]=current.current;return()=>{listeners.delete(key)}},[key]);
 const set=useCallback((next:SetStateAction<T>)=>{const resolved=typeof next==="function"?(next as (previous:T)=>T)(current.current):next;current.current=resolved;change(key,resolved);setValue(resolved)},[key]);
 return [value,set] as const;
}
export function useWorkspaceControls(board:React.RefObject<HTMLElement|null>){
 useEffect(()=>{
  const element=board.current;if(!element)return;load();const scroll=values["board.scroll"] as {x:number;y:number}|undefined;
  if(scroll)requestAnimationFrame(()=>{element.scrollLeft=scroll.x;element.scrollTop=scroll.y});
  let pan:{x:number;y:number;left:number;top:number;id:number}|null=null;
  const begin=(event:PointerEvent)=>{flush();gesture=true;if(event.button!==1||!element.contains(event.target as Node))return;event.preventDefault();event.stopImmediatePropagation();pan={x:event.clientX,y:event.clientY,left:element.scrollLeft,top:element.scrollTop,id:event.pointerId};element.setPointerCapture(event.pointerId);element.classList.add("board-panning")};
  const move=(event:PointerEvent)=>{if(!pan)return;event.preventDefault();event.stopImmediatePropagation();element.scrollLeft=pan.left+pan.x-event.clientX;element.scrollTop=pan.top+pan.y-event.clientY};
  const end=(event:PointerEvent)=>{if(pan){event.preventDefault();event.stopImmediatePropagation();if(element.hasPointerCapture(pan.id))element.releasePointerCapture(pan.id);pan=null;element.classList.remove("board-panning")}gesture=false;flush()};
  const key=(event:KeyboardEvent)=>{const target=event.target as HTMLElement;const editing=target?.closest('textarea,input:not([type="range"]):not([type="checkbox"]):not([type="radio"]),[contenteditable="true"]');if(editing||!(event.ctrlKey||event.metaKey)||event.key.toLowerCase()!=="z")return;event.preventDefault();event.stopImmediatePropagation();gesture=false;event.altKey?redoWorkspace():undoWorkspace()};
  const remember=()=>{values["board.scroll"]={x:element.scrollLeft,y:element.scrollTop};if(saveTimer)clearTimeout(saveTimer);saveTimer=setTimeout(save,150)};
  const leave=()=>{gesture=false;flush();save()};
  const aux=(event:MouseEvent)=>{if(event.button===1)event.preventDefault()};
  window.addEventListener("pointerdown",begin,true);window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",end,true);window.addEventListener("pointercancel",end,true);window.addEventListener("keydown",key,true);window.addEventListener("blur",leave);window.addEventListener("pagehide",leave);element.addEventListener("scroll",remember);element.addEventListener("auxclick",aux);
  return()=>{leave();window.removeEventListener("pointerdown",begin,true);window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",end,true);window.removeEventListener("pointercancel",end,true);window.removeEventListener("keydown",key,true);window.removeEventListener("blur",leave);window.removeEventListener("pagehide",leave);element.removeEventListener("scroll",remember);element.removeEventListener("auxclick",aux)};
 },[board]);
}

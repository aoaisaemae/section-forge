// Streaming GIF89a with a fixed color+gray palette. Clear before 9-bit code growth.
export class ModelGif {
 private chunks:Uint8Array[]=[];
 frames=0;
 constructor(private width:number,private height:number){
 const bytes=[...Array.from('GIF89a').map(c=>c.charCodeAt(0)),width&255,width>>8,height&255,height>>8,247,0,0];
 for(let i=0;i<216;i++)bytes.push(Math.floor(i/36)*51,Math.floor(i/6)%6*51,i%6*51);
 for(let i=0;i<40;i++){const gray=Math.round(i*255/39);bytes.push(gray,gray,gray)}
 bytes.push(33,255,11,...Array.from('NETSCAPE2.0').map(c=>c.charCodeAt(0)),3,1,0,0,0);this.chunks.push(new Uint8Array(bytes));
 }
 add(rgba:Uint8ClampedArray,delay=12){
 const pixels=new Uint8Array(this.width*this.height);for(let i=0;i<pixels.length;i++){const r=rgba[i*4],g=rgba[i*4+1],b=rgba[i*4+2];pixels[i]=Math.max(r,g,b)-Math.min(r,g,b)<24?216+Math.round((r+g+b)/3*39/255):Math.round(r/51)*36+Math.round(g/51)*6+Math.round(b/51);}
 const packed:number[]=[];let bits=0,count=0;const code=(v:number)=>{bits|=v<<count;count+=9;while(count>=8){packed.push(bits&255);bits>>>=8;count-=8}};
 for(let i=0;i<pixels.length;i++){if(i%200===0)code(256);code(pixels[i]);}code(257);if(count)packed.push(bits&255);
 const bytes=[33,249,4,4,delay&255,delay>>8,0,0,44,0,0,0,0,this.width&255,this.width>>8,this.height&255,this.height>>8,0,8];for(let i=0;i<packed.length;i+=255){const block=packed.slice(i,i+255);bytes.push(block.length,...block)}bytes.push(0);this.chunks.push(new Uint8Array(bytes));this.frames++;
 }
 blob(){return new Blob([...this.chunks,new Uint8Array([59])],{type:'image/gif'});}
}

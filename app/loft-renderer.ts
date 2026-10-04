type V = [number, number, number];
export type LoftModel = { rows: V[][]; rails: V[][] };
export type LoftCamera = { yaw: number; pitch: number; zoom: number; panX: number; panY: number; orthographic?: boolean };
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V, b: V): V => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const normalize = (p: V): V => { const l = Math.hypot(...p); return l > 1e-9 ? p.map(v=>v/l) as V : [0,1,0]; };

// Shared grid normals give a smooth membrane, rather than individually shaded facets.
export function loftDisplayMesh(rows: V[][], thickness: number) {
  const positions: number[] = [], normals: number[] = [];
  if (rows.length < 2 || !rows[0]?.length) return { positions: new Float32Array(), normals: new Float32Array() };
  const height = rows.length, width = rows[0].length;
  const ns = rows.map((row,i)=>row.map((p,j)=>{
    let left = Math.max(0,j-1), right = Math.min(width-1,j+1);
    while(left>0 && Math.hypot(...sub(p,row[left]))<1e-8)left--;
    while(right<width-1 && Math.hypot(...sub(p,row[right]))<1e-8)right++;
    const along = sub(rows[Math.min(height-1,i+1)][j],rows[Math.max(0,i-1)][j]);
    return normalize(cross(along,sub(row[right],row[left])));
  }));
  // Coincident samples share one extrusion direction so adjacent patches cannot split.
  const key=(p:V)=>p.map(v=>v.toFixed(8)).join(',');
  const shared=new Map<string,V>();
  rows.forEach((row,i)=>row.forEach((p,j)=>{const k=key(p),n=shared.get(k)||[0,0,0];shared.set(k,n.map((v,d)=>v+ns[i][j][d]) as V);}));
  rows.forEach((row,i)=>row.forEach((p,j)=>{ns[i][j]=normalize(shared.get(key(p))!);}));
  const lower=rows.map((row,i)=>row.map((p,j)=>p.map((v,d)=>v-thickness*ns[i][j][d]) as V));
  const add = (p: V,n: V) => { positions.push(...p);normals.push(...n); };
  for(let i=0;i<height-1;i++)for(let j=0;j<width-1;j++){
    const ids=[[i,j],[i+1,j],[i+1,j+1],[i,j+1]];
    for(const k of [0,1,2,0,2,3]) { const [a,b]=ids[k];add(rows[a][b],ns[a][b]); }
    if(thickness>0)for(const k of [0,2,1,0,3,2]) { const [a,b]=ids[k];add(lower[a][b],ns[a][b].map(v=>-v) as V); }
  }
  if(thickness>0){
    const rim:[number,number][]=[...rows[0].map((_,j)=>[0,j] as [number,number]),...rows.slice(1).map((_,i)=>[i+1,width-1] as [number,number]),...rows[height-1].slice(0,-1).map((_,j)=>[height-1,width-2-j] as [number,number]),...rows.slice(1,-1).map((_,i)=>[height-2-i,0] as [number,number])];
    rim.forEach(([i,j],index)=>{
      const [k,l]=rim[(index+1)%rim.length],a=rows[i][j],b=rows[k][l],c=lower[k][l],d=lower[i][j];
      for(const triangle of [[a,d,c],[a,c,b]]){const n=normalize(cross(sub(triangle[1],triangle[0]),sub(triangle[2],triangle[0])));triangle.forEach(p=>add(p,n));}
    });
  }
  return {positions:new Float32Array(positions),normals:new Float32Array(normals)};
}

type RenderMesh = {positions:Float32Array;normals:Float32Array};
export function clipLoftMesh(mesh:RenderMesh,planeZ:number|null){
  if(planeZ===null)return {...mesh,section:new Float32Array()};
  const positions:number[]=[],normals:number[]=[],section:number[]=[];
  type Vertex={p:V;n:V};
  for(let t=0;t<mesh.positions.length;t+=9){
    const triangle:Vertex[]=[0,3,6].map(k=>({p:Array.from(mesh.positions.slice(t+k,t+k+3)) as V,n:Array.from(mesh.normals.slice(t+k,t+k+3)) as V}));
    const polygon:Vertex[]=[],hits:V[]=[];
    triangle.forEach((a,i)=>{const b=triangle[(i+1)%3],insideA=a.p[2]>=planeZ,insideB=b.p[2]>=planeZ;
      if(insideA)polygon.push(a);
      if(insideA!==insideB){const u=(planeZ-a.p[2])/(b.p[2]-a.p[2]),p=a.p.map((v,k)=>v+(b.p[k]-v)*u) as V,n=normalize(a.n.map((v,k)=>v+(b.n[k]-v)*u) as V);p[2]=planeZ;polygon.push({p,n});hits.push(p);}
    });
    if(hits.length===2)section.push(...hits[0],...hits[1]);
    for(let k=1;k<polygon.length-1;k++)for(const v of [polygon[0],polygon[k],polygon[k+1]]){positions.push(...v.p);normals.push(...v.n);}
  }
  return {positions:new Float32Array(positions),normals:new Float32Array(normals),section:new Float32Array(section)};
}
function clipDepth(model:LoftModel,clip:number|null){
  if(clip===null)return null;
  let low=Infinity,high=-Infinity;for(const row of model.rows)for(const p of row){low=Math.min(low,p[2]);high=Math.max(high,p[2]);}
  return Number.isFinite(low)?(clip<=0?low-1e-6:clip>=100?high+1e-6:low+(high-low)*clip/100):null;
}

// Pixel depth testing and interpolated normals keep the fallback smoothly shaded.
function createSoftwareLoftRenderer(canvas:HTMLCanvasElement){
  const ctx=canvas.getContext('2d');if(!ctx)throw new Error('Canvas unavailable');
  let cached:V[][]|null=null,lastThickness=-1,lastClip:number|null=null,mesh=clipLoftMesh(loftDisplayMesh([],0),null),center:V=[0,0,0];

  return {draw(model:LoftModel,camera:LoftCamera,thicknessInches:number,blue:boolean,showRails:boolean,clip:number|null=null){
    const thickness=thicknessInches/240*2.1;
    if(cached!==model.rows||lastThickness!==thickness||lastClip!==clip){cached=model.rows;lastThickness=thickness;lastClip=clip;mesh=clipLoftMesh(loftDisplayMesh(model.rows,thickness),clipDepth(model,clip));let low=Infinity,high=-Infinity;for(const row of model.rows)for(const p of row){low=Math.min(low,p[1]);high=Math.max(high,p[1]);}center=[0,Number.isFinite(low)?(low+high)/2:0,0];}
    const w=canvas.width,h=canvas.height,pixels=ctx.createImageData(w,h),depth=new Float32Array(w*h);depth.fill(Infinity);for(let i=3;i<pixels.data.length;i+=4)pixels.data[i]=255;
    const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch);
    const rotate=(x:number,y:number,z:number):V=>{const rx=x*cy+z*sy,rz=-x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]};
    const project=(p:V)=>{const r=rotate(p[0],p[1]-center[1],p[2]),z=5.3+r[2],scale=3.7*camera.zoom*h/2/(camera.orthographic?5.3:z);return[w/2+(r[0]+camera.panX)*scale,h/2-(r[1]+camera.panY)*scale,z] as V;};
    const light=normalize(rotate(-.25,1,-.3)),fill=normalize(rotate(.5,.6,.4)),half=normalize([light[0],light[1],light[2]-1]);
    const base=blue?[.04,.35,.86]:[.83,.85,.88];
    for(let t=0;t<mesh.positions.length;t+=9){
      const ps=[0,3,6].map(k=>project([mesh.positions[t+k],mesh.positions[t+k+1],mesh.positions[t+k+2]]));
      const ns=[0,3,6].map(k=>rotate(mesh.normals[t+k],mesh.normals[t+k+1],mesh.normals[t+k+2]));
      const [a,b,c]=ps,den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(Math.abs(den)<1e-8)continue;
      const minX=Math.max(0,Math.floor(Math.min(a[0],b[0],c[0]))),maxX=Math.min(w-1,Math.ceil(Math.max(a[0],b[0],c[0]))),minY=Math.max(0,Math.floor(Math.min(a[1],b[1],c[1]))),maxY=Math.min(h-1,Math.ceil(Math.max(a[1],b[1],c[1])));
      const sign=(ns[0][2]+ns[1][2]+ns[2][2])>0?-1:1;
      for(let y=minY;y<=maxY;y++)for(let x=minX;x<=maxX;x++){
        let u=((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/den,v=((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/den,q=1-u-v;if(u<0||v<0||q<0)continue;
        const index=y*w+x;let z:number;if(camera.orthographic){z=u*a[2]+v*b[2]+q*c[2]}else{u/=a[2];v/=b[2];q/=c[2];z=1/(u+v+q);u*=z;v*=z;q*=z}if(z>=depth[index])continue;depth[index]=z;
        let nx=(ns[0][0]*u+ns[1][0]*v+ns[2][0]*q)*sign,ny=(ns[0][1]*u+ns[1][1]*v+ns[2][1]*q)*sign,nz=(ns[0][2]*u+ns[1][2]*v+ns[2][2]*q)*sign;const l=Math.hypot(nx,ny,nz)||1;nx/=l;ny/=l;nz/=l;
        const diffuse=Math.max(0,nx*light[0]+ny*light[1]+nz*light[2]),secondary=Math.max(0,nx*fill[0]+ny*fill[1]+nz*fill[2]),spec=Math.pow(Math.max(0,nx*half[0]+ny*half[1]+nz*half[2]),42),rim=Math.pow(1-Math.abs(nz),3),strength=.14+.72*diffuse+.18*secondary;
        for(let k=0;k<3;k++)pixels.data[index*4+k]=Math.min(255,Math.pow(base[k]*strength+.42*spec+.065*rim,.85)*255);
      }
    }
    ctx.putImageData(pixels,0,0);
    const planeZ=clipDepth(model,clip);
    if(mesh.section.length){ctx.strokeStyle='#ff268c';ctx.lineWidth=1.8;ctx.beginPath();for(let i=0;i<mesh.section.length;i+=6){const a=project(Array.from(mesh.section.slice(i,i+3)) as V),b=project(Array.from(mesh.section.slice(i+3,i+6)) as V);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}ctx.stroke();}
    if(showRails){ctx.strokeStyle='#1680ff';ctx.lineWidth=1.4;model.rails.forEach(rail=>{ctx.beginPath();rail.forEach((p,i)=>{const s=project(p);if(planeZ!==null&&p[2]<planeZ)return;i&&!(planeZ!==null&&rail[i-1][2]<planeZ)?ctx.lineTo(s[0],s[1]):ctx.moveTo(s[0],s[1])});ctx.stroke()});}
  },dispose(){ctx.clearRect(0,0,canvas.width,canvas.height)}};
}

export function createLoftRenderer(canvas: HTMLCanvasElement, forceSoftware = false) {
  const gl=forceSoftware?null:canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true});
  if(!gl)return createSoftwareLoftRenderer(canvas);
  const shader=(type:number,source:string)=>{const s=gl.createShader(type)!;gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s)??'Shader failed');return s;};
  const vertex=shader(gl.VERTEX_SHADER,`
    attribute vec3 position;attribute vec3 normal;
    uniform vec4 camera;uniform vec2 pan;uniform float aspect;uniform vec3 center;
    varying vec3 N;varying vec3 P;varying vec3 L;varying vec3 F;
    vec3 rotate(vec3 p){float cy=cos(camera.x),sy=sin(camera.x),cp=cos(camera.y),sp=sin(camera.y);float x=p.x*cy+p.z*sy,z=-p.x*sy+p.z*cy;return vec3(x,p.y*cp-z*sp,p.y*sp+z*cp);}
    void main(){vec3 p=rotate(position-center);p.xy+=pan;p.z+=5.3;float f=3.7*camera.z;gl_Position=camera.w>0.5?vec4(p.x*f/(aspect*5.3),p.y*f/5.3,(p.z-5.3)/10.,1.):vec4(p.x*f/aspect,p.y*f,1.002*p.z-0.2002,p.z);N=rotate(normal);P=p;L=rotate(normalize(vec3(-.25,1.,-.3)));F=rotate(normalize(vec3(.5,.6,.4)));}
  `);
  const fragment=shader(gl.FRAGMENT_SHADER,`
    precision mediump float;varying vec3 N;varying vec3 P;varying vec3 L;varying vec3 F;uniform vec3 material;uniform bool lineMode;
    void main(){if(lineMode){gl_FragColor=vec4(material,1.);return;}vec3 n=normalize(N),v=normalize(-P);if(dot(n,v)<0.)n=-n;vec3 l=normalize(L),fill=normalize(F);float diffuse=max(dot(n,l),0.),secondary=max(dot(n,fill),0.);float spec=pow(max(dot(n,normalize(l+v)),0.),42.);float rim=pow(1.-abs(dot(n,v)),3.);vec3 color=material*(.14+.72*diffuse+.18*secondary)+vec3(.42*spec+.065*rim);gl_FragColor=vec4(pow(color,vec3(.85)),1.);}
  `);
  const program=gl.createProgram()!;gl.attachShader(program,vertex);gl.attachShader(program,fragment);gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error('Renderer failed');
  const pb=gl.createBuffer()!,nb=gl.createBuffer()!,rb=gl.createBuffer()!,sb=gl.createBuffer()!;
  const pos=gl.getAttribLocation(program,'position'),norm=gl.getAttribLocation(program,'normal');
  const uniform=(name:string)=>gl.getUniformLocation(program,name);
  const uniforms={camera:uniform('camera'),pan:uniform('pan'),aspect:uniform('aspect'),center:uniform('center'),material:uniform('material'),lineMode:uniform('lineMode')};
  let cached:V[][]|null=null,lastThickness=-1,lastClip:number|null=null,vertexCount=0,railCount=0,sectionCount=0,center:V=[0,0,0];
  const attribute=(buffer:WebGLBuffer,location:number)=>{gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,3,gl.FLOAT,false,0,0);};
  return {
    draw(model:LoftModel,camera:LoftCamera,thicknessInches:number,blue:boolean,showRails:boolean,clip:number|null=null){
      const thickness=thicknessInches/240*2.1;
      if(cached!==model.rows||lastThickness!==thickness||lastClip!==clip){
        cached=model.rows;lastThickness=thickness;lastClip=clip;
        const planeZ=clipDepth(model,clip),mesh=clipLoftMesh(loftDisplayMesh(model.rows,thickness),planeZ);
        sectionCount=mesh.section.length/3;gl.bindBuffer(gl.ARRAY_BUFFER,sb);gl.bufferData(gl.ARRAY_BUFFER,mesh.section,gl.STATIC_DRAW);vertexCount=mesh.positions.length/3;
        gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,mesh.positions,gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER,nb);gl.bufferData(gl.ARRAY_BUFFER,mesh.normals,gl.STATIC_DRAW);
        const lines=model.rails.flatMap(rail=>rail.slice(1).flatMap((p,i)=>{const a=rail[i];if(planeZ===null)return[...a,...p];if(a[2]<planeZ&&p[2]<planeZ)return[];const intersection=()=>a.map((v,k)=>v+(p[k]-v)*(planeZ-a[2])/(p[2]-a[2]));return [...(a[2]<planeZ?intersection():a),...(p[2]<planeZ?intersection():p)]}));railCount=lines.length/3;
        gl.bindBuffer(gl.ARRAY_BUFFER,rb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(lines),gl.STATIC_DRAW);
        let low=Infinity,high=-Infinity;for(const row of model.rows)for(const p of row){low=Math.min(low,p[1]);high=Math.max(high,p[1]);}center=[0,Number.isFinite(low)?(low+high)/2:0,0];
      }
      gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(0,0,0,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.useProgram(program);
      gl.uniform4f(uniforms.camera,camera.yaw,camera.pitch,camera.zoom,camera.orthographic?1:0);gl.uniform2f(uniforms.pan,camera.panX,camera.panY);gl.uniform1f(uniforms.aspect,canvas.width/canvas.height);gl.uniform3fv(uniforms.center,center);
      gl.uniform3fv(uniforms.material,blue?[.04,.35,.86]:[.83,.85,.88]);gl.uniform1i(uniforms.lineMode,0);
      gl.enable(gl.POLYGON_OFFSET_FILL);gl.polygonOffset(1,1);attribute(pb,pos);attribute(nb,norm);gl.drawArrays(gl.TRIANGLES,0,vertexCount);
      gl.disable(gl.POLYGON_OFFSET_FILL);
      if(sectionCount){gl.uniform1i(uniforms.lineMode,1);gl.uniform3fv(uniforms.material,[1,.15,.55]);attribute(sb,pos);gl.disableVertexAttribArray(norm);gl.vertexAttrib3f(norm,0,0,-1);gl.drawArrays(gl.LINES,0,sectionCount);}
      if(showRails){gl.uniform1i(uniforms.lineMode,1);gl.uniform3fv(uniforms.material,[.08,.5,1]);attribute(rb,pos);gl.disableVertexAttribArray(norm);gl.vertexAttrib3f(norm,0,1,0);gl.depthFunc(gl.LEQUAL);gl.drawArrays(gl.LINES,0,railCount);gl.depthFunc(gl.LESS);}
    },
    dispose(){[pb,nb,rb,sb].forEach(b=>gl.deleteBuffer(b));gl.deleteProgram(program);gl.deleteShader(vertex);gl.deleteShader(fragment);}
  };
}

"""3D 可视化: 生成完全自包含的交互式 HTML (内置 WebGL 渲染器, 不依赖任何外部库/CDN).

  * 鼠标左键拖拽 = 旋转, 右键/Shift+拖拽 = 平移, 滚轮 = 缩放
  * 在可通行面上单击 = 发布 3D 目标点
      - 离线模式: 显示该点坐标并生成对应的命令行
      - 服务模式 (python -m dddmr_py serve): 直接请求规划并实时画出路径
"""

from __future__ import annotations

import base64
import json
from typing import Optional

import numpy as np

from .perception import GroundMap
from .planner import Path


# --------------------------------------------------------------------------
def _b64(arr: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode()


def _cost_colors(cost: np.ndarray, lethal: np.ndarray, lethal_cost: float) -> np.ndarray:
    """代价 -> 颜色 (绿 -> 黄 -> 橙 -> 红), 致命节点为深红."""
    t = np.clip(cost / max(lethal_cost, 1e-6), 0.0, 1.0)
    stops = np.array([[47, 158, 110], [200, 185, 59], [224, 123, 57], [192, 57, 43]],
                     dtype=np.float64)
    pos = np.array([0.0, 0.35, 0.7, 1.0])
    rgb = np.empty((len(t), 3))
    for k in range(3):
        rgb[:, k] = np.interp(t, pos, stops[:, k])
    rgb[lethal] = np.array([150, 40, 45])
    return rgb.astype(np.uint8)


def _subsample(n: int, limit: int, seed: int = 0) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=limit, replace=False)


def build_scene_data(gmap: GroundMap, path: Optional[Path] = None, start=None, goal=None,
                     map_name: str = "map.pcd", max_nodes: int = 60000,
                     max_obstacles: int = 60000) -> dict:
    ni = _subsample(len(gmap.nodes), max_nodes)
    oi = _subsample(len(gmap.obstacles), max_obstacles, seed=1)
    nodes = gmap.nodes[ni]
    obst = gmap.obstacles[oi]
    node_col = _cost_colors(gmap.cost[ni], gmap.lethal[ni], gmap.config.lethal_cost)
    obst_col = np.tile(np.array([[110, 118, 132]], dtype=np.uint8), (len(obst), 1))

    pos = np.vstack([nodes, obst]) if len(obst) else nodes
    col = np.vstack([node_col, obst_col]) if len(obst) else node_col
    center = pos.mean(axis=0)
    extent = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))

    data = {
        "map_name": map_name,
        "n_ground": int(len(nodes)),
        "n_total": int(len(pos)),
        "pos": _b64(pos, np.float32),
        "col": _b64(col, np.uint8),
        "bbox": [pos.min(axis=0).tolist(), pos.max(axis=0).tolist()],
        "center": center.tolist(),
        "extent": extent,
        "path": None,
        "info": None,
        "start": [float(v) for v in start] if start is not None else None,
        "goal": [float(v) for v in goal] if goal is not None else None,
        "stats": {"nodes": int(len(gmap.nodes)), "lethal": int(gmap.lethal.sum()),
                  "obstacles": int(len(gmap.obstacles))},
    }
    if path is not None and path.success and len(path.points):
        data["path"] = _b64(path.points, np.float32)
        data["n_path"] = int(len(path.points))
        data["info"] = {"length": path.length, "climb": path.climb,
                        "planning_time": path.planning_time}
    return data


_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>dddmr-py · 3D 导航可视化</title>
<style>
  html,body{margin:0;height:100%;background:#0d1017;color:#e6e9ef;overflow:hidden;
    font-family:-apple-system,"PingFang SC","Microsoft YaHei",Segoe UI,sans-serif;}
  canvas{display:block;width:100vw;height:100vh;cursor:crosshair;}
  #panel{position:fixed;top:14px;left:14px;background:rgba(16,19,26,.93);border:1px solid #262c38;
    border-radius:10px;padding:12px 14px;min-width:300px;font-size:13px;line-height:1.75;
    box-shadow:0 8px 28px rgba(0,0,0,.5);backdrop-filter:blur(6px);}
  #panel h3{margin:0 0 8px;font-size:14px;font-weight:600;letter-spacing:.3px;}
  .kv{color:#8b95a5;display:inline-block;min-width:64px;}
  .val{color:#e6e9ef;font-variant-numeric:tabular-nums;}
  button{background:#2b6cb0;color:#fff;border:0;border-radius:6px;padding:6px 11px;font-size:12px;
    cursor:pointer;margin:8px 6px 0 0;}
  button:hover{background:#3b82d6;} button.sec{background:#333a48;} button.sec:hover{background:#414a5c;}
  button.on{background:#2f9e6e;}
  code{background:#161a22;padding:3px 6px;border-radius:4px;font-size:11px;word-break:break-all;
    display:block;margin-top:6px;color:#9fd0ff;}
  .hint{color:#78839a;font-size:12px;margin-top:8px;}
  #legend{position:fixed;right:14px;bottom:14px;background:rgba(16,19,26,.9);border:1px solid #262c38;
    border-radius:8px;padding:9px 12px;font-size:12px;color:#8b95a5;}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px;}
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="panel">
  <h3>dddmr-py · 3D 全局规划</h3>
  <div><span class="kv">起点</span><span class="val" id="startTxt">-</span></div>
  <div><span class="kv">目标点</span><span class="val" id="goalTxt">-</span></div>
  <div><span class="kv">路径长度</span><span class="val" id="lenTxt">-</span></div>
  <div><span class="kv">累计爬升</span><span class="val" id="climbTxt">-</span></div>
  <div><span class="kv">规划耗时</span><span class="val" id="timeTxt">-</span></div>
  <div id="cmdBox"></div>
  <button id="goalBtn" class="on">设定目标点</button>
  <button id="startBtn" class="sec">设定起点</button>
  <button id="resetBtn" class="sec">重置视角</button>
  <div class="hint" id="hint">左键拖拽旋转 · 右键拖拽平移 · 滚轮缩放 · 单击地面发布目标点</div>
</div>
<div id="legend">
  <div><span class="sw" style="background:#2f9e6e"></span>可通行 (代价低)</div>
  <div><span class="sw" style="background:#e07b39"></span>接近障碍 (代价高)</div>
  <div><span class="sw" style="background:#96282d"></span>不可通行 (碰撞/低净空)</div>
  <div><span class="sw" style="background:#6e7684"></span>障碍点云</div>
  <div><span class="sw" style="background:#ff3b6b"></span>全局路径</div>
</div>
<script>
const DATA = __DATA__;
const SERVER_MODE = __SERVER_MODE__;

function decode(b64, Type){
  const bin = atob(b64); const buf = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) buf[i]=bin.charCodeAt(i);
  return new Type(buf.buffer);
}
const POS = decode(DATA.pos, Float32Array);
const COL = decode(DATA.col, Uint8Array);
let PATH = DATA.path ? decode(DATA.path, Float32Array) : null;
let startPt = DATA.start, goalPt = DATA.goal, pickMode = "goal";

/* ---------------- 矩阵工具 (列主序) ---------------- */
function mul(a,b){const o=new Float32Array(16);
  for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;}
function perspective(fovy,asp,n,f){const t=1/Math.tan(fovy/2);const o=new Float32Array(16);
  o[0]=t/asp;o[5]=t;o[10]=(f+n)/(n-f);o[11]=-1;o[14]=2*f*n/(n-f);return o;}
function lookAt(eye,ctr,up){
  const z=norm(sub(eye,ctr)), x=norm(cross(up,z)), y=cross(z,x);const o=new Float32Array(16);
  o[0]=x[0];o[4]=x[1];o[8]=x[2];o[1]=y[0];o[5]=y[1];o[9]=y[2];o[2]=z[0];o[6]=z[1];o[10]=z[2];
  o[12]=-dot(x,eye);o[13]=-dot(y,eye);o[14]=-dot(z,eye);o[15]=1;return o;}
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const scale=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];};

/* ---------------- 相机 ---------------- */
const cam = {target: DATA.center.slice(), dist: Math.max(DATA.extent*0.85, 3),
             az: -Math.PI*0.62, el: 0.55};
function resetCam(){ cam.target=DATA.center.slice(); cam.dist=Math.max(DATA.extent*0.85,3);
  cam.az=-Math.PI*0.62; cam.el=0.55; draw(); }
function eyePos(){ const ce=Math.cos(cam.el);
  return add(cam.target, [cam.dist*ce*Math.cos(cam.az), cam.dist*ce*Math.sin(cam.az),
                          cam.dist*Math.sin(cam.el)]); }

/* ---------------- WebGL ---------------- */
const canvas=document.getElementById("gl");
const gl=canvas.getContext("webgl",{antialias:true,alpha:false});
if(!gl){document.body.innerHTML="<p style='padding:20px'>浏览器不支持 WebGL</p>";}
const VS=`attribute vec3 aPos;attribute vec3 aCol;uniform mat4 uMVP;uniform float uSize;
uniform float uProj;varying vec3 vCol;void main(){gl_Position=uMVP*vec4(aPos,1.0);vCol=aCol;
gl_PointSize=clamp(uSize*uProj/max(gl_Position.w,0.001),1.5,40.0);}`;
const FS=`precision mediump float;varying vec3 vCol;uniform float uRound;
void main(){ if(uRound>0.5){vec2 d=gl_PointCoord-vec2(0.5);if(dot(d,d)>0.25) discard;}
gl_FragColor=vec4(vCol,1.0);}`;
function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s); return s;}
const prog=gl.createProgram();
gl.attachShader(prog,shader(gl.VERTEX_SHADER,VS));gl.attachShader(prog,shader(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(prog);gl.useProgram(prog);
const aPos=gl.getAttribLocation(prog,"aPos"), aCol=gl.getAttribLocation(prog,"aCol");
const uMVP=gl.getUniformLocation(prog,"uMVP"), uSize=gl.getUniformLocation(prog,"uSize"),
      uRound=gl.getUniformLocation(prog,"uRound"), uProj=gl.getUniformLocation(prog,"uProj");
gl.enable(gl.DEPTH_TEST); gl.clearColor(0.051,0.063,0.09,1);

function buffer(data){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);return b;}
const cloudPos=buffer(POS);
const cloudCol=buffer(new Float32Array(Array.from(COL,v=>v/255)));
let pathBuf=null, pathCol=null, pathN=0;
function setPath(arr){
  PATH=arr; pathN = arr ? arr.length/3 : 0;
  if(!arr){pathBuf=null;return;}
  pathBuf=buffer(arr);
  const c=new Float32Array(pathN*3);
  for(let i=0;i<pathN;i++){c[i*3]=1.0;c[i*3+1]=0.23;c[i*3+2]=0.42;}
  pathCol=buffer(c);
}
if(PATH) setPath(PATH);
let markBuf=null, markCol=null, markN=0;
function setMarkers(){
  const pts=[],cols=[];
  if(startPt){pts.push(...startPt);cols.push(0.23,0.65,1.0);}
  if(goalPt){pts.push(...goalPt);cols.push(1.0,0.82,0.4);}
  markN=pts.length/3;
  if(markN){markBuf=buffer(new Float32Array(pts));markCol=buffer(new Float32Array(cols));}
}
/* 包围盒线框, 提供空间参照 */
const bboxLines=(()=>{const [mn,mx]=DATA.bbox;const c=[[mn[0],mn[1],mn[2]],[mx[0],mn[1],mn[2]],
  [mx[0],mx[1],mn[2]],[mn[0],mx[1],mn[2]],[mn[0],mn[1],mx[2]],[mx[0],mn[1],mx[2]],
  [mx[0],mx[1],mx[2]],[mn[0],mx[1],mx[2]]];
  const e=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const a=[];e.forEach(([i,j])=>{a.push(...c[i],...c[j]);});return new Float32Array(a);})();
const boxBuf=buffer(bboxLines);
const boxCol=buffer(new Float32Array(Array(bboxLines.length).fill(0.17)));

function bind(posB,colB){
  gl.bindBuffer(gl.ARRAY_BUFFER,posB);gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,colB);gl.enableVertexAttribArray(aCol);
  gl.vertexAttribPointer(aCol,3,gl.FLOAT,false,0,0);
}
let MVP=null;
function draw(){
  const dpr=Math.min(window.devicePixelRatio||1,2);
  const w=canvas.clientWidth,h=canvas.clientHeight;
  if(canvas.width!==w*dpr||canvas.height!==h*dpr){canvas.width=w*dpr;canvas.height=h*dpr;}
  gl.viewport(0,0,canvas.width,canvas.height);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  const P=perspective(Math.PI/4, w/h, Math.max(cam.dist*0.002,0.05), cam.dist*20);
  MVP=mul(P, lookAt(eyePos(), cam.target, [0,0,1]));
  gl.uniformMatrix4fv(uMVP,false,MVP);
  gl.uniform1f(uProj, canvas.height*0.5/Math.tan(Math.PI/8));   // 世界尺寸 -> 屏幕像素
  gl.uniform1f(uRound,0.0);
  bind(boxBuf,boxCol); gl.uniform1f(uSize,0.02); gl.drawArrays(gl.LINES,0,bboxLines.length/3);
  bind(cloudPos,cloudCol); gl.uniform1f(uSize,0.075); gl.drawArrays(gl.POINTS,0,DATA.n_total);
  gl.uniform1f(uRound,1.0);
  if(pathBuf){ bind(pathBuf,pathCol); gl.uniform1f(uSize,0.16);
               gl.drawArrays(gl.LINE_STRIP,0,pathN); gl.drawArrays(gl.POINTS,0,pathN); }
  if(markN){ bind(markBuf,markCol); gl.uniform1f(uSize,0.5); gl.drawArrays(gl.POINTS,0,markN); }
}

/* ---------------- 交互 ---------------- */
let drag=null, moved=0;
canvas.addEventListener("contextmenu",e=>e.preventDefault());
canvas.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,btn:e.button,shift:e.shiftKey};moved=0;});
window.addEventListener("mouseup",e=>{
  if(drag && moved<5 && drag.btn===0) pick(e.clientX,e.clientY);
  drag=null;});
window.addEventListener("mousemove",e=>{
  if(!drag) return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  moved+=Math.abs(dx)+Math.abs(dy);
  drag.x=e.clientX;drag.y=e.clientY;
  if(drag.btn===2||drag.shift){
    const eye=eyePos(); const fwd=norm(sub(cam.target,eye));
    const right=norm(cross(fwd,[0,0,1])); const up=cross(right,fwd);
    const k=cam.dist*0.0016;
    cam.target=add(cam.target, add(scale(right,-dx*k), scale(up,dy*k)));
  }else{
    cam.az-=dx*0.006; cam.el=Math.max(-1.5,Math.min(1.5,cam.el+dy*0.006));
  }
  draw();});
canvas.addEventListener("wheel",e=>{e.preventDefault();
  cam.dist=Math.max(0.6,Math.min(DATA.extent*4, cam.dist*Math.exp(e.deltaY*0.0012)));draw();},
  {passive:false});
window.addEventListener("resize",draw);

/* 屏幕空间拾取: 只在可通行面(前 n_ground 个点)里找 */
function pick(cx,cy){
  const rect=canvas.getBoundingClientRect();
  const px=(cx-rect.left), py=(cy-rect.top);
  let best=-1,bestD=1e9,bestW=1e9;
  for(let i=0;i<DATA.n_ground;i++){
    const x=POS[i*3],y=POS[i*3+1],z=POS[i*3+2];
    const w=MVP[3]*x+MVP[7]*y+MVP[11]*z+MVP[15];
    if(w<=0.001) continue;
    const sx=((MVP[0]*x+MVP[4]*y+MVP[8]*z+MVP[12])/w*0.5+0.5)*rect.width;
    const sy=(0.5-(MVP[1]*x+MVP[5]*y+MVP[9]*z+MVP[13])/w*0.5)*rect.height;
    const d=(sx-px)*(sx-px)+(sy-py)*(sy-py);
    if(d>144) continue;                     // 12 px 以内
    if(d<bestD-25 || (Math.abs(d-bestD)<=25 && w<bestW)){best=i;bestD=d;bestW=w;}
  }
  if(best<0){setHint("这里没有可通行的点, 请点在绿色/黄色区域上");return;}
  const p=[POS[best*3],POS[best*3+1],POS[best*3+2]];
  if(pickMode==="start"){startPt=p;}else{goalPt=p;}
  setMarkers();draw();updateInfo(null);requestPlan();
}

const $=id=>document.getElementById(id);
const fmt=p=>p?`(${p[0].toFixed(2)}, ${p[1].toFixed(2)}, ${p[2].toFixed(2)})`:"-";
function setHint(t){$("hint").textContent=t;}
function updateInfo(res){
  $("startTxt").textContent=fmt(startPt); $("goalTxt").textContent=fmt(goalPt);
  $("lenTxt").textContent = res&&res.length!=null ? res.length.toFixed(2)+" m" : "-";
  $("climbTxt").textContent = res&&res.climb!=null ? res.climb.toFixed(2)+" m" : "-";
  $("timeTxt").textContent = res&&res.planning_time!=null ?
      (res.planning_time*1000).toFixed(0)+" ms" : "-";
  if(!SERVER_MODE && goalPt && startPt){
    $("cmdBox").innerHTML="<code>python -m dddmr_py plan "+DATA.map_name+
      " --start "+startPt.map(v=>v.toFixed(2)).join(" ")+
      " --goal "+goalPt.map(v=>v.toFixed(2)).join(" ")+" --viz out.html</code>";
  }
}
async function requestPlan(){
  if(!SERVER_MODE){ setHint("离线模式: 已生成对应命令行, 复制执行即可重新规划"); return; }
  if(!startPt||!goalPt){ return; }
  setHint("规划中…");
  try{
    const r=await fetch("/plan",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({start:startPt,goal:goalPt})});
    const res=await r.json();
    if(res.success){
      const n=res.path.x.length; const arr=new Float32Array(n*3);
      for(let i=0;i<n;i++){arr[i*3]=res.path.x[i];arr[i*3+1]=res.path.y[i];arr[i*3+2]=res.path.z[i];}
      setPath(arr); draw(); updateInfo(res);
      setHint("规划成功 ("+n+" 个位姿), 继续单击可发布新的目标点");
    }else{ setPath(null); draw(); updateInfo(res); setHint("规划失败: "+res.message); }
  }catch(e){ setHint("请求失败: "+e); }
}
$("goalBtn").onclick=()=>{pickMode="goal";$("goalBtn").className="on";$("startBtn").className="sec";
  setHint("单击可通行面设定目标点");};
$("startBtn").onclick=()=>{pickMode="start";$("startBtn").className="on";$("goalBtn").className="sec";
  setHint("单击可通行面设定起点");};
$("resetBtn").onclick=resetCam;

setMarkers(); draw(); updateInfo(DATA.info);
</script>
</body>
</html>
"""


def render_html_string(gmap: GroundMap, path: Optional[Path] = None, start=None, goal=None,
                       map_name: str = "map.pcd", server_mode: bool = False) -> str:
    data = build_scene_data(gmap, path, start, goal, map_name)
    return (_HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
                 .replace("__SERVER_MODE__", "true" if server_mode else "false"))


def render_html(gmap: GroundMap, path: Optional[Path] = None, out_path: str = "path.html",
                start=None, goal=None, map_name: str = "map.pcd",
                server_mode: bool = False) -> str:
    """生成自包含的可视化 HTML 文件, 返回文件路径."""
    html = render_html_string(gmap, path, start, goal, map_name, server_mode)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path

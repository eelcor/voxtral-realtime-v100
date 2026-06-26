'use strict';
const $ = (id) => document.getElementById(id);
const recBtn = $('recBtn'), micIcon = $('micIcon'), stopIcon = $('stopIcon');
const transcriptEl = $('transcript'), countsEl = $('counts');
const statusDot = $('statusDot'), statusText = $('statusText'), statusLine = $('statusLine');
const timerEl = $('timer'), glow = $('glow'), levelI = document.querySelector('.levelbar i');
const copyBtn = $('copyBtn'), dlBtn = $('dlBtn'), clearBtn = $('clearBtn');
const micSelect = $('micSelect'), viz = $('viz');
const fxEnhance = $('fxEnhance'), fxGate = $('fxGate'), clipLed = $('clipLed');

let recording = false, ws = null, wsReady = false;
let audioCtx = null, micStream = null, worklet = null, analyser = null, srcNode = null, chainNodes = [];
let rafId = null, tStart = 0, timerInt = null;
let committed = '', live = '';
let freqData = null;

// signal-processing opties
let enhance = true, gateOn = false;
let noiseFloor = 0, gateOpenUntil = 0, clipTimer = null;

/* ---------- status & toasts ---------- */
function setStatus(state, text){ statusDot.className = 'dot ' + (state||''); statusText.textContent = text; }
function toast(msg, isErr){
  const t = document.createElement('div');
  t.className = 'toast' + (isErr ? ' err' : ''); t.textContent = msg;
  $('toasts').appendChild(t);
  setTimeout(() => { t.style.opacity='0'; t.style.transition='opacity .3s'; setTimeout(()=>t.remove(),300); }, 2600);
}

/* ---------- backend health & websocket ---------- */
async function checkHealth(){
  try{ const r = await fetch('/healthz');
    if(r.ok){ setStatus('ok','model gereed'); return true; }
    setStatus('bad','model offline'); return false;
  }catch{ setStatus('bad','server onbereikbaar'); return false; }
}
function connectWS(){
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { wsReady = true; };
  ws.onclose = () => { wsReady = false; setTimeout(connectWS, 1500); };
  ws.onerror = () => {};
  ws.onmessage = (ev) => {
    let m; try{ m = JSON.parse(ev.data); }catch{ return; }
    if(m.type === 'ready'){ statusLine.textContent = 'Aan het luisteren…'; }
    else if(m.type === 'delta'){ live += m.text; renderTranscript(); }
    else if(m.type === 'segment_done'){
      const seg = (m.text || live).trim();
      if(seg) committed += (committed ? ' ' : '') + seg;
      live = ''; renderTranscript();
    }
    else if(m.type === 'error'){ toast(m.msg || 'Fout', true); }
  };
}

/* ---------- transcript ---------- */
function renderTranscript(){
  transcriptEl.innerHTML = '';
  if(committed) transcriptEl.appendChild(document.createTextNode(committed + (live ? ' ' : '')));
  if(live){ const s = document.createElement('span'); s.className='live'; s.textContent=live; transcriptEl.appendChild(s); }
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  const full = (committed + ' ' + live).trim();
  const words = full ? full.split(/\s+/).length : 0;
  countsEl.textContent = `${words} woorden · ${full.length} tekens`;
  copyBtn.disabled = dlBtn.disabled = clearBtn.disabled = !full.length;
}

/* ---------- mics ---------- */
async function listMics(){
  try{
    const mics = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
    micSelect.innerHTML = '';
    mics.forEach((d,i) => { const o=document.createElement('option'); o.value=d.deviceId; o.textContent=d.label||`Microfoon ${i+1}`; micSelect.appendChild(o); });
    if(!mics.length){ const o=document.createElement('option'); o.textContent='Geen microfoon'; micSelect.appendChild(o); }
  }catch{}
}

/* ---------- recording + signal processing ---------- */
function flashClip(){ clipLed.classList.add('hot'); clearTimeout(clipTimer); clipTimer = setTimeout(()=>clipLed.classList.remove('hot'), 450); }

function onFrame(buf){
  if(!(ws && wsReady && recording)) return;
  const a = new Int16Array(buf);
  let sum = 0, peak = 0;
  for(let i=0;i<a.length;i++){ const v = a[i]; sum += v*v; const av = v<0?-v:v; if(av>peak) peak=av; }
  const rms = Math.sqrt(sum / a.length) / 32768;
  if(peak > 32200) flashClip();
  if(gateOn){
    // adaptief ruisvloer-volgen tijdens lage energie
    if(noiseFloor === 0 || rms < noiseFloor*2.5) noiseFloor = noiseFloor ? 0.97*noiseFloor + 0.03*rms : rms;
    const thr = Math.max(0.006, noiseFloor*4);
    const now = performance.now();
    if(rms > thr) gateOpenUntil = now + 300;          // open + 300ms hold
    if(now > gateOpenUntil){ ws.send(new Int16Array(a.length).buffer); return; }  // dicht -> stilte
  }
  ws.send(buf);
}

async function start(){
  if(!wsReady){ toast('Nog geen verbinding met de server', true); return; }
  try{
    const deviceId = micSelect.value || undefined;
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { deviceId: deviceId ? { exact: deviceId } : undefined,
               channelCount:1, echoCancellation:true, noiseSuppression:true, autoGainControl:true }
    });
  }catch(e){ toast('Geen toegang tot de microfoon: ' + e.message, true); return; }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.audioWorklet.addModule('/static/recorder-worklet.js');
  srcNode = audioCtx.createMediaStreamSource(micStream);
  analyser = audioCtx.createAnalyser(); analyser.fftSize = 512; analyser.smoothingTimeConstant = 0.72;
  freqData = new Uint8Array(analyser.frequencyBinCount);
  worklet = new AudioWorkletNode(audioCtx, 'recorder-processor');
  const silent = audioCtx.createGain(); silent.gain.value = 0;

  // verwerkingsketen (optioneel): highpass -> notch(50) -> lowpass(AA) -> compressor
  chainNodes = [];
  let node = srcNode;
  if(enhance){
    const hp = audioCtx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=80; hp.Q.value=0.707;
    const notch = audioCtx.createBiquadFilter(); notch.type='notch'; notch.frequency.value=50; notch.Q.value=8;
    const lp = audioCtx.createBiquadFilter(); lp.type='lowpass'; lp.frequency.value=7500; lp.Q.value=0.707;
    const comp = audioCtx.createDynamicsCompressor();
    comp.threshold.value=-30; comp.knee.value=20; comp.ratio.value=3; comp.attack.value=0.005; comp.release.value=0.2;
    node.connect(hp); hp.connect(notch); notch.connect(lp); lp.connect(comp);
    chainNodes = [hp, notch, lp, comp]; node = comp;
  }
  node.connect(analyser);
  node.connect(worklet); worklet.connect(silent); silent.connect(audioCtx.destination);
  worklet.port.onmessage = (e) => onFrame(e.data);

  noiseFloor = 0; gateOpenUntil = 0;
  ws.send(JSON.stringify({ type:'start' }));
  recording = true;
  document.body.classList.add('recording');
  micIcon.style.display='none'; stopIcon.style.display='';
  recBtn.title = 'Stoppen (spatie)'; statusLine.textContent = 'Verbinden…';
  fxEnhance.disabled = true;
  tStart = Date.now(); timerInt = setInterval(updateTimer, 250); updateTimer();
  if(live){ committed += (committed ? ' ' : '') + live.trim(); live=''; }
  drawLoop();
}

function stop(){
  recording = false;
  document.body.classList.remove('recording');
  micIcon.style.display=''; stopIcon.style.display='none';
  recBtn.title = 'Opnemen (spatie)'; statusLine.textContent = 'Klaar om op te nemen';
  fxEnhance.disabled = false;
  if(ws && wsReady) ws.send(JSON.stringify({ type:'stop' }));
  if(timerInt){ clearInterval(timerInt); timerInt=null; }
  if(rafId){ cancelAnimationFrame(rafId); rafId=null; }
  glow.style.setProperty('--vol',0); document.querySelector('.levelbar').style.setProperty('--vol',0);
  try{ worklet&&worklet.disconnect(); chainNodes.forEach(n=>n.disconnect()); srcNode&&srcNode.disconnect(); analyser&&analyser.disconnect(); }catch{}
  if(micStream){ micStream.getTracks().forEach(t=>t.stop()); micStream=null; }
  if(audioCtx){ audioCtx.close().catch(()=>{}); audioCtx=null; }
  drawIdle();
}

function updateTimer(){
  const s = Math.floor((Date.now()-tStart)/1000);
  timerEl.textContent = String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
}

/* ---------- visualizer + spraakband-meter ---------- */
function fitCanvas(){ const dpr=window.devicePixelRatio||1; viz.width=viz.clientWidth*dpr; viz.height=viz.clientHeight*dpr; }
function drawLoop(){
  if(!analyser){ return; }
  analyser.getByteFrequencyData(freqData);
  const binHz = (audioCtx.sampleRate/2) / freqData.length;
  // spraakband-energie (200..3800 Hz) voor de volumemeter
  let lo = Math.floor(200/binHz), hi = Math.min(freqData.length-1, Math.ceil(3800/binHz));
  let s=0; for(let i=lo;i<=hi;i++) s += freqData[i];
  const vol = Math.min(1, (s/(hi-lo+1))/95);
  glow.style.setProperty('--vol', vol.toFixed(3));
  document.querySelector('.levelbar').style.setProperty('--vol', vol.toFixed(3));
  // displaybalken: groepeer de onderste ~9kHz in 48 bars
  const BARS=48, span=Math.min(freqData.length, Math.ceil(9000/binHz)), per=Math.max(1,Math.floor(span/BARS));
  const disp=new Uint8Array(BARS);
  for(let b=0;b<BARS;b++){ let m=0; for(let k=0;k<per;k++){ const idx=b*per+k; if(idx<freqData.length&&freqData[idx]>m) m=freqData[idx]; } disp[b]=m; }
  drawBars(disp);
  rafId = requestAnimationFrame(drawLoop);
}
function drawBars(data){
  const ctx=viz.getContext('2d'), W=viz.width, H=viz.height, dpr=window.devicePixelRatio||1;
  ctx.clearRect(0,0,W,H);
  const n=data?data.length:48, bw=W/n;
  const grad=ctx.createLinearGradient(0,0,W,0);
  grad.addColorStop(0,'#6366f1'); grad.addColorStop(.5,'#a855f7'); grad.addColorStop(1,'#ec4899');
  ctx.fillStyle=grad;
  for(let i=0;i<n;i++){
    const v=data?data[i]/255:0.04+0.02*Math.sin(i*0.5);
    const h=Math.max(2*dpr, v*H*0.92), x=i*bw+bw*0.18, w=bw*0.64, y=(H-h)/2, r=Math.min(w/2,5*dpr);
    ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x,y,w,h,r) : ctx.rect(x,y,w,h); ctx.fill();
  }
}
function drawIdle(){ fitCanvas(); drawBars(null); }

/* ---------- actions ---------- */
recBtn.addEventListener('click', () => recording ? stop() : start());
document.addEventListener('keydown', (e) => {
  if(e.code==='Space' && e.target.tagName!=='SELECT' && e.target.tagName!=='BUTTON' && !e.target.isContentEditable){
    e.preventDefault(); recording ? stop() : start();
  }
});
fxEnhance.addEventListener('click', () => { enhance=!enhance; fxEnhance.classList.toggle('on',enhance); toast('Ruisfilter '+(enhance?'aan':'uit')); });
fxGate.addEventListener('click', () => { gateOn=!gateOn; fxGate.classList.toggle('on',gateOn); noiseFloor=0; toast('Stiltepoort '+(gateOn?'aan':'uit')); });
copyBtn.addEventListener('click', async () => {
  try{ await navigator.clipboard.writeText((committed+' '+live).trim()); toast('Gekopieerd naar klembord'); }
  catch{ toast('Kopiëren mislukt', true); }
});
dlBtn.addEventListener('click', () => {
  const blob=new Blob([(committed+' '+live).trim()],{type:'text/plain'});
  const a=document.createElement('a'), ts=new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
  a.href=URL.createObjectURL(blob); a.download=`transcriptie-${ts}.txt`; a.click(); URL.revokeObjectURL(a.href); toast('Bestand gedownload');
});
clearBtn.addEventListener('click', () => { committed=''; live=''; renderTranscript(); toast('Transcriptie gewist'); });
$('themeBtn').addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme')==='dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next); localStorage.setItem('voxtral-theme', next); drawIdle();
});
micSelect.addEventListener('change', () => { if(recording){ stop(); setTimeout(start,200); } });
window.addEventListener('resize', () => { if(!recording) drawIdle(); else fitCanvas(); });

/* ---------- init ---------- */
(async function init(){
  const savedTheme = localStorage.getItem('voxtral-theme');
  if(savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);
  fitCanvas(); drawIdle(); renderTranscript();
  connectWS(); await checkHealth(); setInterval(checkHealth, 8000);
  await listMics();
  navigator.mediaDevices.addEventListener?.('devicechange', listMics);
})();

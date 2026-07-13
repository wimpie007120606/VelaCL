"use client";
import {Fragment,useEffect,useMemo,useState} from "react";

const API=process.env.NEXT_PUBLIC_API_URL||(process.env.NODE_ENV==="development"?"http://localhost:8000":"");
const BASE_PATH=process.env.NEXT_PUBLIC_BASE_PATH||"";
const TABS=["Stream Monitor","Continual Learning","Active Learning","Model Registry","Safety","Serving"];
type Bundle={summary:any,experiments:Record<string,any>};
const pct=(v:number)=>`${(100*v).toFixed(1)}%`;

function Heatmap({artifact}:{artifact:any}) {
  const matrix=artifact.forgetting_matrix as number[][];
  return <div className="heatmap"><div></div>{matrix[0]?.map((_:number,i:number)=><b key={i}>S{i}</b>)}{matrix.map((row:number[],i:number)=><Fragment key={`r${i}`}><b>Task {i}</b>{row.map((v,j)=><span key={`${i}-${j}`} title={v.toFixed(3)} style={{background:`rgba(76,201,176,${.12+.88*v})`}}>{v.toFixed(2)}</span>)}</Fragment>)}</div>
}
function Bars({values}:{values:Record<string,number>}) {return <div className="bars">{Object.entries(values).map(([k,v])=><div key={k}><label>{k}<em>{pct(v)}</em></label><span><i style={{width:pct(v)}}/></span></div>)}</div>}

export default function Home(){
 const [tab,setTab]=useState(TABS[1]); const [data,setData]=useState<Bundle|null>(null); const [error,setError]=useState("");
 useEffect(()=>{
  const source=API?`${API}/v1/experiments`:`${BASE_PATH}/experiments.json`;
  fetch(source).then(r=>{if(!r.ok)throw Error("Experiment data is unavailable");return r.json()}).then(setData).catch(e=>setError(e.message))
 },[]);
 const active=data?.experiments?.active_balanced_replay; const latest=active?.stages?.at(-1)?.aggregate;
 const events=useMemo(()=>active?.annotation_queue||[],[active]);
 return <main><header><div><small>VELA / CONTINUAL INTELLIGENCE</small><h1>Research Console</h1></div><span className="live">● MEASURED RUNS</span></header><nav>{TABS.map(t=><button className={tab===t?"on":""} onClick={()=>setTab(t)} key={t}>{t}</button>)}</nav>
 {!data?<section className="empty"><h2>{error||"Loading experiment artifacts…"}</h2><p>The dashboard never substitutes hard-coded metrics.</p></section>:<>
 {tab==="Stream Monitor"&&<section><h2>Stream monitor</h2><div className="cards"><article><small>STAGES</small><strong>{active.stages.length}</strong></article><article><small>LANGUAGES</small><strong>{Object.keys(latest.by_language).length}</strong></article><article><small>DATASET HASH</small><code>{active.dataset_sha256.slice(0,12)}</code></article></div><div className="panel"><h3>Latest language performance</h3><Bars values={latest.by_language}/></div></section>}
 {tab==="Continual Learning"&&<section><h2>Continual learning</h2><div className="cards">{Object.values(data.experiments).map((x:any)=><article key={x.method}><small>{x.method.replaceAll("_"," ")}</small><strong>{pct(x.continual.average_accuracy)}</strong><p>forgetting {pct(x.continual.average_forgetting)}</p></article>)}</div><div className="panel"><h3>Active balanced replay · forgetting matrix</h3><p>Rows are stream tasks; columns are the model after each training stage.</p><Heatmap artifact={active}/></div></section>}
 {tab==="Active Learning"&&<section><h2>Annotation queue</h2><div className="cards"><article><small>SELECTED</small><strong>{events.length}</strong></article><article><small>BUDGET / STAGE</small><strong>8</strong></article><article><small>PENDING</small><strong>{events.filter((x:any)=>x.annotation_status==="pending").length}</strong></article></div><div className="queue">{events.slice(0,12).map((x:any)=><article key={x.event_id}><div><span>{x.language}</span><span>{x.intent}</span><span className={x.privacy_risk?"warn":""}>{x.privacy_risk?"privacy":"standard"}</span></div><p>{x.text}</p><footer>score {x.score.toFixed(3)} · {x.selection_reason}</footer></article>)}</div></section>}
 {tab==="Model Registry"&&<section><h2>Model registry</h2><div className="panel"><p className="champion">CHAMPION · {data.summary.registry.aliases.champion}</p>{data.summary.registry.models.map((x:any)=><div className="model" key={x.method}><b>{x.method}</b><span>accuracy {pct(x.metrics.average_accuracy)}</span><span>forgetting {pct(x.metrics.average_forgetting)}</span></div>)}</div></section>}
 {tab==="Safety"&&<section><h2>Safety posture</h2><div className="notice"><b>Evaluation not claimed in V1</b><p>The stream records privacy and safety flags, and serving detects possible PII and prompt injection. A representative multilingual safety corpus is required before reporting unsafe-response or over-refusal rates.</p></div><div className="panel"><h3>Risk-prioritized selections</h3><strong>{events.filter((x:any)=>x.safety_importance>0).length}</strong> examples selected with safety importance.</div></section>}
 {tab==="Serving"&&<section><h2>Serving performance</h2><div className="notice"><b>No invented load-test results</b><p>Prometheus exposes live request count and inference latency. Run the documented benchmark on deployment hardware to populate latency percentiles and throughput.</p></div><div className="panel"><code>{API?`GET ${API}/metrics`:"Static research dashboard · inference API deployment pending"}</code><p>Batched inference (1–32 texts), SSE streaming, version selection, health/readiness, validation, warnings, and rate limiting are implemented in the repository.</p></div></section>}
 </>}</main>
}

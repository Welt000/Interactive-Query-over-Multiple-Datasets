"""Server-only CarGurus paired SharedQ and four SUR experiments."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse,csv,json,platform,subprocess,sys,time,zipfile,hashlib,signal
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
DATA=ROOT/'data'
SEED=2026102603
METHODS=['SharedQ','SUR_HD-PI','SUR_RH','SUR_UH-Simplex','SUR_UH-Random']
ATTRS=['price','horsepower','mileage','year']

def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_points(p,x,elapsed=None):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        f.write(f'{len(x)} {x.shape[1]}\n')
        if elapsed is not None:f.write(f'{elapsed:.17g}\n')
        np.savetxt(f,x,fmt='%.17g')

def prepare():
    import preprocess_skyline as sky
    from baseline.experiment_utils import load_dataset
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'manifest.json').exists(),'Already prepared'
    source=ROOT/'source'/'us-used-cars-dataset.zip'
    while not source.exists():
        print('Waiting for source download',flush=True);time.sleep(10)
    chunks=[];raw_rows=0
    with zipfile.ZipFile(source) as z:
        name=next(n for n in z.namelist() if n.endswith('used_cars_data.csv'))
        with z.open(name) as f:
            for chunk in pd.read_csv(f,usecols=ATTRS+['listing_color','dealer_zip','vin'],dtype={'dealer_zip':'str','vin':'str'},chunksize=100000,low_memory=False):
                raw_rows+=len(chunk)
                for col in ATTRS:chunk[col]=pd.to_numeric(chunk[col],errors='coerce')
                mask=np.isfinite(chunk[ATTRS]).all(axis=1)&chunk[ATTRS].gt(0).all(axis=1)
                mask &= chunk.year.eq(chunk.year.round())&chunk.year.between(1886,2021)
                mask &= chunk.dealer_zip.str.fullmatch(r'\d{5}(?:-\d{4})?',na=False)
                v=chunk.loc[mask,ATTRS+['dealer_zip','vin']].copy();v['group']=v.dealer_zip.str[0]
                chunks.append(v);print('READ',raw_rows,'VALID',sum(len(x) for x in chunks),flush=True)
    df=pd.concat(chunks,ignore_index=True);valid_rows=len(df)
    df=df.drop_duplicates(['vin'])
    vin_unique=len(df);df=df.drop_duplicates(['group']+ATTRS)
    lo=df[ATTRS].min().to_numpy();hi=df[ATTRS].max().to_numpy();assert (hi>lo).all()
    rng=np.random.default_rng(SEED);utilities={str(d):rng.dirichlet(np.ones(d),10).tolist() for d in (2,3,4)}
    manifest=dict(source_url='https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset',archive_sha256=digest(source),
        archive_member=name,raw_rows=raw_rows,valid_rows=valid_rows,unique_vins=vin_unique,unique_4d_within_groups=len(df),
        attributes=ATTRS,directions=['min','max','min','max'],minimum=lo.tolist(),maximum=hi.tolist(),
        normalization='Global linear min-max on cleaned 4D mother data; invert price and mileage; deduplicate within groups after projection',
        cleaning='All four numeric attributes finite and positive; integer year 1886..2021 (September 2020 source); valid US five-digit dealer ZIP; deduplicate VIN then group+attributes; no sampling, synthetic copies, logs or clipping',
        grouping='First digit of dealer ZIP; groups 0..9; natural sorted group order',seed=SEED,utilities=utilities,
        methods=METHODS,dimensions={},server=platform.node(),python=sys.version,numpy=np.__version__,pandas=pd.__version__,
        workers=4,blas_threads=1,timeout_seconds=3600,rho=.1,alpha=1,beta=1,sur_resample_timeouts=False,
        timing='All measurements on this server. SUR total includes summed server skyline time, SharedQ uses full data and charges internal preparation. Disk loading and independent verification excluded.')
    sky.EPS=0.0
    for d in (2,3,4):
        entries=[]
        for group in map(str,range(10)):
            raw=df.loc[df.group==group,ATTRS].to_numpy();assert len(raw)>0
            x=(raw-lo)/(hi-lo);x[:,0]=1-x[:,0];x[:,2]=1-x[:,2];x=np.unique(x[:,:d],axis=0)
            path=DATA/f'{d}d'/f'zip_{group}.txt';write_points(path,x)
            assert np.array_equal(load_dataset(path),x)
            t=time.perf_counter();ids=sky.skyline_indices_serial(x);elapsed=time.perf_counter()-t;front=x[ids]
            sp=DATA/f'{d}d_skyline'/path.name;write_points(sp,front,elapsed)
            us=np.asarray(utilities[str(d)]);assert np.max(np.abs(np.max(x@us.T,axis=0)-np.max(front@us.T,axis=0)))<1e-12
            entries.append(dict(group=group,rows=len(x),path=str(path),sha256=digest(path),skyline_path=str(sp),skyline_rows=len(front),skyline_elapsed=elapsed,skyline_sha256=digest(sp)))
            print('PREP',d,group,len(x),len(front),elapsed,flush=True)
        manifest['dimensions'][str(d)]=dict(attributes=ATTRS[:d],total_points=sum(e['rows'] for e in entries),datasets=entries)
    with zipfile.ZipFile(OUT/'source_code.zip','w',zipfile.ZIP_DEFLATED) as z:
        for directory in ('SharedQ','SUR','HD_PI','RH','UH_Random','UH_Simplex','structure','baseline'):
            for p in (ROOT/directory).rglob('*.py'):z.write(p,p.relative_to(ROOT))
        for name in ('experiment_config.py','preprocess_skyline.py','cargurus_benchmark.py'):z.write(ROOT/name,name)
    manifest['source_code_sha256']=digest(OUT/'source_code.zip')
    save(OUT/'manifest.json',manifest);save(OUT/'utilities.json',utilities)
    (OUT/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    summarize()

def worker(d,method,uid):
    from experiment_config import ExperimentConfig,run_one_experiment_job
    from baseline.experiment_final import radius_from_rho
    from baseline.experiment_utils import load_dataset
    manifest=json.loads((OUT/'manifest.json').read_text());u=np.asarray(manifest['utilities'][str(d)][uid])
    dest=OUT/f'{d}d'/method;dest.mkdir(parents=True,exist_ok=True);detail=dest/f'u{uid:02d}_detail.csv'
    cfg=ExperimentConfig(source_dataset_dir=str(DATA/f'{d}d'),skyline_dataset_dir=str(DATA/f'{d}d_skyline'),dimension=d,
        methods=('SUR','SharedQ'),seed=SEED,parallel_u_workers=1,random_utility_count=0,utilities_txt=','.join(map(str,u)),
        sur_resample_timeouts=False,sharedq_detect_workers=1,sharedq_detect_point_timeout_seconds=None)
    class Collector:
        payload=None
        def put(self,v):self.payload=v
    q=Collector();shared=method=='SharedQ'
    run_one_experiment_job(cfg,'SharedQ' if shared else 'SUR','HD-PI' if shared else method[4:],uid,u,.1 if shared else None,
        radius_from_rho(.1,d) if shared else None,1. if shared else None,detail,q)
    status,payload=q.payload
    value=dict(dimension=d,method=method,utility_id=uid,utility=u.tolist(),status=status,config=asdict(cfg))
    if status=='ok':
        value['summaries']=[asdict(s) for s in payload];checks=[]
        rows=list(csv.DictReader(detail.open()));assert len(rows)==10 and {int(r['dataset_id']) for r in rows}==set(range(10))
        for r in rows:
            idx=int(r['dataset_id']);e=manifest['dimensions'][str(d)]['datasets'][idx]
            raw=load_dataset(Path(e['path']));used=raw if shared else load_dataset(Path(e['skyline_path']))
            ri=int(r['result_index']);assert 0<=ri<len(used)
            score=float(used[ri]@u);truth=float(np.max(raw@u));gap=truth-score
            checks.append(dict(dataset_id=idx,score=score,truth_score=truth,gap=gap,correct=abs(gap)<=1e-9))
        value['verification']=checks
        if not all(c['correct'] for c in checks):value['status']='incorrect'
    else:value['error']=payload
    save(dest/f'u{uid:02d}.json',value)

def summarize():
    manifest=json.loads((OUT/'manifest.json').read_text());records=[];runs=[]
    for d in (2,3,4):
        for m in METHODS:
            values=[json.loads(p.read_text()) for uid in range(10) if (p:=OUT/f'{d}d'/m/f'u{uid:02d}.json').exists()]
            ok=[v for v in values if v['status']=='ok'];ss=[v['summaries'][0] for v in ok]
            prep=0. if m=='SharedQ' else sum(e['skyline_elapsed'] for e in manifest['dimensions'][str(d)]['datasets'])
            q=float(np.mean([s['avg_total_questions_per_utility'] for s in ss])) if ok else None
            elapsed=float(np.mean([s['avg_total_elapsed_per_utility'] for s in ss])) if ok else None
            records.append(dict(dimension=d,method=m,attempted=len(values),successful=len(ok),timeouts=sum(v['status']=='timeout' for v in values),
                errors=sum(v['status'] not in ('ok','timeout') for v in values),questions=q,elapsed=elapsed,search_elapsed=elapsed-prep if ok else None,skyline_elapsed=prep))
            for v in values:
                r={k:v[k] for k in ('dimension','method','utility_id','status')}
                if v['status']=='ok':r.update(questions=v['summaries'][0]['avg_total_questions_per_utility'],elapsed=v['summaries'][0]['avg_total_elapsed_per_utility'])
                runs.append(r)
    save(OUT/'summary.json',records)
    for name,rows in (('summary.csv',records),('per_utility.csv',runs)):
        if rows:
            keys=list(dict.fromkeys(k for r in rows for k in r))
            with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    return records

def run():
    pending=[(d,m,u) for d in (2,3,4) for u in range(10) for m in METHODS if not (OUT/f'{d}d'/m/f'u{u:02d}.json').exists()];active=[]
    while pending or active:
        while pending and len(active)<4:
            d,m,u=pending.pop(0);dest=OUT/f'{d}d'/m;dest.mkdir(parents=True,exist_ok=True)
            log=(dest/f'u{u:02d}.log').open('w')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--dimension',str(d),'--method',m,'--utility',str(u)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append((p,time.monotonic(),d,m,u,log));print('START',d,m,u,'PID',p.pid,flush=True)
        for item in active[:]:
            p,t,d,m,u,log=item;expired=time.monotonic()-t>3600
            if p.poll() is not None or expired:
                if expired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                log.close();path=OUT/f'{d}d'/m/f'u{u:02d}.json'
                if not path.exists():save(path,dict(dimension=d,method=m,utility_id=u,status='timeout' if expired else 'error',returncode=p.returncode))
                print('END',d,m,u,json.loads(path.read_text())['status'],'WALL',round(time.monotonic()-t,2),flush=True)
                active.remove(item);summarize()
        time.sleep(.5)
    audit()

def audit():
    from baseline.experiment_utils import load_dataset
    manifest=json.loads((OUT/'manifest.json').read_text());assert digest(OUT/'source_code.zip')==manifest['source_code_sha256']
    with zipfile.ZipFile(OUT/'source_code.zip') as z:
        for n in z.namelist():assert (ROOT/n).read_bytes()==z.read(n),n
    verified=[];counts={}
    for d in (2,3,4):
        us=np.asarray(manifest['utilities'][str(d)])
        for idx,e in enumerate(manifest['dimensions'][str(d)]['datasets']):
            assert digest(Path(e['path']))==e['sha256'] and digest(Path(e['skyline_path']))==e['skyline_sha256']
            raw=load_dataset(Path(e['path']));front=load_dataset(Path(e['skyline_path']));truth=np.max(raw@us.T,axis=0)
            for m in METHODS:
                for uid in range(10):
                    p=OUT/f'{d}d'/m/f'u{uid:02d}.json';v=json.loads(p.read_text())
                    if idx==0:counts[v['status']]=counts.get(v['status'],0)+1
                    if v['status']!='ok':continue
                    assert v['utility']==us[uid].tolist()
                    rows=list(csv.DictReader((p.parent/f'u{uid:02d}_detail.csv').open()))
                    r=next(r for r in rows if int(r['dataset_id'])==idx);used=raw if m=='SharedQ' else front
                    score=float(used[int(r['result_index'])]@us[uid]);gap=float(truth[uid]-score);assert abs(gap)<=1e-9
                    assert abs(score-float(r['score']))<=1e-12 and abs(truth[uid]-float(r['truth_score']))<=1e-12
                    verified.append(dict(dimension=d,method=m,utility_id=uid,dataset_id=idx,truth=float(truth[uid]),score=score,gap=gap))
    assert sum(counts.values())==150
    if verified:
        with (OUT/'verified_answers.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(verified[0]));w.writeheader();w.writerows(verified)
    save(OUT/'audit.json',dict(planned_runs=150,status_counts=counts,verified_answers=len(verified),max_abs_score_gap=max((abs(v['gap']) for v in verified),default=None),data_and_source_hashes_match=True,server=platform.node()))
    records=summarize()
    lines=['# CarGurus 真实数据服务器实验','',f"原始 {manifest['raw_rows']:,} 条；有效 {manifest['valid_rows']:,} 条；去重VIN {manifest['unique_vins']:,} 辆；四维组内去重 {manifest['unique_4d_within_groups']:,} 点。",
        '来源：Kaggle ananaymital/us-used-cars-dataset，2020年9月CarGurus库存。按dealer_zip首位分10个组。',
        '2D价格（小优）、马力（大优）；3D加入里程（小优）；4D加入年份（大优）。全部数据组共同线性Min–Max，无抽样、人工副本、log或截断。',
        '清洗：四属性有限且正，年份为1886至2021的整数，合法五位美国邮编；先按VIN去重，再按组+属性去重，投影后再次组内去重。',
        f'每维10个固定Dirichlet(1)效用，seed={SEED}，五种方法共用；单u联合10组。SharedQ rho=0.1, alpha=beta=1, detect_workers=1；每u硬限3600秒，超时不换u。',
        '4个任务并行，各BLAS/OMP/MKL单线程。SUR总时间含服务器精确Skyline预处理，SharedQ用完整数据并计入内部预处理；磁盘加载及独立核验不计时。所有时间均来自本次服务器运行。',
        f'状态：{counts}；独立核验答案 {len(verified)} 个。','', '|维度|投影去重点数|Skyline点数|','|---|---:|---:|']
    for d in (2,3,4):
        info=manifest['dimensions'][str(d)];lines.append(f"|{d}|{info['total_points']:,}|{sum(e['skyline_rows'] for e in info['datasets']):,}|")
    lines+=['','|维度|方法|成功/10|超时|错误|平均问题数|总时间(s)|搜索时间(s)|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        fmt=lambda x:'—' if x is None else f'{x:.6f}'
        lines.append(f"|{r['dimension']}|{r['method']}|{r['successful']}/10|{r['timeouts']}|{r['errors']}|{fmt(r['questions'])}|{fmt(r['elapsed'])}|{fmt(r['search_elapsed'])}|")
    lines+=['','如存在失败或超时，成功样本均值不构成完整配对比较。数据来源、哈希、清洗统计、归一化极值、效用和运行环境见manifest.json；代码快照source_code.zip；每u日志、JSON及每组明细均保留。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('COMPLETE',counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker','audit']);p.add_argument('--dimension',type=int);p.add_argument('--method');p.add_argument('--utility',type=int);a=p.parse_args()
    if a.action=='worker':worker(a.dimension,a.method,a.utility)
    else:globals()[a.action]()

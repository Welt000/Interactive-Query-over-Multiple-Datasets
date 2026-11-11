"""Fresh server-only, nine-method Realtor experiment; never mixes local timings."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, csv, json, platform, subprocess, sys, time, zipfile, traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import real_estate_benchmark as b
from baseline.experiment_utils import load_dataset

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'original_datasets'/'Realtor_real_125m_20261026'
OUT=ROOT/'results'/'Realtor_server_fresh_9methods_20261026'
METHODS=['SharedQ']+[prefix+a for prefix in ('SUR_','Baseline_') for a in ('HD-PI','RH','UH-Simplex','UH-Random')]
SEED=2026102602

def setup():
    import scipy, pandas, preprocess_skyline as skyline
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists(): raise RuntimeError('Existing manifest: use run to resume, not setup')
    original=json.loads((ROOT/'transfer_manifest.json').read_text(encoding='utf-8'))
    old=original['utilities']
    rng=np.random.default_rng(SEED)
    utilities={str(d):rng.dirichlet(np.ones(d),size=10).tolist() for d in (2,3,4)}
    for d in (2,3,4):
        for u in utilities[str(d)]:
            assert not any(np.allclose(u,v,atol=1e-12,rtol=0) for v in old[str(d)])
    manifest={k:v for k,v in original.items() if k not in ('utilities','seed','dimensions','machine','python','numpy','pandas','scipy','methods','source_code_sha256')}
    manifest.update(utilities=utilities,previous_utilities=old,seed=SEED,dimensions={},methods=METHODS,
        server=platform.node(),machine=platform.platform(),python=sys.version,numpy=np.__version__,pandas=pandas.__version__,scipy=scipy.__version__,
        workers=4,timeout_seconds=3600,blas_threads=1,preprocessing='Skyline recomputed on this server, charged once per joint utility for SUR and Baseline',
        baseline='Each subset searched independently; joint questions/time are sums over all ten subsets',
        prior_local_results_reused=False)
    for d in (5,2,3,4):
        entries=[]
        for previous in original['dimensions'][str(d)]['datasets']:
            name=Path(previous['path'].replace('\\','/')).name
            path=DATA/f'{d}d'/name
            assert b.digest(path)==previous['sha256'],str(path)
            points=load_dataset(path)
            assert points.shape==(previous['rows'],d)
            entry={k:previous[k] for k in ('group','rows','sha256')}
            entry['path']=str(path)
            if d!=5:
                skyline.EPS=0.0
                t=time.perf_counter();ids=skyline.skyline_indices_serial(points);elapsed=time.perf_counter()-t
                front=points[ids]
                sky=DATA/f'{d}d_skyline'/name
                b.write_points(sky,front,elapsed)
                assert np.max(np.abs(np.max(points@np.asarray(utilities[str(d)]).T,axis=0)-np.max(front@np.asarray(utilities[str(d)]).T,axis=0)))<1e-12
                entry.update(skyline_path=str(sky),skyline_rows=len(front),skyline_elapsed=elapsed,skyline_sha256=b.digest(sky))
            entries.append(entry)
            print(f'SERVER PREP {d}d {name} rows={len(points)} skyline={entry.get("skyline_rows",0)}',flush=True)
        manifest['dimensions'][str(d)]={'attributes':original['dimensions'][str(d)]['attributes'],'total_points':sum(x['rows'] for x in entries),'datasets':entries}
    snapshot=OUT/'source_code.zip'
    with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as z:
        for directory in ('SharedQ','SUR','HD_PI','RH','UH_Random','UH_Simplex','structure','baseline'):
            for p in (ROOT/directory).rglob('*.py'): z.write(p,p.relative_to(ROOT))
        for name in ('experiment_config.py','preprocess_skyline.py','real_estate_benchmark.py','server_real_estate_benchmark.py'):
            z.write(ROOT/name,name)
    manifest['source_code_sha256']=b.digest(snapshot)
    b.save(OUT/'manifest.json',manifest);b.save(OUT/'utilities.json',utilities)
    (OUT/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    summarize()

def worker(d,method,uid):
    from experiment_config import ExperimentConfig,run_one_experiment_job
    from baseline.experiment_final import radius_from_rho
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    u=np.asarray(manifest['utilities'][str(d)][uid])
    dest=OUT/f'{d}d'/method;dest.mkdir(parents=True,exist_ok=True)
    cfg=ExperimentConfig(source_dataset_dir=str(DATA/f'{d}d'),skyline_dataset_dir=str(DATA/f'{d}d_skyline'),
        dimension=d,methods=('Baseline','SUR','SharedQ'),seed=SEED,parallel_u_workers=1,
        random_utility_count=0,utilities_txt=','.join(map(str,u)),sur_resample_timeouts=False,
        sharedq_detect_workers=1,sharedq_detect_point_timeout_seconds=None)
    shared=method=='SharedQ'
    kind,algorithm=('SharedQ','HD-PI') if shared else method.split('_',1)
    class Collector:
        payload=None
        def put(self,v): self.payload=v
    q=Collector();detail=dest/f'u{uid:02d}_detail.csv'
    run_one_experiment_job(cfg,kind,algorithm,uid,u,0.1 if shared else None,
        radius_from_rho(0.1,d) if shared else None,1.0 if shared else None,detail,q)
    status,payload=q.payload
    value={'dimension':d,'method':method,'utility_id':uid,'utility':u.tolist(),'status':status,'config':asdict(cfg)}
    if status=='ok':
        value['summaries']=[asdict(s) for s in payload]
        rows=list(csv.DictReader(detail.open(encoding='utf-8')))
        assert len(rows)==10 and {int(r['dataset_id']) for r in rows}==set(range(10))
        checks=[]
        for r in rows:
            idx=int(r['dataset_id']);e=manifest['dimensions'][str(d)]['datasets'][idx]
            raw=load_dataset(Path(e['path']));used=raw if shared else load_dataset(Path(e['skyline_path']))
            ri=int(r['result_index']);assert 0<=ri<len(used)
            score=float(used[ri]@u);truth=float(np.max(raw@u));gap=truth-score
            checks.append(dict(dataset_id=idx,score=score,truth_score=truth,gap=gap,correct=abs(gap)<=1e-9))
        value['verification']=checks
        if not all(c['correct'] for c in checks): value['status']='incorrect'
    else: value['error']=payload
    b.save(dest/f'u{uid:02d}.json',value)

def summarize():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    records=[];runs=[]
    for d in (2,3,4):
        sky=sum(e['skyline_elapsed'] for e in manifest['dimensions'][str(d)]['datasets'])
        for method in METHODS:
            values=[json.loads(p.read_text(encoding='utf-8')) for i in range(10) if (p:=OUT/f'{d}d'/method/f'u{i:02d}.json').exists()]
            ok=[v for v in values if v['status']=='ok']
            sums=[v['summaries'][0] for v in ok]
            prep=0.0 if method=='SharedQ' else sky
            q=float(np.mean([s['avg_total_questions_per_utility'] for s in sums])) if ok else None
            elapsed=float(np.mean([s['avg_total_elapsed_per_utility'] for s in sums])) if ok else None
            records.append(dict(dimension=d,method=method,attempted=len(values),successful=len(ok),
                timeouts=sum(v['status']=='timeout' for v in values),errors=sum(v['status'] not in ('ok','timeout') for v in values),
                questions=q,elapsed=elapsed,search_elapsed=elapsed-prep if ok else None,skyline_elapsed=prep))
            for v in values:
                r={k:v[k] for k in ('dimension','method','utility_id','status')}
                if v['status']=='ok':
                    s=v['summaries'][0];r.update(questions=s['avg_total_questions_per_utility'],elapsed=s['avg_total_elapsed_per_utility'],
                        search_elapsed=s['avg_total_elapsed_per_utility']-prep,skyline_elapsed=prep)
                runs.append(r)
    b.save(OUT/'summary.json',records)
    for name,rows in (('summary.csv',records),('per_utility.csv',runs)):
        if rows:
            keys=list(dict.fromkeys(k for r in rows for k in r))
            with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    return records

def run():
    pending=[(d,u,m) for d in (2,3,4) for u in range(10) for m in METHODS if not (OUT/f'{d}d'/m/f'u{u:02d}.json').exists()]
    active=[]
    while pending or active:
        while pending and len(active)<4:
            d,u,m=pending.pop(0);dest=OUT/f'{d}d'/m;dest.mkdir(parents=True,exist_ok=True)
            log=(dest/f'u{u:02d}.log').open('w',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--dimension',str(d),'--method',m,'--utility',str(u)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append((p,time.monotonic(),d,u,m,log));print(f'START {d}d {m} u{u:02d} pid={p.pid}',flush=True)
        for item in active[:]:
            p,t,d,u,m,log=item;expired=time.monotonic()-t>3600
            if p.poll() is not None or expired:
                if expired:
                    import signal
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
                log.close();path=OUT/f'{d}d'/m/f'u{u:02d}.json'
                if not path.exists(): b.save(path,dict(dimension=d,method=m,utility_id=u,status='timeout' if expired else 'error',returncode=p.returncode))
                print(f'END {d}d {m} u{u:02d} {json.loads(path.read_text())["status"]} wall={time.monotonic()-t:.2f}',flush=True)
                active.remove(item);summarize()
        time.sleep(.5)
    audit()

def audit():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert b.digest(OUT/'source_code.zip')==manifest['source_code_sha256']
    with zipfile.ZipFile(OUT/'source_code.zip') as z:
        for name in z.namelist(): assert (ROOT/name).read_bytes()==z.read(name),name
    verified=[];status_counts={}
    for d in (2,3,4):
        utilities=np.asarray(manifest['utilities'][str(d)])
        for idx,e in enumerate(manifest['dimensions'][str(d)]['datasets']):
            assert b.digest(Path(e['path']))==e['sha256']
            assert b.digest(Path(e['skyline_path']))==e['skyline_sha256']
            raw=load_dataset(Path(e['path']));front=load_dataset(Path(e['skyline_path']))
            truth=np.max(raw@utilities.T,axis=0)
            for m in METHODS:
                for uid in range(10):
                    p=OUT/f'{d}d'/m/f'u{uid:02d}.json';v=json.loads(p.read_text(encoding='utf-8'))
                    if idx==0: status_counts[v['status']]=status_counts.get(v['status'],0)+1
                    assert v.get('utility',utilities[uid].tolist())==utilities[uid].tolist()
                    if v['status']!='ok': continue
                    rows=list(csv.DictReader((p.parent/f'u{uid:02d}_detail.csv').open(encoding='utf-8')))
                    r=next(r for r in rows if int(r['dataset_id'])==idx)
                    used=raw if m=='SharedQ' else front;score=float(used[int(r['result_index'])]@utilities[uid]);gap=float(truth[uid]-score)
                    assert abs(gap)<=1e-9,(d,m,uid,idx,gap)
                    assert abs(score-float(r['score']))<=1e-12
                    assert abs(truth[uid]-float(r['truth_score']))<=1e-12
                    verified.append(dict(dimension=d,method=m,utility_id=uid,dataset_id=idx,truth=float(truth[uid]),score=score,gap=gap))
    for e in manifest['dimensions']['5']['datasets']: assert b.digest(Path(e['path']))==e['sha256']
    assert sum(status_counts.values())==270
    with (OUT/'verified_answers.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(verified[0]));w.writeheader();w.writerows(verified)
    records=summarize()
    b.save(OUT/'audit.json',dict(planned_runs=270,status_counts=status_counts,verified_answers=len(verified),
        max_abs_score_gap=max(abs(v['gap']) for v in verified),data_and_source_hashes_match=True,new_utilities_verified=True,
        previous_results_reused=False,server=platform.node()))
    lines=['# 房产服务器实验：新u，SharedQ、4个SUR、4个Baseline','',
        f'服务器：{platform.node()}；全部结果在本服务器新运行，未复用本地计时。每个维度10个新固定Dirichlet效用向量，seed={SEED}，与上一轮30个u逐维排重；同一维度9种方法共用同批u。',
        '数据沿用五维母数据1,254,996个组内去重点，分10个邮编组。2D价格/住宅面积，3D加入土地面积，4D加入浴室数。全局线性Min–Max、价格反向，无人工副本。',
        'Baseline每组独立查询，问题数和搜索时间按10组求和。SUR共享偏好范围。SharedQ用完整投影点集；SUR/Baseline用服务器重新计算的精确Skyline，总时间计入10组Skyline预处理；磁盘加载和独立核验不计时。',
        '4个独立进程并行，BLAS/OMP/MKL各1线程，SharedQ detect_workers=1；rho=0.1，alpha=beta=1；每u联合10组硬限3600秒，超时不更换u。',
        f'状态：{status_counts}；独立重新计算{len(verified)}个答案，最大得分误差{max(abs(v["gap"]) for v in verified):.3g}。',
        '', '|维度|方法|成功/10|平均问题数|含预处理时间(s)|搜索时间(s)|','|---|---|---:|---:|---:|---:|']
    for r in records:
        fmt=lambda x:'—' if x is None else f'{x:.6f}'
        lines.append(f'|{r["dimension"]}|{r["method"]}|{r["successful"]}/10|{fmt(r["questions"])}|{fmt(r["elapsed"])}|{fmt(r["search_elapsed"])}|')
    lines.extend(['','计时差异需结合Skyline预处理与搜索时间分别解读；成功样本均值如遇超时/失败不可视为完整配对结果。原始数据保留极端值及历史已售记录，清洗与归一化规则和上一轮一致。',
        'manifest.json保存数据哈希、服务器环境、旧u和新u；pip_freeze.txt保存依赖；source_code.zip保存本轮代码快照；各方法目录保存逐u日志、JSON与逐组明细。'])
    (OUT/'REPORT.md').write_text('\n\n'.join(lines[:2])+'\n'+'\n'.join(lines[2:])+'\n',encoding='utf-8')
    print('COMPLETE '+json.dumps(status_counts),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','run','worker','audit']);p.add_argument('--dimension',type=int);p.add_argument('--method');p.add_argument('--utility',type=int);a=p.parse_args()
    if a.action=='setup': setup()
    elif a.action=='run': run()
    elif a.action=='audit': audit()
    else: worker(a.dimension,a.method,a.utility)

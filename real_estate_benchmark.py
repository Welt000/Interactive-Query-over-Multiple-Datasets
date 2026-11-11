"""Reproducible real-only Realtor preparation and paired SharedQ/SUR benchmark."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, csv, hashlib, json, platform, subprocess, sys, time, traceback, zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'Realtor_real_125m_20261026'
DATA = ROOT / 'original_datasets' / 'Realtor_real_125m_20261026'
ATTRS = ['price', 'house_size', 'acre_lot', 'bath', 'bed']
METHODS = ['SharedQ', 'SUR_HD-PI', 'SUR_RH', 'SUR_UH-Simplex', 'SUR_UH-Random']

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def write_points(path, points, elapsed=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        f.write(f'{len(points)} {points.shape[1]}\n')
        if elapsed is not None: f.write(f'{elapsed:.17g}\n')
        np.savetxt(f, points, fmt='%.17g')

def prepare():
    import requests
    import preprocess_skyline as skyline
    from baseline.experiment_utils import load_dataset
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / 'source' / 'usa-real-estate-dataset.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://www.kaggle.com/api/v1/datasets/download/ahmedshahriarsakib/usa-real-estate-dataset'
    if not archive.exists():
        print('Downloading original Kaggle archive', flush=True)
        with requests.get(url, stream=True, timeout=(30,120)) as r:
            r.raise_for_status()
            with archive.with_suffix('.part').open('wb') as f:
                for block in r.iter_content(1024*1024): f.write(block)
        archive.with_suffix('.part').replace(archive)
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith('realtor-data.zip.csv'))
        source = archive.parent / 'realtor-data.zip.csv'
        with z.open(name) as f, source.open('wb') as g:
            import shutil
            shutil.copyfileobj(f,g)
    df = pd.read_csv(source, usecols=ATTRS+['zip_code','status'])
    mask = np.isfinite(df[ATTRS]).all(axis=1) & df[ATTRS].gt(0).all(axis=1)
    mask &= df.zip_code.notna() & df.zip_code.between(0,99999) & df.zip_code.eq(df.zip_code.round())
    valid = df.loc[mask].copy()
    valid['group'] = valid.zip_code.astype(int).astype(str).str.zfill(5).str[0]
    unique = valid.drop_duplicates(['group']+ATTRS)
    lo, hi = unique[ATTRS].min().to_numpy(), unique[ATTRS].max().to_numpy()
    assert (hi>lo).all()
    rng = np.random.default_rng(2026102601)
    utilities = {str(d): rng.dirichlet(np.ones(d),size=10).tolist() for d in (2,3,4)}
    save(OUT/'utilities.json', utilities)
    manifest = {'source_url':url, 'source_sha256':digest(source), 'archive_sha256':digest(archive),
        'raw_rows':len(df), 'valid_rows':len(valid), 'unique_5d_within_groups':len(unique),
        'valid_status_counts':valid.status.value_counts().to_dict(),
        'attributes':ATTRS, 'directions':['min','max','max','max','max'],
        'normalization':'global linear min-max on the 5D mother dataset; invert price; no log, clipping, or synthetic copies',
        'minimum':lo.tolist(),'maximum':hi.tolist(), 'grouping':'first digit of zero-padded 5-digit ZIP; 0..9',
        'deduplication':'within each group; repeat after projection; cross-group identical vectors retained',
        'seed':2026102601,'utility_distribution':'Dirichlet(1,...,1)', 'utilities':utilities,
        'dimensions':{},'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
        'machine':platform.platform(), 'methods':METHODS,'timeout_seconds':3600,
        'workers':4,'blas_threads':1,'sur_preprocessing':'skyline; elapsed charged to every utility',
        'sharedq_preprocessing':'original projected data; algorithm internal preprocessing included',
        'source_matches_existing': digest(source)==digest(ROOT/'original_datasets'/'realtor-data.zip.csv')}
    for d in (5,2,3,4):
        entries=[]
        for group in map(str,range(10)):
            raw = unique.loc[unique.group==group, ATTRS].to_numpy()
            points = (raw-lo)/(hi-lo)
            points[:,0]=1-points[:,0]
            points = np.unique(points[:,:d],axis=0)
            folder=DATA/f'{d}d'
            path=folder/f'zip_{group}.txt'
            write_points(path, points)
            # Verify the exact serialized input consumed by the experiment.
            loaded=load_dataset(path)
            assert np.array_equal(loaded,points)
            entry={'group':group,'rows':len(points),'path':str(path),'sha256':digest(path)}
            if d != 5:
                # Exact dominance is required for small normalized coordinates.
                skyline.EPS=0.0
                t=time.perf_counter()
                ids=skyline.skyline_indices_serial(points)
                elapsed=time.perf_counter()-t
                front=points[ids]
                sky=DATA/f'{d}d_skyline'/path.name
                write_points(sky,front,elapsed)
                for u in np.asarray(utilities[str(d)]):
                    assert abs(np.max(points@u)-np.max(front@u)) <= 1e-12
                entry.update(skyline_path=str(sky),skyline_rows=len(front),skyline_elapsed=elapsed,skyline_sha256=digest(sky))
            entries.append(entry)
            print(f'Prepared {d}d ZIP {group}: {len(points)} -> {entry.get("skyline_rows","mother")}',flush=True)
        manifest['dimensions'][str(d)]={'attributes':ATTRS[:d], 'total_points':sum(e['rows'] for e in entries),'datasets':entries}
    import scipy
    manifest['scipy']=scipy.__version__
    snapshot=OUT/'source_code.zip'
    with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as z:
        for parent in ('SharedQ','SUR','HD_PI','RH','UH_Random','UH_Simplex','structure','baseline'):
            for p in (ROOT/parent).rglob('*.py'): z.write(p,p.relative_to(ROOT))
        for name in ('experiment_config.py','preprocess_skyline.py','real_estate_benchmark.py'):
            z.write(ROOT/name,name)
    manifest['source_code_sha256']=digest(snapshot)
    save(OUT/'manifest.json',manifest)
    print('Preparation complete',flush=True)

def worker(d,method,uid):
    from experiment_config import ExperimentConfig, run_one_experiment_job
    from baseline.experiment_final import radius_from_rho
    from baseline.experiment_utils import load_dataset
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    u=np.asarray(manifest['utilities'][str(d)][uid])
    dest=OUT/f'{d}d'/method
    detail=dest/f'u{uid:02d}_detail.csv'
    dest.mkdir(parents=True,exist_ok=True)
    config=ExperimentConfig(source_dataset_dir=str(DATA/f'{d}d'),skyline_dataset_dir=str(DATA/f'{d}d_skyline'),
        dimension=d,methods=('SUR','SharedQ'),seed=2026102601,parallel_u_workers=1,
        sur_resample_timeouts=False,sharedq_detect_workers=1,sharedq_detect_point_timeout_seconds=None)
    class Collector:
        payload=None
        def put(self,value): self.payload=value
    q=Collector()
    shared=method=='SharedQ'
    run_one_experiment_job(config,'SharedQ' if shared else 'SUR', 'HD-PI' if shared else method[4:],
        uid,u,0.1 if shared else None,radius_from_rho(0.1,d) if shared else None,1.0 if shared else None,detail,q)
    status,payload=q.payload
    result={'dimension':d,'method':method,'utility_id':uid,'utility':u.tolist(),'status':status,'config':asdict(config)}
    if status=='ok':
        result['summaries']=[asdict(row) for row in payload]
        rows=list(csv.DictReader(detail.open(encoding='utf-8')))
        assert len(rows)==10 and {int(r['dataset_id']) for r in rows}==set(range(10))
        verified=[]
        for row in rows:
            idx=int(row['dataset_id'])
            entry=manifest['dimensions'][str(d)]['datasets'][idx]
            raw=load_dataset(Path(entry['path']))
            used=raw if shared else load_dataset(Path(entry['skyline_path']))
            ri=int(row['result_index'])
            assert 0<=ri<len(used)
            score=float(used[ri]@u)
            truth=float(np.max(raw@u))
            gap=truth-score
            verified.append({'dataset_id':idx,'score':score,'raw_truth_score':truth,'score_gap':gap,'correct':abs(gap)<=1e-9})
        result['verification']=verified
        result['all_correct']=all(v['correct'] for v in verified)
        if not result['all_correct']: result['status']='incorrect'
    else: result['error']=payload
    save(dest/f'u{uid:02d}.json',result)

def summarize():
    records=[]
    for d in (2,3,4):
        for method in METHODS:
            files=[OUT/f'{d}d'/method/f'u{i:02d}.json' for i in range(10)]
            values=[json.loads(f.read_text(encoding='utf-8')) for f in files if f.exists()]
            ok=[v for v in values if v['status']=='ok']
            summaries=[v['summaries'][0] for v in ok]
            rec={'dimension':d,'method':method,'attempted':len(values),'successful':len(ok),
                'timeouts':sum(v['status']=='timeout' for v in values),'errors':sum(v['status'] not in ('ok','timeout') for v in values),
                'avg_questions':float(np.mean([s['avg_total_questions_per_utility'] for s in summaries])) if ok else None,
                'avg_elapsed':float(np.mean([s['avg_total_elapsed_per_utility'] for s in summaries])) if ok else None,
                'verified_answers':sum(len(v['verification']) for v in ok)}
            records.append(rec)
    save(OUT/'summary.json',records)
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    lines=['# 房产真实数据实验', '',
        f"原始 {manifest['raw_rows']:,} 条，清洗后 {manifest['valid_rows']:,} 条；按邮编首位分为10组，组内五维去重后 {manifest['unique_5d_within_groups']:,} 点。",
        '五维顺序：价格（小优）、住宅面积、土地面积、浴室数、卧室数（其余大优）。2D/3D/4D使用该顺序前2/3/4列。',
        '采用全部组共同的线性Min–Max，价格反向；无人工副本，无log变换，无额外异常值删除。保留在售和已售记录。',
        '降维后组内重新去重；125万是五维母数据的组内去重总量，不是各降维版本的点数，也不是跨组全局去重数。',
        '每维10个固定Dirichlet效用，5种方法复用同批效用和组顺序。每次联合查询10组，硬限3600秒，超时不替换。',
        '并行4个独立效用进程，BLAS/OMP/MKL各1线程，SharedQ内部detect_workers=1。',
        'SharedQ: rho=0.1, alpha=beta=1；其余算法参数保存在逐次JSON config。',
        'SUR使用精确Skyline（无epsilon支配近似），时间包含10组Skyline预处理时间；SharedQ用完整点集，内部预处理计入算法时间。磁盘加载和独立答案核验不计入算法时间。',
        '所有成功结果均对完整投影点集重新计算最高效用与返回点得分，不仅依赖算法correct字段。',
        '原始数据存在极端面积、房间数等值，按上述既定规则保留；其对全局归一化和偏好结果的影响需在论文讨论。', '',
        '|维度|原始投影去重点数|Skyline点数|', '|---|---:|---:|']
    for d in (2,3,4):
        info=manifest['dimensions'][str(d)]
        lines.append(f"|{d}|{info['total_points']:,}|{sum(e['skyline_rows'] for e in info['datasets']):,}|")
    lines.extend(['','|维度|方法|完成/10|超时|错误|平均问题数|平均时间(s)|','|---|---|---:|---:|---:|---:|---:|'])
    for r in records:
        q='—' if r['avg_questions'] is None else f"{r['avg_questions']:.2f}"
        t='—' if r['avg_elapsed'] is None else f"{r['avg_elapsed']:.6f}"
        lines.append(f"|{r['dimension']}|{r['method']}|{r['successful']}/10|{r['timeouts']}|{r['errors']}|{q}|{t}|")
    lines.extend(['','均值仅覆盖成功运行；若有失败或超时，不能把成功样本均值视为完整配对比较。',
        '文件：manifest.json（数据哈希、统计、归一化、效用），source/（重新下载的原始压缩包及CSV），source_code.zip（代码快照），utilities.json，summary.csv；各维度/方法目录内有逐u JSON、日志和逐数据集明细。'])
    (OUT/'REPORT.md').write_text('\n\n'.join(lines[:2])+'\n'+'\n'.join(lines[2:])+'\n',encoding='utf-8')
    return records

def run(workers):
    pending=[(d,m,u) for d in (2,3,4) for u in range(10) for m in METHODS
             if not (OUT/f'{d}d'/m/f'u{u:02d}.json').exists()]
    active=[]
    while pending or active:
        while pending and len(active)<workers:
            d,m,u=pending.pop(0)
            folder=OUT/f'{d}d'/m;folder.mkdir(parents=True,exist_ok=True)
            log=(folder/f'u{u:02d}.log').open('w',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--dimension',str(d),'--method',m,'--utility',str(u)],stdout=log,stderr=subprocess.STDOUT)
            active.append((p,time.monotonic(),d,m,u,log))
            print(f'START {d}d {m} u{u:02d} pid={p.pid}',flush=True)
        for item in active[:]:
            p,t,d,m,u,log=item
            expired=time.monotonic()-t>3600
            if p.poll() is not None or expired:
                if expired: p.kill();p.wait()
                log.close()
                path=OUT/f'{d}d'/m/f'u{u:02d}.json'
                if not path.exists(): save(path,{'dimension':d,'method':m,'utility_id':u,'status':'timeout' if expired else 'error','returncode':p.returncode})
                status=json.loads(path.read_text(encoding='utf-8'))['status']
                print(f'END {d}d {m} u{u:02d}: {status} wall={time.monotonic()-t:.1f}s',flush=True)
                active.remove(item);summarize()
        time.sleep(0.5)
    print(json.dumps(summarize(),indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','worker','summarize'])
    parser.add_argument('--dimension',type=int)
    parser.add_argument('--method')
    parser.add_argument('--utility',type=int)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='worker': worker(args.dimension,args.method,args.utility)
    elif args.action=='run': run(args.workers)
    else: print(json.dumps(summarize(),indent=2))

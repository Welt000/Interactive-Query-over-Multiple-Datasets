"""Independent final integrity checks, paired exports, and comparison figures."""
from pathlib import Path
import csv, json, zipfile
import real_estate_benchmark as b
import numpy as np
import pandas as pd

def main():
    manifest=json.loads((b.OUT/'manifest.json').read_text(encoding='utf-8'))
    assert b.digest(b.OUT/'source'/'realtor-data.zip.csv')==manifest['source_sha256']
    assert b.digest(b.OUT/'source'/'usa-real-estate-dataset.zip')==manifest['archive_sha256']
    assert b.digest(b.OUT/'source_code.zip')==manifest['source_code_sha256']
    with zipfile.ZipFile(b.OUT/'source_code.zip') as z:
        for name in z.namelist():
            assert (b.ROOT/name).read_bytes()==z.read(name),name
    for info in manifest['dimensions'].values():
        for e in info['datasets']:
            assert b.digest(Path(e['path']))==e['sha256']
            if 'skyline_path' in e:
                assert b.digest(Path(e['skyline_path']))==e['skyline_sha256']
    data_rows=[]
    for d,info in manifest['dimensions'].items():
        for entry in info['datasets']:
            data_rows.append({'dimension':int(d),'zip_prefix':entry['group'],'points':entry['rows'],
                'skyline_points':entry.get('skyline_rows',''),'file':entry['path'],'sha256':entry['sha256']})
    pd.DataFrame(data_rows).to_csv(b.DATA/'dataset_summary.csv',index=False,encoding='utf-8-sig')
    (b.DATA/'README.md').write_text(
        '# Realtor real data: 10 groups, 5D mother dataset\n\n'
        'Source: https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset\n\n'
        'Downloaded original CSV SHA-256: '+manifest['source_sha256']+'\n\n'
        'Keep finite positive price, house_size, acre_lot, bath, bed and a nonmissing integer ZIP in [0,99999]. '
        'Retain both sold and for_sale. Group by first digit of five-digit ZIP. Remove equal numeric vectors within each group. '
        'No generated copies; no log, clipping, extra outlier filtering, or subsampling.\n\n'
        'Column order: price (min), house_size (max), acre_lot (max), bath (max), bed (max). '
        'Apply global linear min-max using the 5D mother dataset; invert normalized price. All output columns are larger-is-better. '
        '2D, 3D, 4D keep the first 2, 3, 4 columns respectively, with another within-group deduplication.\n\n'
        '5D mother total: 1,254,996 within-group distinct points. 2D: 1,095,759; 3D: 1,250,435; 4D: 1,253,105. '
        'Cross-group equal vectors remain separate. All dimensions have ZIP groups 0..9.\n\n'
        'TXT format: first line is N D, then N rows of coordinates. Skyline TXT adds preprocessing seconds as the second line. '
        'Coordinates are serialized with 17 significant digits. *_skyline folders are for SUR; SharedQ uses complete projected folders.\n\n'
        'Experiment evidence: '+str(b.OUT)+'\n\n'
        'Reproduce from code_1: python real_estate_benchmark.py prepare; python real_estate_benchmark.py run --workers 4; '
        'python audit_real_estate_benchmark.py. Existing result JSON files are resumed, never silently overwritten. '
        'For a new preparation/environment use new output paths to avoid mixing results.\n',encoding='utf-8')
    runs=[];answers=[];raw_validation=[]
    for d in (2,3,4):
        preprocess=sum(e['skyline_elapsed'] for e in manifest['dimensions'][str(d)]['datasets'])
        values={}
        for method in b.METHODS:
            for uid in range(10):
                p=b.OUT/f'{d}d'/method/f'u{uid:02d}.json'
                v=json.loads(p.read_text(encoding='utf-8'))
                assert (v['dimension'],v['method'],v['utility_id'])==(d,method,uid)
                assert v['utility']==manifest['utilities'][str(d)][uid]
                r={'dimension':d,'method':method,'utility_id':uid,'status':v['status']}
                if v['status']=='ok':
                    assert v['all_correct'] and len(v['verification'])==10
                    s=v['summaries'][0]
                    assert s['utilities']==1 and s['datasets']==10 and s['runs']==10
                    r.update(questions=s['avg_total_questions_per_utility'],elapsed_including_preprocess=s['avg_total_elapsed_per_utility'],
                        skyline_elapsed=0.0 if method=='SharedQ' else preprocess,
                        search_elapsed=s['avg_total_elapsed_per_utility']-(0.0 if method=='SharedQ' else preprocess))
                    rows=list(csv.DictReader((p.parent/f'u{uid:02d}_detail.csv').open(encoding='utf-8')))
                    assert len(rows)==10
                    for row in rows:
                        idx=int(row['dataset_id']);e=manifest['dimensions'][str(d)]['datasets'][idx]
                        answers.append(dict(dimension=d,method=method,utility_id=uid,**v['verification'][idx]))
                        # Store row references; independently reload each input once below.
                        values[(method,uid,idx)]=row
                runs.append(r)
        from baseline.experiment_utils import load_dataset
        for idx,e in enumerate(manifest['dimensions'][str(d)]['datasets']):
            raw=load_dataset(Path(e['path']))
            front=load_dataset(Path(e['skyline_path']))
            utilities=np.asarray(manifest['utilities'][str(d)])
            truths=np.max(raw@utilities.T,axis=0)
            for method in b.METHODS:
                for uid in range(10):
                    if (method,uid,idx) not in values: continue
                    row=values[(method,uid,idx)]
                    used=raw if method=='SharedQ' else front
                    score=float(used[int(row['result_index'])]@utilities[uid])
                    gap=float(truths[uid]-score)
                    assert abs(gap)<=1e-9,(d,method,uid,idx,gap)
                    assert abs(score-float(row['score']))<=1e-12
                    assert abs(truths[uid]-float(row['truth_score']))<=1e-12
                    raw_validation.append({'dimension':d,'method':method,'utility_id':uid,'dataset_id':idx,'raw_truth':float(truths[uid]),'score':score,'gap':gap})
    frame=pd.DataFrame(runs)
    assert len(frame)==150
    frame.to_csv(b.OUT/'per_utility.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(raw_validation).to_csv(b.OUT/'verified_answers.csv',index=False,encoding='utf-8-sig')
    paired=[]
    for d in (2,3,4):
        for m in b.METHODS[1:]:
            sub=frame[(frame.dimension==d)&(frame.method==m)&(frame.status=='ok')].set_index('utility_id')
            sq=frame[(frame.dimension==d)&(frame.method=='SharedQ')&(frame.status=='ok')].set_index('utility_id')
            ids=sorted(set(sub.index)&set(sq.index))
            for uid in ids:
                paired.append({'dimension':d,'sur':m,'utility_id':uid,
                    'sharedq_questions':sq.loc[uid,'questions'],'sur_questions':sub.loc[uid,'questions'],
                    'question_difference_sur_minus_sharedq':sub.loc[uid,'questions']-sq.loc[uid,'questions'],
                    'sharedq_elapsed':sq.loc[uid,'elapsed_including_preprocess'],
                    'sur_elapsed_including_skyline':sub.loc[uid,'elapsed_including_preprocess'],
                    'sur_search_elapsed':sub.loc[uid,'search_elapsed']})
    pd.DataFrame(paired).to_csv(b.OUT/'paired_comparison.csv',index=False,encoding='utf-8-sig')
    b.summarize()
    successful=frame[frame.status=='ok']
    agg=successful.groupby(['dimension','method'],sort=False)[['questions','elapsed_including_preprocess','search_elapsed','skyline_elapsed']].mean().reset_index()
    agg.to_csv(b.OUT/'comparison_with_timing_breakdown.csv',index=False,encoding='utf-8-sig')
    report=b.OUT/'REPORT.md'
    with report.open('a',encoding='utf-8') as f:
        f.write('\n## 独立审计与计时拆分\n\n')
        f.write(f'已复核150个计划运行的记录，成功{len(successful)}次，独立重新计算{len(raw_validation)}个答案；源文件、全部数据文件及运行代码快照哈希一致。\n\n')
        f.write('|维度|方法|平均问题数|平均总时间(s)|其中搜索时间(s)|其中Skyline(s)|\n|---|---|---:|---:|---:|---:|\n')
        for r in agg.to_dict('records'):
            f.write(f"|{r['dimension']}|{r['method']}|{r['questions']:.2f}|{r['elapsed_including_preprocess']:.6f}|{r['search_elapsed']:.6f}|{r['skyline_elapsed']:.6f}|\n")
        f.write('\n总时间口径包含SUR的Skyline预处理，因此应结合搜索时间列解读速度差异。独立审计不计入实验时间。\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    labels=['SharedQ','SUR HD-PI','SUR RH','SUR UH-Simplex','SUR UH-Random']
    colors=['#2674ad','#e3903b','#62a578','#ae79b5','#9b8d79']
    for ax,col,title in zip(axes,['questions','elapsed_including_preprocess','search_elapsed'],['Questions per utility','Time incl. preprocessing (s)','Search time (s)']):
        x=np.arange(3)
        width=0.15
        for k,m in enumerate(b.METHODS):
            vals=[float(agg[(agg.dimension==d)&(agg.method==m)][col].iloc[0]) if len(agg[(agg.dimension==d)&(agg.method==m)]) else np.nan for d in (2,3,4)]
            ax.bar(x+(k-2)*width,vals,width,label=labels[k],color=colors[k])
        ax.set_xticks(x,['2D','3D','4D']);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if col!='questions': ax.set_yscale('log')
    axes[0].legend(fontsize=8)
    fig.savefig(b.OUT/'comparison.png',dpi=180)
    fig.savefig(b.OUT/'comparison.pdf')
    b.save(b.OUT/'audit.json',{'planned_runs':150,'successful_runs':len(successful),'verified_answers':len(raw_validation),
        'max_abs_score_gap':max(abs(v['gap']) for v in raw_validation),'source_and_data_hashes_match':True,'code_snapshot_matches':True,
        'all_methods_use_same_utilities':True,'complete':len(frame)==150,'all_successful':len(successful)==150})
    print(agg.to_string(index=False))
    print('Final independent audit complete')

if __name__=='__main__': main()

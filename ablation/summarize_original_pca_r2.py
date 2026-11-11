"""Audit the forty completed original-partition ablation runs and make artifacts."""
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ablation_sharedq_original_pca_10u_20261025_r2'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.matplotlib'))
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baseline.experiment_utils import load_dataset
from run_ablation import dataset_matches

GROUPS = ['neither', 'pca_original', 'detect_divide_only', 'full']
NAMES = ['两项关闭', '仅PCA：原始全扫描', '仅detect-and-divide', '两项开启（full）']
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write_csv(name, items):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(items[0])); w.writeheader(); w.writerows(items)

m = read(OUT/'manifest.json')
u = np.asarray(m['utilities'])
assert u.shape == (10, 4) and np.all(u > 0) and np.allclose(u.sum(axis=1), 1)
assert len(np.unique(u, axis=0)) == 10
for prior in ['ablation_sharedq', 'ablation_sharedq_10u_20261024', 'ablation_sharedq_dual_pca_10u_20261025']:
    old = read(ROOT/'results'/prior/'manifest.json')
    assert all(not np.allclose(a,b,atol=1e-14,rtol=0) for a in u for b in old['utilities'])
records, rows, utility_rows = {}, [], []
for g in GROUPS:
    rs = [read(OUT/g/f'u{i}.json') for i in range(10)]
    for i,r in enumerate(rs):
        assert r['group'] == g and r['utility_id'] == i and r['status'] == 'ok'
        assert r['resolved'] and not r['reused'] and r['datasets'] == 10 and r['accuracy'] == 1
        assert len(r['details']) == 10 and all(d['correct'] and d['candidate_count']==1 for d in r['details'])
        assert r['elapsed'] > 0 and np.isfinite(r['elapsed'])
        utility_rows.append(dict(group=g,utility_id=i,questions=r['questions'],seconds=r['elapsed'],accuracy=r['accuracy']))
    records[g]=rs
    s=read(OUT/g/'summary.json')
    row=dict(group=g,successful=10,avg_questions=mean(r['questions'] for r in rs),
             avg_seconds=mean(r['elapsed'] for r in rs),accuracy=1.0,timeouts=0,failures=0,
             sd_questions=stdev(r['questions'] for r in rs),sd_seconds=stdev(r['elapsed'] for r in rs))
    assert np.isclose(row['avg_questions'],s['avg_questions_successful']) and np.isclose(row['avg_seconds'],s['avg_seconds_successful'])
    rows.append(row)

answer_rows=[]
for dataset_id,entry in enumerate(m['datasets']):
    path=ROOT/'original_datasets/4d_100k_10'/entry['name']
    assert dataset_matches(path, entry)
    data=load_dataset(path); assert list(data.shape)==entry['shape']
    scores=data @ u.T
    for g in GROUPS:
        for i,r in enumerate(records[g]):
            detail=next(d for d in r['details'] if d['dataset_id']==dataset_id)
            assert detail['dataset_name']==entry['name']
            gap=float(scores[:,i].max()-scores[detail['point_index'],i])
            assert gap<=m['parameters']['tolerance']
            assert scores[:,i].max()-scores[detail['truth_index'],i]<=m['parameters']['tolerance']
            answer_rows.append(dict(group=g,utility_id=i,dataset_id=dataset_id,dataset_name=entry['name'],
                                    point_index=detail['point_index'],score_gap=gap,correct=True))
    print('ANSWER_AUDIT',dataset_id,'40 answers OK',flush=True)
write_csv('verified_summary.csv',rows)
write_csv('per_utility.csv',utility_rows)
write_csv('verified_answers.csv',answer_rows)

paired=[]
for i in range(10):
    a=records['pca_original'][i];b=records['neither'][i]
    paired.append(dict(utility_id=i,pca_questions=a['questions'],neither_questions=b['questions'],pca_seconds=a['elapsed'],neither_seconds=b['elapsed'],pca_minus_neither_questions=a['questions']-b['questions'],pca_minus_neither_seconds=a['elapsed']-b['elapsed']))
write_csv('paired_pca_comparison.csv',paired)
fig,axes=plt.subplots(1,2,figsize=(10.5,4.6),layout='constrained')
colors=['#8a8f99','#5b91be','#65a38a','#af75af']
labels=['Neither','PCA\noriginal scan','Detect & Divide\nonly','Full']
for ax,field,title in [(axes[0],'avg_questions','Mean shared questions'),(axes[1],'avg_seconds','Mean algorithm time (seconds; log scale)')]:
    vals=[r[field] for r in rows];bars=ax.bar(range(4),vals,color=colors)
    ax.set_xticks(range(4),labels);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    if field=='avg_seconds':ax.set_yscale('log');ax.set_ylim(1,4000)
    else:ax.set_ylim(0,32)
    for bar,v in zip(bars,vals):ax.annotate(f'{v:.3f}' if field=='avg_seconds' else f'{v:.1f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
fig.suptitle('SharedQ ablation: PCA-only uses original exhaustive partitions\n10 fresh utilities; all groups 10/10 successful, 100% accuracy, no timeouts or failures',fontsize=11)
fig.savefig(OUT/'ablation_comparison.png',dpi=200);fig.savefig(OUT/'ablation_comparison.pdf');plt.close(fig)
by={r['group']:r for r in rows}
reduction=(1-by['pca_original']['avg_seconds']/by['neither']['avg_seconds'])*100
lines=['# SharedQ 四组消融：仅PCA使用原始全扫描分区（r2）','',
       '四组40次运行全部成功，400个数据集答案经原始点集独立核验全部正确，无超时、无失败。所有结果均来自本轮重新运行，未复用历史结果。','',
       '| 配置 | 成功u | 平均共享问题数 | 平均算法时间/秒 | 正确率 | 超时 | 失败 |','|---|---:|---:|---:|---:|---:|---:|']
for name,r in zip(NAMES,rows):lines.append(f"| {name} | 10/10 | {r['avg_questions']:.1f} | {r['avg_seconds']:.6f} | 100% | 0 | 0 |")
lines+=['','## 实验定义','',
       '10个新固定u：Dirichlet(1,1,1,1)，NumPy default_rng seed2026102502；已与之前30个u排重。四组使用同批u，各u联合查询相同10个100k×4原始数据集，cor:anti:ind=3:3:4。每u单独进程硬限3600秒，LQH环境，BLAS/OMP/MKL各1线程，detect_workers=1。',
       '时间取算法返回的elapsed，包含内部初始化与交互搜索，不包含独立数据文件加载及进程启动。问题数是跨10个数据集共享的提问数；每组正确率覆盖100个数据集答案。','',
       '- neither：PCA reduction及detect-and-divide均关闭；原始逐候选全扫描分区，禁用neighbor walk。',
       '- pca_original：仅PCA reduction开启，分区与neither相同，使用原始逐候选全扫描，禁用neighbor walk。',
       '- detect_divide_only：仅detect-and-divide开启；PCA关闭后保留全量跨数据集评分。',
       '- full：PCA reduction及detect-and-divide均开启。','',
       '## 同批u的观察','',
       f'- 固定原始全扫描分区后（pca_original vs neither），PCA开启使平均问题数20.4→20.6，算法耗时1171.477→946.644秒，耗时减少{reduction:.2f}%。',
       '- DD开启时（full vs detect_divide_only），PCA开启使平均问题数25.3→26.4，算法耗时7.671→8.118秒。',
       '本轮PCA reduction仍未减少平均问题数；其时间影响随是否开启DD而不同。以上仅描述本轮10个u及固定数据集组合。','',
       '## 核验与文件','',
       '已独立检查40个逐u记录及其均值、原始数据集SHA-256、与三批历史u排重。400个答案通过原始点集点积重新计算score_gap核验，并非仅采用服务器correct字段。仅汇总本轮四组。',
       '- verified_summary.csv：均值及样本标准差。',
       '- per_utility.csv：40次逐u问题数与算法时间。',
       '- paired_pca_comparison.csv：相同分区下PCA开关的逐u配对结果。',
       '- verified_answers.csv / validation.json：400个答案独立核验记录及审核摘要。',
       '- ablation_comparison.png / .pdf：四组问题数、耗时对比图。',
       '- manifest.json / utilities.json / source_hashes.sha256 / runtime.json：参数、效用和环境审计。',
       '- completed.json / server_run.log / 各组u*.json：服务器原始记录。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'validation.json').write_text(json.dumps(dict(validated_runs=40,independently_verified_answers=len(answer_rows),dataset_hashes_verified=True,new_utilities_distinct_from_previous_30=True,summary_recomputed=True),indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))

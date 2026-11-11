"""Audit the fifty completed dual-partition ablation runs and make artifacts."""
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ablation_sharedq_dual_pca_10u_20261025'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.matplotlib'))
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baseline.experiment_utils import load_dataset
from run_ablation import dataset_matches

GROUPS = ['neither', 'pca_original', 'pca_hdpi', 'detect_divide_only', 'full']
NAMES = ['两项关闭', '仅PCA：原始全扫描', '仅PCA：HDPI分区', '仅detect-and-divide', '两项开启（full）']
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write_csv(name, items):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(items[0])); w.writeheader(); w.writerows(items)

m = read(OUT/'manifest.json')
u = np.asarray(m['utilities'])
assert u.shape == (10, 4) and np.all(u > 0) and np.allclose(u.sum(axis=1), 1)
assert len(np.unique(u, axis=0)) == 10
for prior in ['ablation_sharedq', 'ablation_sharedq_10u_20261024']:
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
    print('ANSWER_AUDIT',dataset_id,'50 answers OK',flush=True)
write_csv('verified_summary.csv',rows)
write_csv('per_utility.csv',utility_rows)
write_csv('verified_answers.csv',answer_rows)
paired=[]
for i in range(10):
    a=records['pca_original'][i];b=records['pca_hdpi'][i];c=records['neither'][i]
    paired.append(dict(utility_id=i,pca_original_questions=a['questions'],pca_hdpi_questions=b['questions'],
                       neither_questions=c['questions'],pca_original_seconds=a['elapsed'],pca_hdpi_seconds=b['elapsed'],
                       neither_seconds=c['elapsed'],hdpi_minus_original_questions=b['questions']-a['questions'],
                       hdpi_minus_original_seconds=b['elapsed']-a['elapsed']))
write_csv('paired_partition_comparison.csv',paired)

fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
colors=['#8a8f99','#5b91be','#97bdce','#65a38a','#af75af']
labels=['Neither','PCA\noriginal scan','PCA\nHDPI','Detect & Divide\nonly','Full']
for ax,field,title in [(axes[0],'avg_questions','Mean shared questions'),(axes[1],'avg_seconds','Mean algorithm time (seconds; log scale)')]:
    vals=[r[field] for r in rows]; bars=ax.bar(range(5),vals,color=colors)
    ax.set_xticks(range(5),labels);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    if field=='avg_seconds': ax.set_yscale('log');ax.set_ylim(1,4000)
    else: ax.set_ylim(0,28)
    for bar,v in zip(bars,vals): ax.annotate(f'{v:.3f}' if field=='avg_seconds' else f'{v:.1f}',
        (bar.get_x()+bar.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
fig.suptitle('SharedQ ablation: 10 fresh utilities, 10 datasets per utility\nAll five groups: 10/10 successful, 100% accuracy, no timeouts or failures',fontsize=12)
fig.savefig(OUT/'ablation_comparison.png',dpi=200);fig.savefig(OUT/'ablation_comparison.pdf');plt.close(fig)
by={r['group']:r for r in rows}
percent_hdpi=(by['pca_hdpi']['avg_seconds']/by['pca_original']['avg_seconds']-1)*100
reduction=(1-by['pca_original']['avg_seconds']/by['neither']['avg_seconds'])*100
lines=['# SharedQ 五组消融：新增10个u（seed 20261025）','',
       '本轮五组共50次运行全部成功，独立重算验证500个数据集答案全部正确，无超时、无失败。结果与前两轮分开统计，所有组包括full均重新执行。','',
       '| 配置 | 成功u | 平均共享问题数 | 平均算法时间/秒 | 正确率 | 超时 | 失败 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for name,r in zip(NAMES,rows): lines.append(f"| {name} | 10/10 | {r['avg_questions']:.1f} | {r['avg_seconds']:.6f} | 100% | 0 | 0 |")
lines+=['','## 实验定义','',
        '同一批10个新增固定效用向量（Dirichlet(1,1,1,1)，NumPy default_rng seed20261025），已核验不与前两轮重复。每个u联合查询相同10个100k×4原始数据集，cor:anti:ind=3:3:4；每u独立进程硬限3600秒。BLAS/OMP/MKL各1线程，detect_workers=1，环境为服务器LQH。',
        '平均时间使用算法返回的elapsed，包含算法内部初始化和交互搜索，不包含独立数据文件加载与进程启动。问题数是跨10个数据集共享的问题数。每组正确率覆盖100个数据集答案。','',
        '- neither：PCA reduction与detect-and-divide均关闭；逐候选全扫描，禁用neighbor walk。',
        '- pca_original：仅PCA开启；使用与neither相同的逐候选全扫描分区。',
        '- pca_hdpi：仅PCA开启；HDPI式全候选分区初始化和逐答案增量裁剪。',
        '- detect_divide_only：仅detect-and-divide开启；关闭PCA后保留全量跨数据集评分。',
        '- full：PCA reduction与detect-and-divide均开启。','',
        '## 同批u的比较','',
        f'- 固定原始分区实现（pca_original vs neither）：PCA开启后平均问题数18.7→19.6，平均耗时1183.004→917.225秒，耗时减少{reduction:.2f}%。',
        f'- 固定PCA问题选择逻辑（pca_hdpi vs pca_original）：HDPI分区平均问题数19.6→20.0，平均耗时917.225→986.437秒，耗时增加{percent_hdpi:.2f}%。',
        '- 在detect-and-divide开启时（full vs detect_divide_only）：PCA开启后平均问题数22.2→23.1，耗时6.556→7.163秒。',
        '本轮PCA reduction没有减少平均问题数；HDPI分区也没有比原始全扫描更快。分区实现差异可能改变数值结果和后续问题序列，因此相同的PCA选择逻辑不保证逐u问题数相同。这些结论限于本轮10个u及该数据集组合。','',
        '## 核验与文件','',
        '核验了50个逐u JSON、500个答案、数据集SHA-256、10个效用向量排重，以及服务器汇总均值。verified_answers.csv中的score_gap由本地原始点集重新计算，未仅依赖服务器correct标记。',
        '- verified_summary.csv：独立重算的均值与样本标准差。',
        '- per_utility.csv：50次运行的问题数与算法时间。',
        '- paired_partition_comparison.csv：两种PCA分区及neither的逐u配对对比。',
        '- verified_answers.csv：500个答案的独立正确性核验。',
        '- ablation_comparison.png / .pdf：五组问题数与耗时对比图。',
        '- manifest.json / utilities.json / source_hashes.sha256 / runtime.json：参数、效用、源码哈希和环境记录。',
        '- server_run.log / 各组u*.json：原始实验记录。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'validation.json').write_text(json.dumps(dict(validated_runs=50,independently_verified_answers=len(answer_rows),
    dataset_hashes_verified=True,new_utilities_distinct_from_both_previous_batches=True,summary_recomputed=True),indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))

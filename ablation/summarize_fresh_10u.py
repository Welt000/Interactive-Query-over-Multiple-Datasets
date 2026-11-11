"""Validate and report the completed fresh ten-utility ablation."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ablation_sharedq_10u_20261024'
GROUPS = ['neither', 'pca_only', 'detect_divide_only', 'full']
LABELS = ['Neither', 'PCA only', 'Detect & Divide only', 'Full']
manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8-sig'))
utilities = np.asarray(manifest['utilities'])
assert utilities.shape == (10, 4) and np.all(utilities > 0)
assert np.allclose(utilities.sum(axis=1), 1) and len(np.unique(utilities, axis=0)) == 10
old = json.loads((ROOT / 'results' / 'ablation_sharedq' / 'manifest.json').read_text(encoding='utf-8-sig'))
assert all(not np.allclose(u, v, atol=1e-12, rtol=0) for u in utilities for v in old['utilities'])
rows, detail_rows = [], []
for group in GROUPS:
    records = [json.loads((OUT / group / f'u{i}.json').read_text()) for i in range(10)]
    for i, r in enumerate(records):
        assert r['utility_id'] == i and r['group'] == group and r['status'] == 'ok'
        assert r['reused'] is False and r['resolved'] and r['datasets'] == 10 and r['accuracy'] == 1
        assert len(r['details']) == 10 and all(d['correct'] and d['candidate_count'] == 1 for d in r['details'])
        detail_rows.append(dict(group=group, utility_id=i, questions=r['questions'], seconds=r['elapsed'], accuracy=r['accuracy']))
    row = dict(group=group, successful=10, avg_questions=mean(r['questions'] for r in records),
               avg_seconds=mean(r['elapsed'] for r in records), accuracy=1.0, timeouts=0, failures=0,
               sd_questions=stdev(r['questions'] for r in records), sd_seconds=stdev(r['elapsed'] for r in records))
    saved = json.loads((OUT / group / 'summary.json').read_text())
    assert np.isclose(row['avg_questions'], saved['avg_questions_successful'])
    assert np.isclose(row['avg_seconds'], saved['avg_seconds_successful'])
    rows.append(row)
for name, items in [('verified_summary.csv', rows), ('per_utility.csv', detail_rows)]:
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(items[0])); writer.writeheader(); writer.writerows(items)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout='constrained')
colors = ['#8a8f99', '#5e87c7', '#62a38b', '#b675b5']
for ax, field, title in [(axes[0], 'avg_questions', 'Mean shared questions'), (axes[1], 'avg_seconds', 'Mean algorithm time (seconds, log scale)')]:
    values = [r[field] for r in rows]
    bars = ax.bar(range(4), values, color=colors)
    ax.set_xticks(range(4), ['Neither', 'PCA\nonly', 'Detect & Divide\nonly', 'Full'])
    ax.set_title(title, fontsize=11); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    if field == 'avg_seconds': ax.set_yscale('log'); ax.set_ylim(1, 3500)
    else: ax.set_ylim(0, 26)
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}' if field == 'avg_seconds' else f'{val:.1f}', (bar.get_x()+bar.get_width()/2, val), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
fig.suptitle('SharedQ ablation: 10 fresh utilities, 10 datasets per utility\nAll groups: 10/10 successful, 100% accuracy, no timeouts or failures', fontsize=12)
fig.savefig(OUT / 'ablation_comparison.png', dpi=200)
fig.savefig(OUT / 'ablation_comparison.pdf')
plt.close(fig)

lines = ['# SharedQ 新增10个效用向量消融实验', '',
         '四组共40次运行全部成功；每次联合查询10个100k×4原始数据集（cor:anti:ind=3:3:4）。',
         '四组使用同一批新增效用向量，Dirichlet(1,1,1,1)，生成种子20261024；已核验不与上一轮10个u重复。每次硬限3600秒。full本轮重新执行，无历史结果复用。', '',
         '| 组别 | 成功u | 平均共享问题数 | 平均算法时间/秒 | 正确率 | 超时 | 失败 |',
         '|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['group']} | 10/10 | {r['avg_questions']:.1f} | {r['avg_seconds']:.6f} | 100% | 0 | 0 |")
lines += ['', '平均算法时间取每次返回的elapsed，包含该算法内部初始化与交互搜索，不包含独立数据文件加载及进程启动。问题数是跨10个数据集共享的提问数。正确率覆盖每组100个数据集答案。', '',
          '## 实现定义与比较限制', '',
          '- neither：PCA reduction与detect-and-divide均关闭，逐候选全扫描构造分区，禁用neighbor walk。',
          '- pca_only：仅PCA reduction开启，采用HDPI式全候选分区初始化和逐答案增量裁剪。',
          '- detect_divide_only：仅detect-and-divide开启，关闭PCA reduction后保留全量跨数据集评分。',
          '- full：两项均开启；使用detect-and-divide及邻居分区构造。', '',
          '该系列沿用已约定的各组实现，PCA-only与neither的分区构造方式也不同，因此两者时间差不能全部归因于PCA reduction本身。', '',
          '## 结果观察', '',
          f"- full相对neither，算法平均耗时缩短至原来的1/{rows[0]['avg_seconds']/rows[3]['avg_seconds']:.1f}；问题数由19.5变为20.0。",
          '- PCA-only问题数20.1，略高于neither的19.5；full的问题数20.0，低于detect-and-divide only的21.5。',
          '本轮结果描述这10个新效用向量，不代表普遍的性能保证。', '',
          '## 文件', '',
          '- per_utility.csv：40次运行的问题数、算法耗时和正确率。',
          '- verified_summary.csv：从全部逐u JSON独立重算的均值与标准差。',
          '- ablation_comparison.png / .pdf：问题数和耗时对比图。',
          '- manifest.json / utilities.json：数据集哈希、参数和效用向量。',
          '- server_run.log / 各组u*.json：原始运行记录。', '',
          '已核验40个逐u结果状态、各组100个答案正确标记、无结果复用、均值与服务器summary.json一致。无效接口试跑已隔离，未纳入本报告。']
(OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
audit = {'validated_runs': 40, 'validated_dataset_answers': 400, 'new_utilities_distinct_from_previous': True,
         'summary_recomputed': True, 'all_successful': True,
         'local_source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [ROOT/'SharedQ/algorithm.py', ROOT/'ablation/run_ablation_fresh.py']}}
(OUT/'validation.json').write_text(json.dumps(audit, indent=2),encoding='utf-8')
print(json.dumps(rows, indent=2))

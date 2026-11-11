"""Independent full-data audit, including completed but incorrect runs."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from baseline.experiment_utils import load_dataset
from cargurus_benchmark import OUT,METHODS,save,digest

def main():
    manifest=json.loads((OUT/'manifest.json').read_text());checks=[];runs=[];records=[]
    for d in (2,3,4):
        us=np.asarray(manifest['utilities'][str(d)])
        values={(m,u):json.loads((OUT/f'{d}d'/m/f'u{u:02d}.json').read_text()) for m in METHODS for u in range(10)}
        for idx,e in enumerate(manifest['dimensions'][str(d)]['datasets']):
            assert digest(Path(e['path']))==e['sha256'] and digest(Path(e['skyline_path']))==e['skyline_sha256']
            raw=load_dataset(Path(e['path']));front=load_dataset(Path(e['skyline_path']));truth=np.max(raw@us.T,axis=0)
            for m in METHODS:
                for u in range(10):
                    v=values[m,u]
                    if v['status'] not in ('ok','incorrect'):continue
                    assert np.array_equal(v['utility'],us[u])
                    detail=OUT/f'{d}d'/m/f'u{u:02d}_detail.csv'
                    rows=list(csv.DictReader(detail.open()));assert len(rows)==10
                    r=next(r for r in rows if int(r['dataset_id'])==idx)
                    used=raw if m=='SharedQ' else front;ri=int(r['result_index']);assert 0<=ri<len(used)
                    score=float(used[ri]@us[u]);gap=float(truth[u]-score);correct=abs(gap)<=1e-9
                    assert abs(score-float(r['score']))<=1e-12 and abs(truth[u]-float(r['truth_score']))<=1e-12
                    checks.append(dict(dimension=d,method=m,utility_id=u,dataset_id=idx,score=score,truth=float(truth[u]),gap=gap,correct=correct))
        for m in METHODS:
            vs=[values[m,u] for u in range(10)];completed=[v for v in vs if v['status'] in ('ok','incorrect')]
            relevant=[c for c in checks if c['dimension']==d and c['method']==m]
            for u in range(10):
                v=values[m,u];cs=[c for c in relevant if c['utility_id']==u]
                if cs:assert (v['status']=='ok')==all(c['correct'] for c in cs)
                r=dict(dimension=d,method=m,utility_id=u,status=v['status'],correct_answers=sum(c['correct'] for c in cs),answers_checked=len(cs))
                if 'summaries' in v:r.update(questions=v['summaries'][0]['avg_total_questions_per_utility'],elapsed=v['summaries'][0]['avg_total_elapsed_per_utility'])
                runs.append(r)
            prep=0. if m=='SharedQ' else sum(e['skyline_elapsed'] for e in manifest['dimensions'][str(d)]['datasets'])
            avg=lambda key:float(np.mean([v['summaries'][0][key] for v in completed])) if completed else None
            elapsed=avg('avg_total_elapsed_per_utility')
            records.append(dict(dimension=d,method=m,planned=10,completed=len(completed),correct_runs=sum(v['status']=='ok' for v in vs),
                incorrect_runs=sum(v['status']=='incorrect' for v in vs),timeouts=sum(v['status']=='timeout' for v in vs),execution_errors=sum(v['status']=='error' for v in vs),
                correct_answers=sum(c['correct'] for c in relevant),answers_checked=len(relevant),answer_accuracy=sum(c['correct'] for c in relevant)/len(relevant) if relevant else None,
                avg_questions_all_completed=avg('avg_total_questions_per_utility'),avg_elapsed_all_completed=elapsed,
                avg_search_elapsed_all_completed=elapsed-prep if elapsed is not None else None,skyline_elapsed=prep))
    for name,rows in (('comparison_all_runs.csv',records),('per_utility_all_runs.csv',runs),('verified_answers_all_runs.csv',checks)):
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    incorrect=[c for c in checks if not c['correct']]
    save(OUT/'correctness_audit.json',dict(planned_runs=150,completed_runs=sum(r['completed'] for r in records),
        correct_runs=sum(r['correct_runs'] for r in records),incorrect_runs=sum(r['incorrect_runs'] for r in records),
        answers_checked=len(checks),correct_answers=sum(c['correct'] for c in checks),incorrect_answers=incorrect,
        max_abs_score_gap=max(abs(c['gap']) for c in checks),all_data_hashes_match=True,
        aggregation='Means include every completed run including incorrect answers. Correctness columns must be read alongside performance.',
        original_summary='summary.csv contains correct-run-only means; comparison_all_runs.csv contains all-completed means.'))
    old=(OUT/'REPORT.md').read_text(encoding='utf-8')
    lines=['','## 全部完成运行的性能与正确率','',
        '下表包含所有执行完成的u，包括答案错误的运行；正确率单独列出。上表summary.csv均值仅覆盖正确的u。表中时间为联合查询10组的总时间，SUR含Skyline预处理。',
        '', '|维度|方法|完成/10|正确u/10|正确答案/100|平均问题数|总时间(s)|搜索时间(s)|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        fmt=lambda x:'—' if x is None else f'{x:.6f}'
        lines.append(f"|{r['dimension']}|{r['method']}|{r['completed']}/10|{r['correct_runs']}/10|{r['correct_answers']}/{r['answers_checked']}|{fmt(r['avg_questions_all_completed'])}|{fmt(r['avg_elapsed_all_completed'])}|{fmt(r['avg_search_elapsed_all_completed'])}|")
    lines+=['','错误答案（效用差阈值1e-9）：']
    for c in incorrect:lines.append(f"- {c['dimension']}D {c['method']}，u{c['utility_id']:02d}（从0编号），组{c['dataset_id']}，最优效用差 {c['gap']:.12g}。")
    if not incorrect:lines.append('- 无。')
    lines+=['','补充核验对所有执行完成的运行逐组重算真实最优值，包含错误运行；详见correctness_audit.json和verified_answers_all_runs.csv。原始实验代码和结果均未修改。']
    marker='\n## 全部完成运行的性能与正确率'
    (OUT/'REPORT.md').write_text(old.split(marker)[0]+'\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(records,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

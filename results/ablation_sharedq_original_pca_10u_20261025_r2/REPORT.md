# SharedQ 四组消融：仅PCA使用原始全扫描分区（r2）

四组40次运行全部成功，400个数据集答案经原始点集独立核验全部正确，无超时、无失败。所有结果均来自本轮重新运行，未复用历史结果。

| 配置 | 成功u | 平均共享问题数 | 平均算法时间/秒 | 正确率 | 超时 | 失败 |
|---|---:|---:|---:|---:|---:|---:|
| 两项关闭 | 10/10 | 20.4 | 1171.477214 | 100% | 0 | 0 |
| 仅PCA：原始全扫描 | 10/10 | 20.6 | 946.644186 | 100% | 0 | 0 |
| 仅detect-and-divide | 10/10 | 25.3 | 7.670965 | 100% | 0 | 0 |
| 两项开启（full） | 10/10 | 26.4 | 8.118288 | 100% | 0 | 0 |

## 实验定义

10个新固定u：Dirichlet(1,1,1,1)，NumPy default_rng seed2026102502；已与之前30个u排重。四组使用同批u，各u联合查询相同10个100k×4原始数据集，cor:anti:ind=3:3:4。每u单独进程硬限3600秒，LQH环境，BLAS/OMP/MKL各1线程，detect_workers=1。
时间取算法返回的elapsed，包含内部初始化与交互搜索，不包含独立数据文件加载及进程启动。问题数是跨10个数据集共享的提问数；每组正确率覆盖100个数据集答案。

- neither：PCA reduction及detect-and-divide均关闭；原始逐候选全扫描分区，禁用neighbor walk。
- pca_original：仅PCA reduction开启，分区与neither相同，使用原始逐候选全扫描，禁用neighbor walk。
- detect_divide_only：仅detect-and-divide开启；PCA关闭后保留全量跨数据集评分。
- full：PCA reduction及detect-and-divide均开启。

## 同批u的观察

- 固定原始全扫描分区后（pca_original vs neither），PCA开启使平均问题数20.4→20.6，算法耗时1171.477→946.644秒，耗时减少19.19%。
- DD开启时（full vs detect_divide_only），PCA开启使平均问题数25.3→26.4，算法耗时7.671→8.118秒。
本轮PCA reduction仍未减少平均问题数；其时间影响随是否开启DD而不同。以上仅描述本轮10个u及固定数据集组合。

## 核验与文件

已独立检查40个逐u记录及其均值、原始数据集SHA-256、与三批历史u排重。400个答案通过原始点集点积重新计算score_gap核验，并非仅采用服务器correct字段。仅汇总本轮四组。
- verified_summary.csv：均值及样本标准差。
- per_utility.csv：40次逐u问题数与算法时间。
- paired_pca_comparison.csv：相同分区下PCA开关的逐u配对结果。
- verified_answers.csv / validation.json：400个答案独立核验记录及审核摘要。
- ablation_comparison.png / .pdf：四组问题数、耗时对比图。
- manifest.json / utilities.json / source_hashes.sha256 / runtime.json：参数、效用和环境审计。
- completed.json / server_run.log / 各组u*.json：服务器原始记录。

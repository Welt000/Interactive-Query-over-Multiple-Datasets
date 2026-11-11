# SharedQ 五组消融：新增10个u（seed 20261025）

本轮五组共50次运行全部成功，独立重算验证500个数据集答案全部正确，无超时、无失败。结果与前两轮分开统计，所有组包括full均重新执行。

| 配置 | 成功u | 平均共享问题数 | 平均算法时间/秒 | 正确率 | 超时 | 失败 |
|---|---:|---:|---:|---:|---:|---:|
| 两项关闭 | 10/10 | 18.7 | 1183.003771 | 100% | 0 | 0 |
| 仅PCA：原始全扫描 | 10/10 | 19.6 | 917.225208 | 100% | 0 | 0 |
| 仅PCA：HDPI分区 | 10/10 | 20.0 | 986.437312 | 100% | 0 | 0 |
| 仅detect-and-divide | 10/10 | 22.2 | 6.556430 | 100% | 0 | 0 |
| 两项开启（full） | 10/10 | 23.1 | 7.163437 | 100% | 0 | 0 |

## 实验定义

同一批10个新增固定效用向量（Dirichlet(1,1,1,1)，NumPy default_rng seed20261025），已核验不与前两轮重复。每个u联合查询相同10个100k×4原始数据集，cor:anti:ind=3:3:4；每u独立进程硬限3600秒。BLAS/OMP/MKL各1线程，detect_workers=1，环境为服务器LQH。
平均时间使用算法返回的elapsed，包含算法内部初始化和交互搜索，不包含独立数据文件加载与进程启动。问题数是跨10个数据集共享的问题数。每组正确率覆盖100个数据集答案。

- neither：PCA reduction与detect-and-divide均关闭；逐候选全扫描，禁用neighbor walk。
- pca_original：仅PCA开启；使用与neither相同的逐候选全扫描分区。
- pca_hdpi：仅PCA开启；HDPI式全候选分区初始化和逐答案增量裁剪。
- detect_divide_only：仅detect-and-divide开启；关闭PCA后保留全量跨数据集评分。
- full：PCA reduction与detect-and-divide均开启。

## 同批u的比较

- 固定原始分区实现（pca_original vs neither）：PCA开启后平均问题数18.7→19.6，平均耗时1183.004→917.225秒，耗时减少22.47%。
- 固定PCA问题选择逻辑（pca_hdpi vs pca_original）：HDPI分区平均问题数19.6→20.0，平均耗时917.225→986.437秒，耗时增加7.55%。
- 在detect-and-divide开启时（full vs detect_divide_only）：PCA开启后平均问题数22.2→23.1，耗时6.556→7.163秒。
本轮PCA reduction没有减少平均问题数；HDPI分区也没有比原始全扫描更快。分区实现差异可能改变数值结果和后续问题序列，因此相同的PCA选择逻辑不保证逐u问题数相同。这些结论限于本轮10个u及该数据集组合。

## 核验与文件

核验了50个逐u JSON、500个答案、数据集SHA-256、10个效用向量排重，以及服务器汇总均值。verified_answers.csv中的score_gap由本地原始点集重新计算，未仅依赖服务器correct标记。
- verified_summary.csv：独立重算的均值与样本标准差。
- per_utility.csv：50次运行的问题数与算法时间。
- paired_partition_comparison.csv：两种PCA分区及neither的逐u配对对比。
- verified_answers.csv：500个答案的独立正确性核验。
- ablation_comparison.png / .pdf：五组问题数与耗时对比图。
- manifest.json / utilities.json / source_hashes.sha256 / runtime.json：参数、效用、源码哈希和环境记录。
- server_run.log / 各组u*.json：原始实验记录。

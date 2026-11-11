# SharedQ 新增10个效用向量消融实验

四组共40次运行全部成功；每次联合查询10个100k×4原始数据集（cor:anti:ind=3:3:4）。
四组使用同一批新增效用向量，Dirichlet(1,1,1,1)，生成种子20261024；已核验不与上一轮10个u重复。每次硬限3600秒。full本轮重新执行，无历史结果复用。

| 组别 | 成功u | 平均共享问题数 | 平均算法时间/秒 | 正确率 | 超时 | 失败 |
|---|---:|---:|---:|---:|---:|---:|
| neither | 10/10 | 19.5 | 1113.372104 | 100% | 0 | 0 |
| pca_only | 10/10 | 20.1 | 1008.385778 | 100% | 0 | 0 |
| detect_divide_only | 10/10 | 21.5 | 6.263931 | 100% | 0 | 0 |
| full | 10/10 | 20.0 | 5.840895 | 100% | 0 | 0 |

平均算法时间取每次返回的elapsed，包含该算法内部初始化与交互搜索，不包含独立数据文件加载及进程启动。问题数是跨10个数据集共享的提问数。正确率覆盖每组100个数据集答案。

## 实现定义与比较限制

- neither：PCA reduction与detect-and-divide均关闭，逐候选全扫描构造分区，禁用neighbor walk。
- pca_only：仅PCA reduction开启，采用HDPI式全候选分区初始化和逐答案增量裁剪。
- detect_divide_only：仅detect-and-divide开启，关闭PCA reduction后保留全量跨数据集评分。
- full：两项均开启；使用detect-and-divide及邻居分区构造。

该系列沿用已约定的各组实现，PCA-only与neither的分区构造方式也不同，因此两者时间差不能全部归因于PCA reduction本身。

## 结果观察

- full相对neither，算法平均耗时缩短至原来的1/190.6；问题数由19.5变为20.0。
- PCA-only问题数20.1，略高于neither的19.5；full的问题数20.0，低于detect-and-divide only的21.5。
本轮结果描述这10个新效用向量，不代表普遍的性能保证。

## 文件

- per_utility.csv：40次运行的问题数、算法耗时和正确率。
- verified_summary.csv：从全部逐u JSON独立重算的均值与标准差。
- ablation_comparison.png / .pdf：问题数和耗时对比图。
- manifest.json / utilities.json：数据集哈希、参数和效用向量。
- server_run.log / 各组u*.json：原始运行记录。

已核验40个逐u结果状态、各组100个答案正确标记、无结果复用、均值与服务器summary.json一致。无效接口试跑已隔离，未纳入本报告。

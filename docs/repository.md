# 仓库组织指南

本页说明一级目录的职责，并提供各目录的入口。

- [braincell/](../braincell/)：库实现与共置测试，包含实验模块。
- [examples/](../examples/)：使用示例、教学与任务工作流。
- [validation/](../validation/README.md)：数值精度与正确性验证。
- [benchmarks/](../benchmarks/README.md)：性能、资源占用与规模测试；完整实测总结在各实验的 `results/` 随 Git 维护，原始材料在忽略的 `artifacts/`，Design 在对应方案或架构文档中引用实测依据。
- [data/](../data/README.md)：共享输入数据与参考资料。
- [docs/](./)：使用说明、设计、贡献指南与历史记录。

一级目录职责变化时更新本页；内部结构、接口、运行命令和数据细节由所属目录的文档维护。

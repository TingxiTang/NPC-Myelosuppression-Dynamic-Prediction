# 急性骨髓抑制研究用 Web 原型

## 1. 用途与边界

本目录是旧 LightGBM Web 的独立升级副本。它将项目已冻结的三个 XGBoost endpoint 封装为 Streamlit 研究用技术原型，同时显示：

- Grade ≥3 贫血（Hb）锁定概率与阈值状态；
- Grade ≥3 血小板减少（PLT）锁定概率与阈值状态；
- Grade ≥3 白细胞/中性粒细胞减少（WBC/Neut）锁定概率与阈值状态；
- 锁定校准 logit 空间的 XGBoost native TreeSHAP 瀑布图与贡献概览；
- 中英文界面切换，以及与当前界面语言一致的无身份 PDF 风险报告；
- 参考原 Web 的中英文治疗方案多选，并用同一冻结模型对当前所选方案进行一次情景重算。

> 本原型仅用于研究和技术验证，未经前瞻性临床影响评估，不用于诊断、治疗选择、剂量调整或代替临床判断。SHAP 仅是预测贡献，不是治疗因果效应。

## 2. 冻结资产 lineage

- Independent bundle：`independent-bundle-84ad4e9926aa54644d429724`
- Selection lock SHA-256：`494baf126d05e13c0b22793e6755baadfc3e0181210562c3a44fd1acaaf48cad`
- 模型：XGBoost 3.3.0，Hb / PLT / WBC-Neut 各一套
- 输入：106 个 raw clinical features
- 编码后：253 个特征
- 校准：tune 阶段冻结 logistic recalibrator，不在 Web 中重拟合
- 分类规则：概率 `>=` 冻结阈值

| Endpoint | 冻结阈值 |
|---|---:|
| Hb | 0.03211432847337887 |
| PLT | 0.01259942442598835 |
| WBC/Neut | 0.03810385713817803 |

`artifact_contract.json` 只登记公开的冻结身份和哈希。模型文件由 `scripts/stage_artifacts.py` 从已验证的 independent bundle 机械复制到 `.artifacts/`；任一哈希不符即停止。

## 3. 隐私设计

- 仅接收单行、单治疗周期 CSV；
- 拒绝姓名、患者/subject ID、住院号、病案号、身份证、电话、地址、日期/时间和自由文本类字段；
- 拒绝非冻结特征列；
- 上传内容只在当前会话内存处理，程序不写入患者输入；
- PDF 在会话内存生成，仅包含三项风险与 SHAP 贡献，不包含姓名、ID、日期、治疗方案或完整输入行；
- 可部署包不包含 sentinel raw rows、independent rows、旧测试 CSV、旧 PDF 或任何 parquet/joblib。

## 4. 输入方式

推荐使用项目中已冻结特征构建流程生成的 106 列单行 CSV。界面也提供两个完全合成的技术测试样例，可直接运行或下载为 CSV 模板。

上传或选定样例后，界面会从冻结编码恢复当前治疗项目，并按所选语言提供多选框。中英文药名均直接来自同一份经哈希校验的历史药物字典，不改变 Drug_ID、Category_ID 或药物—类别关系。更改选项后，只重建以下“本次方案”字段：`drug_id`、`category_id`、`is_chemo`、`is_target`、`is_immuno`、`is_rt`；基线检验、分期、既往 nadir、累计暴露和周期状态保持不变。药物—类别关系来自原 Web 的映射表，并进一步限制为当前冻结 encoder 的词表；任一字典哈希、编码覆盖或类别关系异常都会停止预测。

界面每次只显示当前所选方案的一次预测，不自动保存基线方案、不并列比较两个方案，也不计算“降低了多少”。用户可以更改方案后重新计算，观察模型预测值是否变化。该结果只是模型情景敏感性，不能解释为换药获益、治疗效应、方案优劣或治疗推荐。

`drug_id` 和 `category_id` 在 CSV 中使用 JSON 数组，例如：

```text
"[2, 17]","[2, 5]"
```

缺少的冻结特征列会被显式补为缺失，再交给 frozen encoder/preprocessor。程序不会沿用旧 Web 的“缺失默认为 0”、新中位数填补或缩放失败后继续预测逻辑。

## 5. 本地运行

使用 Python 3.12，在项目根目录执行：

```bash
python3.12 web_app/scripts/stage_artifacts.py
python3.12 -m venv .venv
.venv/bin/python -m pip install -r web_app/requirements.txt
.venv/bin/python -m streamlit run web_app/app.py
```

默认使用 Streamlit 本地地址。应用启动时会先校验 selection lock、encoder、preprocessor、feature order、UBJ 模型、calibrator 和 threshold 哈希；任一不符会 fail closed。

## 6. 测试

使用已安装 XGBoost 3.3.0 和 pytest 的 Python：

```bash
PYTHONPATH=. python3 -m pytest web_app/tests -q
```

一致性测试包括：

1. 三 endpoint 的全部 16 行 sentinel；
2. 两个固定合成样例；
3. raw margin、raw probability、锁定概率、`>=` 分类结果；
4. encoded SHAP 回聚合到 106 raw features 后的逐特征 parity；
5. raw-margin 和 locked-logit SHAP 加和；
6. 身份/日期字段拒绝；
7. 治疗方案字典哈希、冻结词表覆盖、类别派生和本次治疗标志位。
8. Streamlit 启动导入直接指向实际子模块，且包初始化不提前加载 PDF 模块；
9. AppTest 覆盖中文默认界面、英文切换、治疗方案计算、英文 SHAP 标签和 PDF 下载对象。

当前 Web 测试共 27 项。界面提供中英文切换，并可下载与当前界面语言一致的无身份 PDF 报告。

预测 parity 超过 `1e-12` 或 SHAP 加和超过 `1e-4` 必须停止，不得用改模型、重校准或调阈值解决。

项目源码副本中，聚合且不含患者行的 Figure 5 技术一致性证据位于：

- `evidence/figure5_technical_consistency_2026-07-15.md`
- `evidence/figure5_technical_consistency_2026-07-15.json`

为避免生成循环校验和，`evidence/` 不复制进 tar.gz；部署包与证据作为两个独立交付物保存。

## 7. 构建可部署包

```bash
python3 web_app/scripts/stage_deploy.py
```

输出：

- `web_app/dist/acute_myelotoxicity_research_web/`
- `web_app/dist/acute_myelotoxicity_research_web.tar.gz`

部署目录包含 Streamlit 入口、最小 canonical runtime、已验证模型资产、`package_manifest.json` 和 `checksums.sha256`。不包含患者行、sentinel raw rows、secrets 或本机绝对路径。

## 8. 现有线上部署

- 连接仓库：`TingxiTang/NPC-Myelosuppression-Dynamic-Prediction`
- 分支/入口：`main` / `web_tool/app.py`
- 当前同步版本：以远端 `main` 的最新提交和项目部署审计登记为准
- 应用地址：`https://myelosuppression-pred.streamlit.app/`

截至 2026-07-15，代码已 fast-forward 推送，但应用地址仍要求 Streamlit 认证；可见性设置未改动，认证后的云端构建和功能验收尚未完成。不要把“仓库已同步”写成“新版已完成线上验收”。

## 9. 不允许的改动

本 Web 升级不得：

- 重训练、换模或修改 XGBoost 参数；
- 修改 106 raw / 253 encoded 特征定义与顺序；
- 重拟合 logistic recalibrator；
- 调整三个冻结阈值或改变 `>=` 运算符；
- 用 SHAP 宣称药物的保护/危害因果效应；
- 把方案情景重算解释为治疗效应、换药获益、方案优劣或自动推荐；
- 宣称已临床部署、已验证临床获益或可直接指导治疗。

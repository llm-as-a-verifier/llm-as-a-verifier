# mixed 组 PPT 两两对比选优复现报告（error_signals 单准则）

> 生成于 2026-09-24 · RUN_ID `ppt-dsh-r2`（pairwise）/ `solo-dsh-r2`（solo）·
> verifier `deepseek-v4-pro-0813`（thinking high，temperature=1.0，logprob
> 期望计分）· 数据与脚本见 `reproduce/`

## 1. 数据来源

- 基准：Terminal-Bench 4.0 dsh 迹（`raw_dsh_run_history`，agent = GLM-5.3
  coding plan），全量 60 任务 5 轮跑批
- 从中取 **mixed 组 23 个任务**：任务内五轮轨迹**既有 pass 也有 fail**
  （P1≥1 且 fail≥1），每任务 5 条 = **115 条轨迹**（keep 规则：每轮每任务
  保最优 P1>…>P5，并列取最新；全 ≥P3 时保留多条）
- 轨迹索引 `dsh-glm53-trajs.json`，本地缓存 `reproduce/trajs/<run_id>/`
  （5 run × 23 条，下载一次复用）

## 2. 轨迹本身的质量基线

| 口径 | 数值 |
|---|---|
| 轨迹平均通过率 | **63/115 = 54.8%**（即"随机选一条"的正确率） |
| 任务内 pass 分布 | 4条pass×7、3条×7、2条×5、1条×4 任务（每任务≥1 → oracle 选优上界 100%） |
| pass@1（按轮次单选） | r2=35%、full60=52%、r3/r5=61%、r4=65% |

## 3. 复现结果：PPT（pivot=2）两两对比选优

- 算法：每任务随机哈密顿环 5 对 + pivot(k=2) 轮 7 对，每对 4 次重复
  （偶数 rep 交换 A/B 槽位消偏），soft win `P(i>j)=sigmoid(R_i−R_j)`
  聚合，选 `argmax w/c`
- 准则：**仅 error_signals**（3 准则中唯一有判别信号者，见 §5）

| 选择策略 | 通过率 |
|---|---|
| 随机选一条 | 54.8%（≈12.6/23） |
| solo 单轨迹评分选最高 | 实测效果不佳（48%，劣于随机） |
| **PPT pairwise 选优** | **17/23 = 74%** |
| oracle（任务内必选 pass） | 100% |

**PPT 吃下 oracle headroom 的 ~42%**（+19pp over 随机），与论文
"对比框架能把 fail 轨迹打低"的机制一致；solo 绝对评分在此基准上分数与
正确性脱钩（详见 `solo_scores/results-dsh-r2-errsig.json`）。

## 4. 方法学要点（本轮踩坑与修正，影响数据可信度）

1. **端点仅流式返回 logprobs**：`api.modelverse.cn` 非流式响应无
   `choice.logprobs`，`_post_deepseek` 已切流式聚合
2. **字母位置解析错位（已修）**：字母实际为独立 token（`' X'`，标签闭合
   `'>'` 之后），旧版抓闭合位置的 top-20（噪声候选），分数与真实字母
   脱钩。现按实际字母 token 定位（兼容融合 `'>X'` 形态），并加
   `verify_letter_probs` 自校验（分布与文本字母脱钩即拒绝、不落缓存）
3. **原始返回落缓存**：调用缓存条目含 `raw` 尾窗（60 token 位置 ×
   top-20 全候选），解析口径变更可离线重放，无需重采
4. **读不到 logprobs 即报错**（严格口径，无文本回退）；失败不落缓存、
   重跑自动补
5. **上下文超限本地兜底**：端点输入上限 ~80 万 token（实测 2.35M chars
   通过、3.05M 报 400）。`MAX_PROMPT_CHARS=2.5M` 预检：超限调用直接按
   失败兜底、不发请求（biped 3 对、takens 4 对 + 1 条 solo 轨迹）
6. 限流/服务端错误（429/5xx）按 Retry-After 退避重试（≤5 次）

## 5. 准则有效性与失效模式

- **error_signals 是三准则中唯一有判别信号者**（探针任务调用级方向正确
  率 7/16 vs specification 2/16、output_match 3/16），本轮全量仅采该准则
- **失效模式 = 证据链缺失**：fail 轨迹在轨迹内无可见失败证据时
  verifier 无从判低。典型案例（轨迹内"自证成功"）：
  - mvcc `PJfXrbG`（11/15）：agent 自写回归测试全过，隐藏测试失败零痕迹
  - lake `fkGyfSS`（0/1）：自建 "verifier-style" 自测无报错跑完 ≠ 数值正确
- solo 口径下顽固高分 fail（mean>0.99）27 条——"终态无报错"与"任务正确"
  在此基准上系统性不等价

## 6. 产出文件

| 文件 | 内容 |
|---|---|
| `pair_scores/calls_cache.ppt-dsh-r2.json` | 879 条调用缓存（含 raw，logprobs 0 缺失） |
| `solo_scores/calls_cache.solo-dsh-r2.json` | 456 条调用缓存（含 raw） |
| `ppt_scores/results-dsh-r2-errsig.json` | PPT 选优汇总（17/23）+ 每任务明细 `ppt_scores/<task>.ppt-dsh-r2.json` |
| `solo_scores/results-dsh-r2-errsig.json` | solo 汇总（11/23）+ 每轨迹明细 |
| `ppt_scores/results-dsh-r1-errsig-letter.json` | r1 旧缓存按真实字母离散口径的重算（16/23，交叉验证） |

## 7. 结论

在 GLM-5.3 五轮跑批的 mixed 组 23 任务（115 条轨迹、平均通过率 54.8%）
上，以 error_signals 单准则、logprob 期望计分的 **PPT 两两对比选优达到
17/23 = 74%**，显著高于随机（54.8%）与 pass@1 各轮（35~65%），吃下
oracle headroom 的 ~42%；solo 单轨迹绝对评分实测效果不佳（48%，劣于
随机）。判别信号集中于 error_signals 准则；失效案例的共性是轨迹内
"证据链缺失"（agent 自测通过但隐藏测试失败）。

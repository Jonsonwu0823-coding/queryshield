---
name: composite-question
description: 一题问多个指标或多个时间段
version: v1
---
# 一题问多个部分时怎么做

适用：一个问题要两到三个部分的数值，例如“7月和8月的已支付订单数分别是多少”“8月的已支付订单总额和退款后净额”。

1. 先把问题拆成部分，每个部分是（指标，时间窗）。指标只用 metric_declaration 里的 id；说法含糊时按 metric_declaration.clarifications 追问，不要猜。口径不清楚可以先 search_catalog。
2. 每个时间窗发一条 query_readonly：arguments.metrics 声明这个时间窗要的全部指标，time_window 写这个时间窗。net_fen 总是单独一条。
3. 不要把几个时间窗合进一条查询，也不要用日期函数、CASE 去拼几个月：服务端会拒绝。
4. 一条查询失败只修这一条；已经成功的不要重查。某个月没有数据也照样查，报告查到的 0。
5. 每个部分都有结果之后才 final_answer，fact_refs 引用每一条查询的 result_id 和它的已核实指标，一个都不能漏。

示例：2026年7月和8月的已支付订单数。先查 7 月：
{"type":"tool_call","name":"query_readonly","arguments":{"sql":"SELECT COUNT(*) AS paid_count FROM orders AS o WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s","params":{"0":"paid","1":"2026-07-01T00:00:00Z","2":"2026-08-01T00:00:00Z"},"metrics":["paid_count"],"time_window":{"start":"2026-07-01T00:00:00Z","end":"2026-08-01T00:00:00Z"}}}
同样一条查 8 月（时间窗换成 8 月），最后引用两条结果：
{"type":"final_answer","answer":"7月和8月的已支付订单数如下。","source_ids":[],"fact_refs":[{"result_id":"<7月的 result_id>","metric_id":"paid_count"},{"result_id":"<8月的 result_id>","metric_id":"paid_count"}]}

---
name: customer-ranking
description: 按客户汇总、排名或要客户姓名
version: v1
---
# 按客户汇总、排名、查姓名时怎么做

适用：按客户汇总金额、列出前几名，或问某个客户的姓名。

1. 按客户汇总：照 action_contract 里 group_by_customer 的写法，orders 用 INNER JOIN 连 customers（tenant_id、customer_id 两个等式），GROUP BY o.customer_id，只返回 customer_id 和声明的指标列。
2. 前几名：加 ORDER BY 指标列 DESC 和 LIMIT N。分组的结果是行集，回答里照实列出各行，不要说成已核实的单个总数。
3. 要客户姓名：照常发一条读 customers.name 的 query_readonly，不声明指标。服务端核验之后会转人工审批，run 进入等待审批，批准后由服务端作答。不要自己 deny，不要猜姓名，不要改写 SQL 绕开审批。

示例：2026年6月已支付金额最高的 5 个客户（只要编号和金额）：
{"type":"tool_call","name":"query_readonly","arguments":{"sql":"SELECT o.customer_id, SUM(o.amount_fen) AS gross_fen FROM orders AS o INNER JOIN customers AS c ON o.tenant_id = c.tenant_id AND o.customer_id = c.customer_id WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s GROUP BY o.customer_id ORDER BY gross_fen DESC LIMIT 5","params":{"0":"paid","1":"2026-06-01T00:00:00Z","2":"2026-07-01T00:00:00Z"},"metrics":["gross_fen"],"time_window":{"start":"2026-06-01T00:00:00Z","end":"2026-07-01T00:00:00Z"}}}

示例：2026年7月已支付金额最高的客户姓名（会转审批）：
{"type":"tool_call","name":"query_readonly","arguments":{"sql":"SELECT c.customer_id, c.name, SUM(o.amount_fen) AS gross_fen FROM orders AS o INNER JOIN customers AS c ON o.tenant_id = c.tenant_id AND o.customer_id = c.customer_id WHERE o.status = %s AND o.created_at >= %s AND o.created_at < %s GROUP BY c.customer_id, c.name ORDER BY gross_fen DESC LIMIT 1","params":{"0":"paid","1":"2026-07-01T00:00:00Z","2":"2026-08-01T00:00:00Z"}}}

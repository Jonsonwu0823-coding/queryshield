# commerce-v1 Fixture

- fixture_version: commerce-v1
- currency: CNY
- amount_unit: RMB fen
- conversion: 100 fen = 1 yuan

## Time Window

- timezone: UTC
- window: [2026-09-01T00:00:00Z, 2026-10-01T00:00:00Z)

## Metric Definitions

- paid_count: 只统计 status = paid 的订单数量
- gross_fen: 只统计 paid 订单的金额总和
- refund_fen: 只统计查询窗口内、同一租户、且属于查询窗口内 `status = paid` 订单的退款金额总和
- net_fen: gross_fen - refund_fen
- cancelled 订单保存在数据库中，但不计入 paid_count 和 gross_fen
- 退款通过 `(tenant_id, order_id)` 归属订单；取消订单和窗口外订单的退款不计入 `refund_fen`
- 同一订单可以有多笔退款，但 `gross_fen` 只从 paid 订单集合计算一次
- 金额使用整数分保存，展示时转换为元，禁止使用浮点数保存金额

## Expected Results

| tenant | paid_count | gross_fen | refund_fen | net_fen | net_display |
|---|---:|---:|---:|---:|---:|
| A | 2 | 15000 | 3000 | 12000 | 120.00 元 |
| B | 1 | 990000 | 10000 | 980000 | 9800.00 元 |

## Hand Calculations

### Tenant A

- paid_count = 2
- gross_fen = 10000 + 5000 = 15000 fen = 150.00 yuan
- refund_fen = 2000 + 1000 = 3000 fen
- net_fen = 15000 - 3000 = 12000 fen = 120.00 yuan

### Tenant B

- paid_count = 1
- gross_fen = 990000 fen = 9900.00 yuan
- refund_fen = 10000 fen
- net_fen = 990000 - 10000 = 980000 fen = 9800.00 yuan

## Fixed Questions

1. 2026年9月已支付订单数
2. 2026年9月已支付订单总额
3. 2026年9月退款后净额

身份范围由服务端 token 映射决定，客户端不能通过 body 中的 tenant_id 改变查询租户。

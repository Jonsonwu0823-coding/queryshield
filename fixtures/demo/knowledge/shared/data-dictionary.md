# 数据字典：客户表 customers
customers 保存客户：tenant_id 是租户，customer_id 是租户内的客户编号，name 是客户姓名。客户姓名属于敏感字段，查询姓名需要审批。同一个 customer_id 可以在不同租户里重复，读取时必须带上租户归属。

# 数据字典：订单表 orders
orders 保存订单：tenant_id、order_id 共同标识一张订单，customer_id 指向客户，status 只有 paid 和 cancelled 两种，amount_fen 是订单金额，单位是分，created_at 是 UTC 下单时间。

# 数据字典：退款表 refunds
refunds 保存退款：tenant_id、refund_id 共同标识一张退款单，order_id 指向同一租户的订单，amount_fen 是退款金额，单位是分，created_at 是 UTC 退款时间。退款可以晚于订单所在的月份，同一订单可以有多笔退款。

证据和事实里出现的 commerce-v1，指的是表结构和口径，不是数据行；演示用的数据是另一份数据，表结构与它相同。

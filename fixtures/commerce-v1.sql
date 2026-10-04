INSERT INTO customers (tenant_id, customer_id, name) 
VALUES
    ('A', 'c1', '甲'),
    ('A', 'c2', '乙'),
    ('B', 'c1', '丙')
ON CONFLICT (tenant_id, customer_id) DO NOTHING;

INSERT INTO orders (
    tenant_id,
    order_id,
    customer_id,
    status,
    amount_fen,
    created_at
)
VALUES
    ('A', 'o1', 'c1', 'paid', 10000, '2026-09-05T10:00:00Z'),
    ('A', 'o2', 'c2', 'paid', 5000, '2026-09-06T10:00:00Z'),
    ('A', 'o3', 'c1', 'cancelled', 9000, '2026-09-07T10:00:00Z'),
    ('B', 'o1', 'c1', 'paid', 990000, '2026-09-08T10:00:00Z')

ON CONFLICT (tenant_id, order_id) DO NOTHING;


INSERT INTO refunds (
    tenant_id,
    refund_id,
    order_id,
    amount_fen,
    created_at
)
VALUES
    ('A', 'r1', 'o1', 2000, '2026-09-09T10:00:00Z'),
    ('A', 'r2', 'o1', 1000, '2026-09-10T10:00:00Z'),
    ('B', 'r1', 'o1', 10000, '2026-09-10T11:00:00Z')
ON CONFLICT (tenant_id, refund_id) DO NOTHING;
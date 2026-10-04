COMMERCE_SUMMARY_SQL = """
WITH paid_orders AS (
    SELECT
        tenant_id,
        order_id,
        amount_fen
    FROM orders
    WHERE tenant_id = %s
      AND status = 'paid'
      AND created_at >= %s
      AND created_at < %s
),
paid_summary AS (
    SELECT
        COUNT(*) AS paid_count,
        COALESCE(SUM(amount_fen), 0) AS gross_fen
    FROM paid_orders
),
refund_totals AS (
    SELECT
        COALESCE(SUM(r.amount_fen), 0) AS refund_fen
    FROM refunds AS r
    INNER JOIN paid_orders AS p
        ON p.tenant_id = r.tenant_id
       AND p.order_id = r.order_id
    WHERE r.tenant_id = %s
      AND r.created_at >= %s
      AND r.created_at < %s
)
SELECT
    paid_count,
    gross_fen,
    refund_fen,
    gross_fen - refund_fen AS net_fen
FROM paid_summary
CROSS JOIN refund_totals
"""

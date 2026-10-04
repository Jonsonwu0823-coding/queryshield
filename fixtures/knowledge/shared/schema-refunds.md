# Refunds table

`refunds` contains `tenant_id`, `refund_id`, `order_id`, `amount_fen`, and `created_at`. Refund rows are joined through the order identifier only after tenant and time-window checks.

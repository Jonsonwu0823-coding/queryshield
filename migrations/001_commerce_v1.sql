CREATE TABLE IF NOT EXISTS customers (
    tenant_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    PRIMARY KEY (tenant_id, customer_id)
);

CREATE TABLE IF NOT EXISTS orders (
    tenant_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('paid', 'cancelled' )),
    amount_fen BIGINT NOT NULL CHECK (amount_fen >= 0),
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, order_id),
    FOREIGN KEY (tenant_id, customer_id) REFERENCES customers (tenant_id, customer_id)
);

CREATE TABLE IF NOT EXISTS refunds (
    tenant_id TEXT NOT NULL,
    refund_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    amount_fen BIGINT NOT NULL CHECK (amount_fen >= 0),
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, refund_id),
    FOREIGN KEY (tenant_id, order_id) REFERENCES  orders (tenant_id, order_id)
);

-- W04: database-enforced tenant isolation for the read-only application role.
-- The application role is intentionally not created here: local bootstrap
-- owns role creation and can be rerun against an existing test database.

ALTER TABLE customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE customers FORCE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders FORCE ROW LEVEL SECURITY;
ALTER TABLE refunds ENABLE ROW LEVEL SECURITY;
ALTER TABLE refunds FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS customers_tenant_isolation ON customers;
CREATE POLICY customers_tenant_isolation ON customers
    FOR SELECT
    USING (tenant_id = current_setting('queryshield.tenant_id', true));

DROP POLICY IF EXISTS orders_tenant_isolation ON orders;
CREATE POLICY orders_tenant_isolation ON orders
    FOR SELECT
    USING (tenant_id = current_setting('queryshield.tenant_id', true));

DROP POLICY IF EXISTS refunds_tenant_isolation ON refunds;
CREATE POLICY refunds_tenant_isolation ON refunds
    FOR SELECT
    USING (tenant_id = current_setting('queryshield.tenant_id', true));

COMMENT ON POLICY customers_tenant_isolation ON customers IS
    'W04 server-bound tenant context; guarded SQL also injects tenant predicates';
COMMENT ON POLICY orders_tenant_isolation ON orders IS
    'W04 server-bound tenant context; guarded SQL also injects tenant predicates';
COMMENT ON POLICY refunds_tenant_isolation ON refunds IS
    'W04 server-bound tenant context; guarded SQL also injects tenant predicates';

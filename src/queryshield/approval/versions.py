"""The policy and catalog versions that runs, approvals and Fake results are bound to."""

from queryshield.catalog import DEFAULT_CATALOG_VERSION
from queryshield.policy.sql import SQL_POLICY_VERSION

BOUND_POLICY_VERSION = SQL_POLICY_VERSION
BOUND_CATALOG_VERSION = DEFAULT_CATALOG_VERSION

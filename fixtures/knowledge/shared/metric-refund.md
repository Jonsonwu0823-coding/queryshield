# Refund amount

`refund_fen` is the sum of refunds belonging to paid orders in the same tenant and UTC window. The query first establishes same-window paid-order membership, then aggregates refund amounts.

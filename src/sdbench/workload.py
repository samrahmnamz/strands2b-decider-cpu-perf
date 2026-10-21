"""The prompt-routing workload: one choice question over six destination teams,
with incoming-request prompts graduated by complexity (tiny -> xlarge).

This models a decider sitting on the hot path in front of an agent: a request
arrives and the model picks which team/tool/model should handle it. Latency scales
with the state (request) length, so we test a spread of sizes.
"""
from __future__ import annotations


ROUTE_INSTRUCTIONS = "Route this incoming customer request to the single most appropriate team."

ROUTE_CRITERIA: dict[str, str] = {
    "billing": "payments, invoices, payouts, refunds, charge disputes, subscriptions",
    "technical_support": "bugs, errors, outages, API failures, integration help, latency",
    "sales": "pricing, plans, quotes, upgrades, new purchases, demos, contracts",
    "account_security": "login problems, password resets, suspicious activity, 2FA, account lockout",
    "data_analytics": "reports, dashboards, metrics, SQL queries, data exports, BI",
    "general_inquiry": "anything that does not clearly fit the other destinations",
}

_FILLER = (
    " For additional background, our organization has been a customer since 2019 and "
    "operates across multiple regions with several business units, each maintaining its "
    "own set of integrations, service accounts, and reporting pipelines that occasionally "
    "interact in ways that make it hard to isolate where a given problem originates."
)

PROMPTS: dict[str, str] = {
    "tiny": "Payouts failing for 3 days!",
    "small": (
        "Hi, our scheduled payouts have been failing since Monday and the dashboard shows "
        "a 402 error on every attempt. Can someone look into this urgently?"
    ),
    "medium": (
        "We've had three days of failed payouts. Each night the batch job kicks off, runs "
        "for about twenty minutes, then every transfer bounces back with 'insufficient "
        "funds on source' even though the source account clearly has the balance. Our "
        "finance team is escalating because vendors haven't been paid. We also noticed the "
        "webhook that confirms settlement stopped firing around the same time, so we're not "
        "sure if this is a billing configuration problem or something in the API."
    ),
    "large": (
        "Opening this on behalf of the payments operations team. Since the start of the "
        "week, our automated payout runs have been failing consistently. Below is the "
        "forwarded thread from our internal channel.\n\n"
        "[Mon] Noticed the 9pm payout batch failed. Retried manually, same result.\n"
        "[Tue] Escalated to on-call. On-call saw 402 responses from the payouts endpoint "
        "and a spike in 5xx on the settlement webhook receiver.\n"
        "[Wed] Finance flagged that four vendor invoices are now overdue. Leadership wants "
        "a root-cause and an ETA.\n\n"
        "We need to understand whether this is a billing/account funding issue, a change on "
        "your side to the payouts API, or a problem with our webhook endpoint." + _FILLER
    ),
    "xlarge": (
        "ESCALATION SUMMARY -- PAYOUT FAILURES\n\n"
        "This is a consolidated report covering a multi-day incident affecting our payout "
        "pipeline, intended to give whoever picks this up the full context in one place.\n\n"
        "Background:\n" + _FILLER + _FILLER + "\n\n"
        "Timeline:\n"
        "Over the past several days we have observed a repeating pattern in which the "
        "nightly payout batch initializes normally, enumerates the pending transfers, and "
        "then fails partway through with a mixture of 402 responses on the payouts endpoint "
        "and intermittent 5xx errors on the settlement confirmation webhook. Manual retries "
        "during business hours reproduce the same failures. " + _FILLER + "\n\n"
        "Impact:\n"
        "Several vendor invoices are now overdue, finance has escalated to leadership, and "
        "we are accruing late fees. " + _FILLER + "\n\n"
        "Open questions:\n"
        "Is this a funding/billing configuration issue on our account, a change to the "
        "payouts API behaviour, a credential/permission problem on our service accounts, or "
        "a fault in our own webhook receiver? " + _FILLER + _FILLER
    ),
}

TIERS = ["tiny", "small", "medium", "large", "xlarge"]

# A realistic request mix for throughput tests (weight toward short tickets).
TRAFFIC_MIX = {
    "default": ["tiny"] * 3 + ["small"] * 3 + ["medium"] * 2 + ["large"] * 1 + ["xlarge"] * 1,
    "short": ["tiny"] * 4 + ["small"] * 4 + ["medium"] * 2,
}


def route_question():
    """Build the routing ChoiceQuestion. Imported lazily so the module is cheap."""
    from strands_decider.schema import ChoiceQuestion

    return {"route": ChoiceQuestion(instructions=ROUTE_INSTRUCTIONS, criteria=dict(ROUTE_CRITERIA))}


def state_for_token_target(engine, question, target_tokens: int, *, max_pad: int = 40) -> tuple[str, int]:
    """Grow a routing state until the FULL decision (state + rubric + options) hits
    ~target_tokens, as reported by `engine.ask(...).usage.input_tokens`.

    Use this to reproduce the published-benchmark comparison point: JevBench decisions
    average ~950 input tokens, so `state_for_token_target(engine, q, 950)` builds a
    like-for-like decision. Returns (state_text, measured_full_input_tokens).
    """
    base = ("ESCALATION: payout pipeline failing for days; 402s on the payouts endpoint "
            "and intermittent 5xx on the settlement webhook. ")
    state = base
    for _ in range(max_pad):
        tok = engine.ask(state, question).usage.input_tokens
        if tok >= target_tokens:
            return state, tok
        state += " " + _FILLER.strip()
    return state, engine.ask(state, question).usage.input_tokens

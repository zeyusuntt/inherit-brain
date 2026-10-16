"""Synthetic bug-channel records and a scripted fake chat for the Task 4b tests. Not HERB data."""

from __future__ import annotations

import json
from types import SimpleNamespace

from inherit.data.records import ProductRecords, PullRequest, SlackMessage, parse_mentions

PRODUCT = "Mini"
IAN, EMMA, BOB = "eid_ian00001", "eid_emma0002", "eid_bob00003"
BOT = "slack_admin_bot"
CH = "ch-mini-1"
CH_NAME = "bug-ianmiller-Mini"

EMPLOYEES = {
    IAN: {"employee_id": IAN, "name": "Ian Miller", "role": "Engineering Manager"},
    EMMA: {"employee_id": EMMA, "name": "Emma Martinez", "role": "Software Engineer"},
    BOB: {"employee_id": BOB, "name": "Bob Jones", "role": "Software Engineer"},
}
CUSTOMERS = {
    "CUST-0047": {"id": "CUST-0047", "name": "Tom Hill", "role": "CTO", "company": "TechCorp"},
    "CUST-0108": {"id": "CUST-0108", "name": "Sara Lee", "role": "PM", "company": "TechCorp"},
    "CUST-0064": {"id": "CUST-0064", "name": "Ann Roe", "role": "CIO", "company": "BlueWave"},
}

B_INTRO = "Hi team, several customers are reporting issues with Mini that we need to address. Let's go through them one by one."
B_TITLE1 = ("First up, we have a bug titled 'Duplicate Log Entries for Manual Actions'. This is affecting our Database "
            f"Management & Logging System. This is impacting customer CUST-0047. @{EMMA}, this one's for you!")
B_ACK1 = "Got it! I'll start looking into the duplicate log entries issue right away."
B_TITLE2 = ("Next, we have 'Inefficient Database Query Optimization'. This is causing slow response times in Backend "
            "Processing & Task Queue Management. Customer CUST-0108 is affected. Can you take this one?")
B_ACK2 = "Sure thing! I'll dive into the database queries."
B_TITLE3 = "Lastly, there's a memory leak issue in the suggestion rendering. It hits customer CUST-0064."
C_INTRO = "Hi team, I've found some interesting open-source bugs that might help us."
C_EXT = ("First, a bug in PostgreSQL titled 'Duplicate Log Entries for Transaction Rollbacks'. This is similar to our "
         "'Duplicate Log Entries for Manual Actions' issue. https://github.com/postgres/postgres/pull/3")
D_RES = ("Hi team, the bug titled 'Inefficient Database Query Optimization' has been resolved. You can check the "
         "details in the approved PR here: https://github.com/salesforce/Mini/pull/31.")
E_RES = "Hi team, the bug about duplicate manual action log rows has been resolved."
E_LINK = "You can check out the details in the PR here: https://github.com/salesforce/Mini/pull/32"
F_OPEN = "Hi team, I looked into the reported issue about the 'Memory Leak in Suggestion Rendering'."
F_CUE = "After reviewing the behaviour, it turns out this is not a bug that requires fixing."
G_REOPEN = "Hi team, the customer has reopened the bug titled 'Duplicate Log Entries for Manual Actions'."
G_TAKE = "I'll take it on again and start with the retry logic."

# (id, author, ts, text)
SLACK = [
    ("20260110-0-a0001", BOT, "2026-01-10T08:00:00", f"@{IAN} created this channel on 2026-01-10 08:00:00. "
     f"This is the very beginning of the {CH_NAME} channel."),
    ("20260111-0-b0001", IAN, "2026-01-11T09:00:00", B_INTRO),
    ("20260111-1-b0002", IAN, "2026-01-11T09:02:00", B_TITLE1),
    ("20260111-2-b0003", EMMA, "2026-01-11T09:05:00", B_ACK1),
    ("20260111-3-b0004", IAN, "2026-01-11T09:10:00", B_TITLE2),
    ("20260111-4-b0005", BOB, "2026-01-11T09:12:00", B_ACK2),
    ("20260111-5-b0006", IAN, "2026-01-11T09:20:00", B_TITLE3),
    ("20260113-0-c0001", IAN, "2026-01-13T10:00:00", C_INTRO),
    ("20260113-1-c0002", IAN, "2026-01-13T10:03:00", C_EXT),
    ("20260115-0-d0001", BOB, "2026-01-15T11:00:00", D_RES),
    ("20260115-1-d0002", EMMA, "2026-01-15T11:02:00", "Great work, Bob!"),
    ("20260117-0-e0001", EMMA, "2026-01-17T12:00:00", E_RES),
    ("20260117-1-e0002", EMMA, "2026-01-17T12:03:00", E_LINK),
    ("20260120-0-f0001", EMMA, "2026-01-20T13:00:00", F_OPEN),
    ("20260120-1-f0002", EMMA, "2026-01-20T13:02:00", F_CUE),
    ("20260125-0-g0001", IAN, "2026-01-25T14:00:00", G_REOPEN),
    ("20260125-1-g0002", EMMA, "2026-01-25T14:03:00", G_TAKE),
]


def msg(mid, author, ts, text, channel=CH, name=CH_NAME) -> SlackMessage:
    return SlackMessage(id=mid, channel_id=channel, channel_name=name, user_id=author, ts=ts, text=text,
                        mentions=parse_mentions(text), is_bot=author == BOT, in_directory=author in EMPLOYEES,
                        bug_channel_owner="ianmiller")


def pr(number: int, author: str = BOB, link: str | None = None) -> PullRequest:
    link = link or f"https://github.com/salesforce/Mini/pull/{number}"
    return PullRequest(id=f"github_com_salesforce_Mini_pull_{number}", link=link, number=str(number),
                       title=f"Fix {number}", summary="s", author=author, in_directory=True, merged=True,
                       mergeable=None, merged_raw="True", mergeable_raw="", state="closed",
                       created_at="2026-01-01T00:00:00", reviews=())


def records(slack=SLACK, prs=(31, 32)) -> ProductRecords:
    return ProductRecords(
        product=PRODUCT, messages=[msg(*r) for r in slack], prs=[pr(n) for n in prs],
        employees=dict(EMPLOYEES), customers=dict(CUSTOMERS))


# ---------------------------------------------------------------- scripted fake chat

def cite(rid: str, quote: str) -> dict:
    return {"record_id": rid, "quote": quote}


LEAK_BUG = {"local_id": "x1", "title": "Memory Leak in Suggestion Rendering", "aliases": [],
            "component": "User Interface & Frontend Performance", "impact_text": "memory leak issue",
            "customer_ids": ["CUST-0064"], "message_id": "20260111-5-b0006",
            "citations": [cite("20260111-5-b0006", "memory leak issue in the suggestion rendering")]}
REASSIGN_EVENT = {"type": "ASSIGNED", "bug_ref": "m1", "bug_mention": None, "actor_message_id": "20260125-1-g0002",
                  "assignee_eids": [EMMA], "fix_pr_links": [],
                  "citations": [cite("20260125-1-g0002", "I'll take it on again")]}


def session_json(bugs=(), events=(), external=()) -> str:
    return json.dumps({"bugs": list(bugs), "events": list(events), "external": list(external)})


def link_json(*pairs) -> str:
    return json.dumps({"links": [{"item": i, "bug": b} for i, b in pairs]})


class ScriptedChat:
    """Stands in for inherit.llm.chat. `script` maps a marker substring of the user message to the response text (or a
    list of texts consumed in order). A response is a string; a (text, completion_tokens) tuple sets the usage."""

    def __init__(self, script: dict | None = None, link=None):
        self.script = script or {}
        self.link = link
        self.calls: list[dict] = []

    def __call__(self, messages, *, arm, max_tokens=800, response_format=None, **kw):
        user = messages[-1]["content"]
        self.calls.append({"messages": messages, "arm": arm, "max_tokens": max_tokens,
                           "response_format": response_format, "user": user})
        text, toks = None, 50
        is_link = "LINK_TASK" in messages[0]["content"]
        if is_link:
            text = self.link if isinstance(self.link, str) else (self.link(user) if self.link else link_json())
        else:
            for marker, resp in self.script.items():
                if marker in user:
                    if isinstance(resp, list):
                        resp = resp.pop(0) if len(resp) > 1 else resp[0]
                    text = resp
                    break
            if text is None:
                text = session_json()
        if isinstance(text, tuple):
            text, toks = text
        return SimpleNamespace(message=SimpleNamespace(content=text), prompt_tokens=100, completion_tokens=toks,
                               cost_usd=0.0001, latency_s=0.0)

    @property
    def session_calls(self):
        return [c for c in self.calls if "LINK_TASK" not in c["messages"][0]["content"]]

    @property
    def link_calls(self):
        return [c for c in self.calls if "LINK_TASK" in c["messages"][0]["content"]]


def standard_chat() -> ScriptedChat:
    """Answers the unquoted intake (b6), the reopen follow-up (g2) and the one paraphrased resolution (e1 is I1)."""
    return ScriptedChat(
        script={"20260111-5-b0006": session_json(bugs=[LEAK_BUG]),
                "20260125-1-g0002": session_json(events=[REASSIGN_EVENT])},
        link=lambda user: link_json(("I1", "B1")))

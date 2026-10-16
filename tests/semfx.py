"""Synthetic records and a scripted fake chat for the Task 4c (semantic layer) tests. Not HERB data."""

from __future__ import annotations

import json
from types import SimpleNamespace

from inherit.data.records import (Document, Meeting, MeetingChat, ProductRecords, PullRequest, Review, Segment,
                                  SlackMessage, UrlRecord, parse_mentions)

PRODUCT = "Mini"
ANN, BEN, CY, DEE, EVE, SAM1, SAM2 = ("eid_aa000001", "eid_bb000002", "eid_cc000003", "eid_dd000004", "eid_ee000005",
                                      "eid_ff000006", "eid_ff000007")
BOT = "slack_admin_bot"

EMPLOYEES = {
    ANN: {"employee_id": ANN, "name": "Ann Lee", "role": "Product Manager"},
    BEN: {"employee_id": BEN, "name": "Ben Ray", "role": "Engineering Lead"},
    CY: {"employee_id": CY, "name": "Cy Poe", "role": "QA Specialist"},
    DEE: {"employee_id": DEE, "name": "Dee Fox", "role": "UX Researcher"},
    EVE: {"employee_id": EVE, "name": "Eve Kim", "role": "Software Engineer"},
    SAM1: {"employee_id": SAM1, "name": "Sam Roe", "role": "Software Engineer"},
    SAM2: {"employee_id": SAM2, "name": "Sam Roe", "role": "Marketing Manager"},
}
CUSTOMERS = {
    "CUST-0001": {"id": "CUST-0001", "name": "Tom Hill", "role": "CTO", "company": "TechCorp"},
    "CUST-0002": {"id": "CUST-0002", "name": "Sara Lee", "role": "PM", "company": "BlueWave"},
}

DOC_LINK = "https://sf-internal.slack.com/archives/docs/mini_market_research_report"
FINAL_LINK = "https://sf-internal.slack.com/archives/docs/final_mini_market_research_report"
FEEDBACK = "- Add industry specifics to the Executive Summary.\n- Include regulatory changes in Market Trends.\n"

# ---------------------------------------------------------------- meetings
M1 = "mini_dev_1"
M1_SEGS = [
    (4, "Ann Lee", "Today we plan real-time alerts and predictive analytics. Ben, can you lead real-time alerts?"),
    (5, "Ben Ray", "Yes, I'll lead real-time alerts. We should use WebSockets for pushing alerts."),
    (6, "Cy Poe", "I can help with testing the alerts."),
    (7, "Sam Roe", "I'll handle predictive analytics. The PR is titled 'Integrate LSTM Forecasting'."),
    (8, "Ann Lee", "Great. The 'Add Metrics Export' PR is complete and merged."),
    (9, "Zed Nobody", "I will do some documentation for the alerts."),
]
M2 = "mini_planning_1"
M2_SEGS = [
    (4, "Ann Lee", "Let's review the Market Research Report together."),
    (5, "Dee Fox", "In the Competitive Analysis, could we add a comparison table?"),
    (6, "Ben Ray", "I think the Executive Summary needs more detail on the algorithms."),
]
M2_CHAT = f"2026-03-02T10:30:00\nAnn Lee: {FINAL_LINK}"


def meeting(mid, date, doc_type, participants, segs) -> Meeting:
    return Meeting(mid, date, doc_type, tuple(participants), (),
                   tuple(Segment(f"{mid}#{n}", mid, n, sp, tx) for n, sp, tx in segs))


# ---------------------------------------------------------------- slack
PLAN, DEV, BUGCH = ("ch-plan", "planning-Mini"), ("ch-dev", "develop-ben-Mini"), ("ch-bug", "bug-ben-Mini")

R_SHARE = f"Hi team, I've shared the <{DOC_LINK}|Market Research Report> for Mini. Please give feedback."
R_BEN = "Thanks! In the Product Description section, please elaborate on the advanced algorithms."
R_DEE = "I agree. In the Market Trends section we should mention regulatory changes."
R_ANN = "Thanks for the feedback, everyone!"
C_RIVAL = ("Hi team, I was reading about RivalCo, a competitor product. It offers live dashboards and has limited "
           "export options. Thought it was worth discussing.")
C_DEMO = "@here see how RivalCo works: https://www.rivalco.com/demo"
Q_QA = (f"Hi team, QA found issues in the last release of Mini. The QA specialist, {'eid_cc000003'}, reported that "
        "exports crash on large files.")
Q_CUST = "Customer CUST-0001 mentioned that the limited CRM integration is a problem."
Q_STRENGTH = "And CUST-0002 highlighted that the lead scoring is helping them prioritize leads."
Q_TEAM = "The previous team might not have standardized integration protocols."
CUR_ISSUE = "Customer CUST-0001 mentioned that dashboards are slow."
D_PR3 = "Hi team, PR#3 is approved but not merged yet. The product direction has changed. Should we merge it?"
D_PR3_HOLD = "Agreed. Please do not merge PR#3 for now."
D_PR3_COND = "If PR#3 is already merged, please revert it."
D_PR4 = "PR#4 is approved but not merged yet. Any objection to merging it?"
D_PR4_HOLD = "Let's hold PR#4 until the audit is done."
D_PR4_GO = "The audit is done, so let's move forward with merging PR#4."
D_PR7 = "Please look at PR#7 when you can."
D_CHAT = "Good morning everyone, have a nice day."
B_BUG = "Hi team, several customers are reporting issues with Mini. CUST-0001 is affected."
ART = "@here Explore the PyOD library for outliers. https://github.com/yzhao062/pyod"

SLACK = [
    ("20260305-0-a0001", ANN, "2026-03-05T09:00:00", R_SHARE) + PLAN,
    ("20260305-1-a0002", BEN, "2026-03-05T09:05:00", R_BEN) + PLAN,
    ("20260305-2-a0003", DEE, "2026-03-05T09:10:00", R_DEE) + PLAN,
    ("20260305-3-a0004", ANN, "2026-03-05T09:15:00", R_ANN) + PLAN,
    ("20260310-0-b0001", EVE, "2026-03-10T09:00:00", C_RIVAL) + PLAN,
    ("20260310-1-b0002", BEN, "2026-03-10T09:05:00", "Interesting. What do they lack?") + PLAN,
    ("20260312-0-b0003", EVE, "2026-03-12T09:00:00", C_DEMO) + DEV,
    ("20260315-0-c0001", ANN, "2026-03-15T09:00:00", Q_QA) + PLAN,
    ("20260315-1-c0002", BEN, "2026-03-15T09:05:00", Q_TEAM) + PLAN,
    ("20260316-0-c0003", ANN, "2026-03-16T09:00:00", Q_CUST) + PLAN,
    ("20260316-1-c0004", BEN, "2026-03-16T09:05:00", Q_TEAM) + PLAN,
    ("20260316-2-c0005", ANN, "2026-03-16T09:10:00", Q_STRENGTH) + PLAN,
    ("20260317-0-c0006", ANN, "2026-03-17T09:00:00", CUR_ISSUE) + PLAN,
    ("20260320-0-d0001", EVE, "2026-03-20T09:00:00", D_PR3) + DEV,
    ("20260320-1-d0002", BEN, "2026-03-20T09:05:00", D_PR3_HOLD) + DEV,
    ("20260320-2-d0003", BEN, "2026-03-20T09:10:00", D_PR3_COND) + DEV,
    ("20260322-0-e0001", EVE, "2026-03-22T09:00:00", D_CHAT) + DEV,
    ("20260325-0-f0001", EVE, "2026-03-25T09:00:00", D_PR4) + DEV,
    ("20260325-1-f0002", BEN, "2026-03-25T09:05:00", D_PR4_HOLD) + DEV,
    ("20260402-0-f0003", BEN, "2026-04-02T09:00:00", D_PR4_GO) + DEV,
    ("20260404-0-f0004", EVE, "2026-04-04T09:00:00", D_PR7) + DEV,
    ("20260405-0-g0001", EVE, "2026-04-05T09:00:00", ART) + DEV,
    ("20260406-0-h0001", ANN, "2026-04-06T09:00:00", B_BUG) + BUGCH,
    ("20260406-1-h0002", BEN, "2026-04-06T09:05:00",
     "The crash is fixed in https://github.com/salesforce/Mini/pull/10") + BUGCH,
]


def msg(mid, author, ts, text, channel, name) -> SlackMessage:
    return SlackMessage(id=mid, channel_id=channel, channel_name=name, user_id=author, ts=ts, text=text,
                        mentions=parse_mentions(text), is_bot=author == BOT, in_directory=author in EMPLOYEES,
                        bug_channel_owner="ben" if name.startswith("bug-") else None)


# ---------------------------------------------------------------- PRs
def pr(number, title, *, repo=("salesforce", "Mini"), author=EVE, merged=False, state="open", states=("APPROVED",),
       summary="s", created="2026-03-01T00:00:00") -> PullRequest:
    link = f"https://github.com/{repo[0]}/{repo[1]}/pull/{number}"
    reviews = tuple(Review(i, st, BEN, "ok", f"2026-03-02T00:00:0{i}", True) for i, st in enumerate(states))
    return PullRequest(id=f"github_com_{repo[0]}_{repo[1]}_pull_{number}", link=link, number=str(number), title=title,
                       summary=summary, author=author, in_directory=True, merged=merged, mergeable=None,
                       merged_raw=str(merged), mergeable_raw="", state=state, created_at=created, reviews=reviews)


PRS = [
    pr(1, "Integrate LSTM Forecasting", merged=True, state="closed"),
    pr(2, "Add Metrics Export", merged=True, state="closed"),
    pr(3, "Real-time Alert Delivery", summary="Push alerts to the dashboard as they fire."),
    pr(4, "Alert Throttling", summary="Limit how often one alert can fire."),
    pr(5, "Revert PR# 1: Integrate LSTM Forecasting", merged=True, state="closed", created="2026-04-10T00:00:00",
       summary="This PR reverts PR#1 because of a change in product direction. The model is no longer needed."),
    pr(6, "Predictive Model Evaluation", summary="Evaluate the forecasting model on held-out data.",
       states=("CHANGES_REQUESTED",)),
    pr(7, "Integrate LSTM Forecasting", author=BEN, states=("CHANGES_REQUESTED",)),  # same title, other author
    pr(8, "Unrelated Cleanup", summary="Remove dead code.", state="closed", merged=True),
    pr(9001, "Support RNN Streams", repo=("apache", "kafka"), merged=True, state="closed"),
    pr(10, "Fix Crash In Export", merged=True, state="closed", summary="Fixes the export crash."),  # a bug-fix PR
]

URLS = [
    UrlRecord("www_rivalco_com_demo", "https://www.rivalco.com/demo", ("https://www.rivalco.com/demo",),
              ("A demo showing how RivalCo works.",), (0,)),
    UrlRecord("sf-internal_slack_com_archives_Mini_demo_1", "https://sf-internal.slack.com/archives/Mini/demo_1",
              ("https://sf-internal.slack.com/archives/Mini/demo_1",), ("A demo video of Mini.",), (1,)),
]


def records(slack=SLACK, prs=None, meetings=None, chats=None) -> ProductRecords:
    docs = [
        Document("mini_market_research_report", ANN, "Market Research Report", DOC_LINK, "2026-03-01T00:00:00",
                 "Executive Summary: Mini is a product.", None),
        Document("final_mini_market_research_report", ANN, "Market Research Report", FINAL_LINK,
                 "2026-03-06T00:00:00", "Executive Summary: Mini is a product. Final.", FEEDBACK),
    ]
    mts = meetings if meetings is not None else [
        meeting(M1, "2026-03-01T10:00:00", "product-planning", [ANN, BEN, CY, SAM1, SAM2, EVE], M1_SEGS),
        meeting(M2, "2026-03-02T10:00:00", "Market Research Report", [ANN, DEE, BEN], M2_SEGS),
    ]
    cs = chats if chats is not None else [MeetingChat(f"{M2}_chat", M2_CHAT, "2026-03-02T10:30:00", (), M2, "id")]
    return ProductRecords(
        product=PRODUCT, messages=[msg(*r) for r in slack], documents=docs, meetings=mts, chats=cs,
        prs=prs if prs is not None else list(PRS), urls=list(URLS),
        employees=dict(EMPLOYEES), customers=dict(CUSTOMERS))


# ---------------------------------------------------------------- scripted answers
def cite(rid, quote, **kw):
    return {"record_id": rid, "quote": quote, **kw}


def meeting_json(document_reviews=(), features=(), contributions=(), development_links=(), feature_events=()):
    return json.dumps({"document_reviews": list(document_reviews), "features": list(features),
                       "contributions": list(contributions), "development_links": list(development_links),
                       "feature_events": list(feature_events)})


def session_json(reviews=(), claims=(), pr_decisions=()):
    return json.dumps({"reviews": list(reviews), "claims": list(claims), "pr_decisions": list(pr_decisions)})


def membership_json(*triples):
    return json.dumps({"assignments": [{"pr": p, "feature": f, "quote": q} for p, f, q in triples]})


ALERTS = {"local_id": "f1", "title": "real-time alerts", "details": ["WebSockets for pushing alerts"],
          "citations": [cite(f"{M1}#5", "use WebSockets for pushing alerts")]}
PRED = {"local_id": "f2", "title": "predictive analytics", "details": [],
        "citations": [cite(f"{M1}#4", "predictive analytics")]}
M1_ANSWER = meeting_json(
    features=[ALERTS, PRED],
    contributions=[
        {"feature_ref": "f1", "segment_ref": f"{M1}#5", "kind": "lead",
         "citations": [cite(f"{M1}#5", "I'll lead real-time alerts")]},
        {"feature_ref": "f1", "segment_ref": f"{M1}#6", "kind": "support",
         "citations": [cite(f"{M1}#6", "I can help with testing")]},
        {"feature_ref": "f2", "segment_ref": f"{M1}#7", "kind": "assigned",
         "citations": [cite(f"{M1}#7", "I'll handle predictive analytics")]},
        {"feature_ref": "f1", "segment_ref": f"{M1}#9", "kind": "support",
         "citations": [cite(f"{M1}#9", "I will do some documentation")]},
    ],
    development_links=[
        {"feature_ref": "f2", "pr": "Integrate LSTM Forecasting",
         "citations": [cite(f"{M1}#7", "'Integrate LSTM Forecasting'")]},
        {"feature_ref": "f2", "pr": "Add Metrics Export", "citations": [cite(f"{M1}#8", "'Add Metrics Export' PR")]},
    ],
    feature_events=[
        {"feature_ref": "f1", "kind": "discussed", "reason": None, "pr": None, "segment_ref": f"{M1}#4",
         "citations": [cite(f"{M1}#4", "plan real-time alerts")]},
        {"feature_ref": "f2", "kind": "implemented", "reason": None, "pr": "Add Metrics Export",
         "segment_ref": f"{M1}#8", "citations": [cite(f"{M1}#8", "is complete and merged")]},
    ])
M2_ANSWER = meeting_json(document_reviews=[
    {"segment_ref": f"{M2}#5", "kind": "suggestion", "section": "Competitive Analysis", "text": "Add a comparison table",
     "citations": [cite(f"{M2}#5", "add a comparison table")]},
    {"segment_ref": f"{M2}#6", "kind": "suggestion", "section": "Executive Summary",
     "text": "More detail on the algorithms", "citations": [cite(f"{M2}#6", "needs more detail on the algorithms")]}])

REVIEWS = [
    {"message_id": "20260305-1-a0002", "document_ref": "Market Research Report", "kind": "suggestion",
     "section": "Product Description", "text": "Elaborate on the advanced algorithms",
     "citations": [cite("20260305-1-a0002", "elaborate on the advanced algorithms")]},
    {"message_id": "20260305-2-a0003", "document_ref": DOC_LINK, "kind": "suggestion", "section": "Market Trends",
     "text": "Mention regulatory changes", "citations": [cite("20260305-2-a0003", "mention regulatory changes")]},
]
RIVAL_CLAIMS = [
    {"message_id": "20260310-0-b0001", "kind": "competitor_feature", "subject_ref": "RivalCo",
     "text": "RivalCo offers live dashboards", "customer_ids": [], "release_scope": "unknown",
     "citations": [cite("20260310-0-b0001", "live dashboards")]},
    {"message_id": "20260310-0-b0001", "kind": "competitor_weakness", "subject_ref": "RivalCo",
     "text": "RivalCo has limited export options", "customer_ids": [], "release_scope": "unknown",
     "citations": [cite("20260310-0-b0001", "limited export options")]},
]
QA_CLAIMS = [
    {"message_id": "20260315-0-c0001", "kind": "qa_issue", "subject_ref": "Mini",
     "text": "Exports crash on large files", "customer_ids": [], "release_scope": "previous",
     "citations": [cite("20260315-0-c0001", "exports crash on large files")]},
]
CUST_CLAIMS = [
    {"message_id": "20260316-0-c0003", "kind": "customer_issue", "subject_ref": "Mini",
     "text": "Limited CRM integration", "customer_ids": ["CUST-0001"], "release_scope": "previous",
     "citations": [cite("20260316-0-c0003", "limited CRM integration")]},
    {"message_id": "20260316-2-c0005", "kind": "customer_strength", "subject_ref": "Mini",
     "text": "Lead scoring helps prioritize leads", "customer_ids": ["CUST-0002"], "release_scope": "previous",
     "citations": [cite("20260316-2-c0005", "lead scoring is helping them prioritize leads")]},
]
CURRENT_CLAIM = [
    {"message_id": "20260317-0-c0006", "kind": "customer_issue", "subject_ref": "Mini", "text": "Dashboards are slow",
     "customer_ids": ["CUST-0001"], "release_scope": "previous",  # the model over-reads: no previous-release cue
     "citations": [cite("20260317-0-c0006", "dashboards are slow")]},
]
PR3_DECISIONS = [
    {"message_id": "20260320-0-d0001", "prs": ["3"], "action": "question", "reason": "product_direction",
     "conditional": False, "citations": [cite("20260320-0-d0001", "Should we merge it?")]},
    {"message_id": "20260320-1-d0002", "prs": ["3"], "action": "withhold", "reason": "product_direction",
     "conditional": False, "citations": [cite("20260320-1-d0002", "do not merge PR#3")]},
    {"message_id": "20260320-2-d0003", "prs": ["3"], "action": "revert", "reason": None, "conditional": True,
     "citations": [cite("20260320-2-d0003", "please revert it")]},
]
PR4_HOLD = [{"message_id": "20260325-1-f0002", "prs": ["4"], "action": "withhold", "reason": "other",
             "conditional": False, "citations": [cite("20260325-1-f0002", "hold PR#4")]}]
PR4_GO = [{"message_id": "20260402-0-f0003", "prs": ["4"], "action": "resume", "reason": None, "conditional": False,
           "citations": [cite("20260402-0-f0003", "move forward with merging PR#4")]}]


class ScriptedChat:
    """Stands in for inherit.llm.chat. `script` maps a marker substring of the user message to a response text (or a
    list consumed in order). Unmatched calls get an empty answer of the right shape."""

    def __init__(self, script: dict | None = None, membership=None):
        self.script = script or {}
        self.membership = membership
        self.calls: list[dict] = []

    def __call__(self, messages, *, arm, max_tokens=800, response_format=None, **kw):
        user = messages[-1]["content"]
        self.calls.append({"messages": messages, "arm": arm, "max_tokens": max_tokens, "user": user,
                           "response_format": response_format})
        system = messages[0]["content"]
        text, toks = None, 50
        if "PR_MEMBERSHIP_TASK" in system:
            text = self.membership(user) if callable(self.membership) else (self.membership or membership_json())
        else:
            for marker, resp in self.script.items():
                if marker in user:
                    if isinstance(resp, list):
                        resp = resp.pop(0) if len(resp) > 1 else resp[0]
                    text = resp
                    break
            if text is None:
                text = meeting_json() if "five arrays" in system else session_json()
        if isinstance(text, tuple):
            text, toks = text
        return SimpleNamespace(message=SimpleNamespace(content=text), prompt_tokens=100, completion_tokens=toks,
                               cost_usd=0.0001, latency_s=0.0)

    def kind(self, name):
        key = {"meeting": "five arrays", "session": "three arrays", "membership": "PR_MEMBERSHIP_TASK"}[name]
        return [c for c in self.calls if key in c["messages"][0]["content"]]


def standard_chat() -> ScriptedChat:
    quotes = {"Real-time Alert Delivery": ("alerts", "Push alerts to the dashboard"),
              "Alert Throttling": ("alerts", "Limit how often one alert can fire"),
              "Predictive Model Evaluation": ("predictive", "Evaluate the forecasting model")}

    def membership(user):
        lines = user.splitlines()
        out = []
        for title, (topic, quote) in quotes.items():
            ref = next((ln.split(":")[0] for ln in lines if ln.startswith("P") and title in ln), None)
            fref = next((ln.split(":")[0] for ln in lines if ln.startswith("F") and topic in ln), None)
            if ref and fref:
                out.append((ref, fref, quote))
        return membership_json(*out)

    return ScriptedChat(
        script={
            f"MEETING: {M1}": M1_ANSWER,
            f"MEETING: {M2}": M2_ANSWER,
            "[20260305-1-a0002]": session_json(reviews=REVIEWS),
            "[20260310-0-b0001]": session_json(claims=RIVAL_CLAIMS),
            "[20260315-0-c0001]": session_json(claims=QA_CLAIMS),
            "[20260316-0-c0003]": session_json(claims=CUST_CLAIMS),
            "[20260317-0-c0006]": session_json(claims=CURRENT_CLAIM),
            "[20260320-0-d0001]": session_json(pr_decisions=PR3_DECISIONS),
            "[20260325-1-f0002]": session_json(pr_decisions=PR4_HOLD),
            "[20260402-0-f0003]": session_json(pr_decisions=PR4_GO),
        },
        membership=membership)

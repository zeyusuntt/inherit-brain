import ast
import collections
import re
from pathlib import Path

import pytest

from eval import questions as Q
from inherit.contracts import QuestionKey
from inherit.data.split import split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
needs_data = pytest.mark.skipif(not (RAW / "products").is_dir(), reason="data/raw is absent")


def test_mini_questions_have_stable_keys(mini_raw):
    qs = Q.load_all_questions(mini_raw)
    assert qs
    ans = [q for q in qs if q.key.kind == "ans"]
    unans = [q for q in qs if q.key.kind == "unans"]
    assert [q.key.index for q in ans] == list(range(len(ans)))
    assert [q.key.index for q in unans] == list(range(len(unans)))
    assert all(q.type == "unanswerable" and q.gold is None and q.citations == [] for q in unans)
    assert all(q.type != "unanswerable" and q.gold is not None for q in ans)
    assert str(qs[0].key) == "mini/ans/0" and qs[0].product == "mini"
    assert QuestionKey.parse(str(ans[-1].key)) == ans[-1].key


def test_unknown_split_refused(mini_raw):
    with pytest.raises(ValueError):
        Q.load_questions("test", mini_raw)


@needs_data
def test_counts_and_order_on_real_data():
    dev, held = Q.load_questions("dev"), Q.load_questions("held")
    c = lambda qs, k: sum(q.key.kind == k for q in qs)  # noqa: E731
    assert (c(dev, "ans"), c(dev, "unans")) == (156, 155)
    assert (c(held, "ans"), c(held, "unans")) == (659, 544)
    names = sorted(p.stem for p in (RAW / "products").glob("*.json"))
    dev_names, held_names = split(names)
    assert list(dict.fromkeys(q.key.product for q in dev)) == dev_names
    assert list(dict.fromkeys(q.key.product for q in held)) == held_names
    assert len({q.key for q in dev + held}) == len(dev + held)


@needs_data
def test_seen_unseen_counts_from_question_text_only():
    held = Q.load_questions("held")
    ans = [q for q in held if q.key.kind == "ans"]
    assert collections.Counter(q.seen for q in ans) == {True: 416, False: 243}
    assert all(q.seen is None for q in Q.load_questions("dev"))


@needs_data
def test_template_normaliser_is_the_split_script_normaliser():
    """Run the original analysis/herb_split_templates.py `norm` against ours on every real question."""
    tree = ast.parse((ROOT / "analysis" / "herb_split_templates.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "norm")
    ns: dict = {"re": re}
    exec(compile(ast.Module([fn], []), "herb_split_templates.py", "exec"), ns)
    for q in Q.load_all_questions():
        assert Q.template(q.text, q.product) == ns["norm"](q.text, q.product)


@needs_data
def test_dev_products_match_the_script_constant():
    dev = {q.key.product for q in Q.load_questions("dev")}
    assert dev == {"ActionGenie", "AnomalyForce", "ContentForce", "KnowledgeForce", "LeadForce", "TrendForce"}


def test_only_questions_py_reads_question_files():
    """Rule 2. These names appear only in the question loader and in the deny lists that refuse them."""
    allowed = {"eval/questions.py", "inherit/ioguard.py", "inherit/data/scan.py"}
    offenders = []
    for p in (ROOT / "src").rglob("*.py"):
        rel = p.relative_to(ROOT / "src").as_posix()
        text = p.read_text()
        if rel not in allowed and re.search(r"answerable_questions|ground_truth", text):
            offenders.append(rel)
    assert offenders == []

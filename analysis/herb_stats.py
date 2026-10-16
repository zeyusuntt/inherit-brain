import json, sys, os, collections, re
D = sys.argv[1]; P = os.path.join(D, "products")
prods = {f[:-5]: json.load(open(os.path.join(P, f))) for f in sorted(os.listdir(P))}
q0 = next(iter(prods.values()))["answerable_questions"][0]
print("question keys:", list(q0.keys()))
tot = collections.Counter(); per = {}
for name, d in prods.items():
    c = collections.Counter(q.get("type") for q in d["answerable_questions"])
    per[name] = (len(d["answerable_questions"]), len(d["unanswerable_questions"]), dict(c))
    tot.update(c)
print("answerable", sum(v[0] for v in per.values()), "unanswerable", sum(v[1] for v in per.values()))
print("types", dict(tot))
for n, v in per.items(): print(n, v)
# artifact counts
art = collections.Counter()
for d in prods.values():
    for k in ["slack","documents","meeting_transcripts","meeting_chats","urls","prs"]:
        art[k] += len(d.get(k, []))
print("artifacts", dict(art), sum(art.values()))

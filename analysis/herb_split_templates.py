import json, sys, os, collections, re
D = sys.argv[1]; P = os.path.join(D, "products")
DEV = {'ActionGenie','AnomalyForce','ContentForce','KnowledgeForce','LeadForce','TrendForce'}
prods = {f[:-5]: json.load(open(os.path.join(P, f))) for f in sorted(os.listdir(P))}
def norm(q, name):
    q = q.replace(name, "<P>")
    q = re.sub(r"eid_[0-9a-f]+|EMP_\d+", "<E>", q); q = re.sub(r"https?://\S+", "<URL>", q)
    q = re.sub(r"\b[A-Za-z]*Force[A-Za-z]*\b", "<X>", q); q = re.sub(r"\d+", "<N>", q)
    q = re.sub(r"\b(TechCorp|FusionTech|BlueWave|DataSolutions|NextGenTech|CloudSync|ComputeWorks|InnovateAI|AITransformers|SmartData)\b", "<C>", q)
    return q.strip().rstrip("?.").lower()
dev_t = set(); 
for n in DEV:
    for q in prods[n]["answerable_questions"]: dev_t.add(norm(q["question"], n))
seen = collections.Counter(); unseen = collections.Counter(); n_held = 0
for n, d in prods.items():
    if n in DEV: continue
    for q in d["answerable_questions"]:
        n_held += 1
        (seen if norm(q["question"], n) in dev_t else unseen)[q["type"]] += 1
print("held-out answerable", n_held, "seen-template", sum(seen.values()), dict(seen), "unseen-template", sum(unseen.values()), dict(unseen))
dev_n = sum(len(prods[n]["answerable_questions"]) for n in DEV); dev_u = sum(len(prods[n]["unanswerable_questions"]) for n in DEV)
print("dev answerable", dev_n, "dev unanswerable", dev_u, "held unanswerable", sum(len(d["unanswerable_questions"]) for n, d in prods.items() if n not in DEV))
# citation counts by type (how many artifacts a gold answer spans)
cc = collections.defaultdict(list)
for n, d in prods.items():
    for q in d["answerable_questions"]: cc[q["type"]].append(len(q["citations"]))
for t, v in cc.items():
    v.sort(); print(t, "citations median", v[len(v)//2], "p90", v[int(len(v)*.9)], "max", v[-1])

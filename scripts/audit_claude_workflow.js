export const meta = {
  name: 'dev-audit-claude-auditor',
  description: 'Auditor 2 of the dev acceptance audit: one fresh sonnet agent per batch, packet-only input, verdict file per batch',
  phases: [{ title: 'Audit', detail: 'one sonnet agent per batch, two at a time' }],
}

// args: { rd: absolute round dir, src: 'batches' | 'third_batches', who: 'claude', names: [batch names] }
const { rd, src, who, names } = args
const RESULT = {
  type: 'object',
  properties: { written: { type: 'boolean' }, packets: { type: 'integer' }, note: { type: 'string' } },
  required: ['written', 'packets'],
}

function prompt(name) {
  const inFile = `${rd}/${src}/${name}.txt`
  const outFile = `${rd}/verdicts/${who}/${name}.json`
  return `You are an independent auditor for a research benchmark's acceptance check. Your judgments are compared with another model's, so every packet needs your own careful reading.

Read exactly one file with the Read tool: ${inFile}
It holds audit instructions followed by audit packets. Do not open, search or list any other file or directory, do not use Bash or any script, and do not use the network. Judge each packet only from its own text, following the instructions exactly (verdict values, required fields, recall item fields).

Write your complete answer, exactly the JSON the instructions ask for (one entry per packet in the file), with the Write tool to: ${outFile}
Then return written=true and the number of packets you judged.`
}

phase('Audit')
const out = []
for (let i = 0; i < names.length; i += 2) {
  const pair = names.slice(i, i + 2)
  const res = await parallel(pair.map(n => () =>
    agent(prompt(n), { label: `${who}:${n}`, phase: 'Audit', model: 'sonnet', schema: RESULT })
      .then(r => ({ name: n, ...(r || { written: false, packets: 0, note: 'no result' }) }))))
  out.push(...res.filter(Boolean))
  log(`${out.length}/${names.length} batches`)
}
const missing = out.filter(r => !r.written).map(r => r.name)
if (missing.length) log(`not written: ${missing.join(', ')}`)
return { done: out.length, missing }

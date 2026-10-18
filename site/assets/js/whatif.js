// "If this person leaves": data/whatif.json, one product at a time, owners (default) or Engineering Leads.
import { initTheme, getJSON, esc, escClean, int } from "./common.js";

initTheme();
const $ = (id) => document.getElementById(id);
let data, tab = "owners";

function storedProduct() { try { return localStorage.getItem("whatif-product"); } catch { return null; } }
function storeProduct(p) { try { localStorage.setItem("whatif-product", p); } catch { /* storage unavailable */ } }

const KIND = { Bug: "Bug", PR: "PR", Feature: "Feature" };

function itemHTML(it) {
  return `<li><span class="k">${esc(KIND[it.kind] || it.kind)}</span><span>${escClean(it.title || it.id)}<code>${esc(it.id)}</code></span></li>`;
}
function affectedHTML(a) {
  const steps = (a.path || []).map((p, i) => {
    const via = i > 0 ? `<span class="via">depends on${a.rules?.[i - 1] ? `, via <code>${esc(a.rules[i - 1])}</code>,` : ""}</span>` : "";
    return `${via}<span>${esc(KIND[p.kind] || p.kind)}: ${escClean(p.title || p.id)}</span>`;
  }).join("");
  return `<li><span class="k">${esc(KIND[a.affected.kind] || a.affected.kind)}</span><span>${escClean(a.affected.title || a.affected.id)}<code>${esc(a.affected.id)}</code>
    <span class="dep" aria-label="Dependency path">${steps}</span></span></li>`;
}

function personHTML(x, i, openIdx) {
  const c = x.counts;
  const num = (n, label) => `<span class="c${n === 0 ? " zero" : ""}"><small>${label}</small>${int(n)}</span>`;
  const any = x.newly_unowned.length || x.affected.length;
  const body = any ? `
      ${x.newly_unowned.length ? `<div><h4>Newly unowned (${int(x.newly_unowned.length)})</h4><ul class="items">${x.newly_unowned.map(itemHTML).join("")}</ul></div>` : ""}
      ${x.affected.length ? `<div><h4>Affected (${int(x.affected.length)}), with the dependency path from the affected item to the unowned one</h4><ul class="items">${x.affected.map(affectedHTML).join("")}</ul></div>` : ""}`
    : `<p class="nothing">Nothing open has this person as its only owner, so nothing is newly unowned or affected.</p>`;
  const n = x.newly_unowned.length + x.affected.length;
  const toggle = any ? `<span class="toggle"><span class="show">Show ${n === 1 ? "the item" : `the ${int(n)} items`}</span><span class="hide">Hide the items</span></span>` : "";
  return `<li><details${i === openIdx ? " open" : ""}>
    <summary class="person-sum"><span class="who"><b>${escClean(x.person.name)}</b><span>${escClean(x.person.role)}, <code>${esc(x.person.eid)}</code></span>${toggle}</span>
      ${num(c.newly_unowned, "newly unowned")}${num(c.affected, "affected")}${num(c.already_unowned, "already unowned")}</summary>
    <div class="person-body">${body}${x.bound_hit ? `<p class="micro">The dependency search hit its ${x.max_hops}-hop bound.</p>` : ""}</div></details></li>`;
}

// Open the first person whose departure affects other work, else the first with anything newly unowned.
function openIndex(list) {
  const a = list.findIndex((x) => x.affected.length);
  return a >= 0 ? a : list.findIndex((x) => x.newly_unowned.length);
}

function render() {
  const p = $("wi-product").value;
  const prod = data.products[p];
  const list = prod[tab] || [];
  $("tab-note").textContent = tab === "owners"
    ? "The five people in this product whose departure leaves the most open items without an owner."
    : `The three of the product's ${int(prod.leads_in_product)} Engineering Leads whose departure leaves the most open items unowned. Many own nothing open; their zeros are shown as they are.`;
  $("people-panel").setAttribute("aria-labelledby", tab === "owners" ? "tab-owners" : "tab-leads");
  $("people-panel").innerHTML = list.length
    ? `<div class="wi-head" aria-hidden="true"><span>Person</span><span>Newly unowned</span><span>Affected</span><span>Already unowned</span></div><ul class="people">${list.map((x, i) => personHTML(x, i, openIndex(list))).join("")}</ul>`
    : `<p class="nothing">No people in this list for ${esc(p)}.</p>`;
}

function setTab(t) {
  tab = t;
  $("tab-owners").setAttribute("aria-selected", String(t === "owners"));
  $("tab-leads").setAttribute("aria-selected", String(t === "leads"));
  $("tab-owners").tabIndex = t === "owners" ? 0 : -1;
  $("tab-leads").tabIndex = t === "leads" ? 0 : -1;
  render();
}

async function main() {
  try {
    data = await getJSON("data/whatif.json");
  } catch (e) {
    $("people-panel").innerHTML = `<p class="state-msg err">Could not load data/whatif.json (${esc(e.message)}). Reload the page to try again.</p>`;
    $("defs").innerHTML = "";
    return;
  }
  const names = Object.keys(data.products);
  $("wi-product").innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  const h = new URLSearchParams(location.hash.slice(1)).get("p");  // deep link: whatif.html#p=<product>
  const s = h && data.products[h] ? h : storedProduct();
  const first = names.find((n) => data.products[n].owners.some((o) => o.counts.affected > 0)) || names[0];
  $("wi-product").value = s && data.products[s] ? s : first;
  $("defs").innerHTML = `<p><b>Definitions.</b> ${escClean(data.definitions)}</p><p><b>Basis.</b> ${escClean(data.basis)}</p><p><b>Who is listed.</b> ${escClean(data.owners_note)}</p>`;
  $("wi-product").addEventListener("change", (e) => { storeProduct(e.target.value); render(); });
  $("tab-owners").addEventListener("click", () => setTab("owners"));
  $("tab-leads").addEventListener("click", () => setTab("leads"));
  document.querySelector(".wi-tabs").addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const t = tab === "owners" ? "leads" : "owners";
      setTab(t);
      $(t === "owners" ? "tab-owners" : "tab-leads").focus();
    }
  });
  setTab("owners");
}
main();

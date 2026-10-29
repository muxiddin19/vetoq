# Merges Are Not Splits — Technical Report

Companion to the ICDE 2027 submission "Merges Are Not Splits: Operator-Aware Entity Equivalence for
Semantic Queries over Dirty Keys" (draft v1.1, 2026-10-24). Section, lemma, theorem, corollary and table
numbers in brackets refer to the paper and are numbered exactly as there (Lemmas 1–8, Theorems 1–2,
Proposition 1, Corollaries 1–4).

## A. Notation [§II]

r: n tuples; K: dirty key; e(t): entity of tuple t (ground truth, not in r); κ: an equivalence (partition of
tuples). Contingency n_{c,e} = number of tuples of entity e in class c.

- **Representative class.** c\*(e) = argmax_c n_{c,e}. Ties go to the smallest class (among classes with the
  same n_{c,e}, the one with the fewest foreign tuples, i.e. the purest), then to the tied class holding
  e's earliest tuple in tuple order. The rule depends only on the partition and the tuple order, never on class
  labels (implementation: `exp_t1/contingency.py`, `rep_class`). For two relations R, S, κ partitions R ∪ S
  and c\*(e) is taken on the contingency table of R ∪ S.
- **Split mass** s(e) = |e| − n_{c\*(e),e}; **merge mass** m(e) = |c\*(e)| − n_{c\*(e),e}. Per table
  T ∈ {R, S}: s_T(e) = |e|_T − n^T_{c\*(e),e}, m_T(e) = |c\*(e)|_T − n^T_{c\*(e),e}.
- **Mixed mass** X = number of tuples in classes that meet two or more entities (`operator_errors`, column `X`).
- **Collapse-free:** every class meets one entity, equivalently X = 0. **Split-free:** every entity meets one
  class.
- m ≡ 0 does **not** imply X = 0. Example: entity a has 9 tuples in class c1 and 1 in c3; entity b has 5 in c2
  and 1 in c3. Then c\*(a) = c1, c\*(b) = c2, m(a) = m(b) = 0, but c3 mixes a and b (X = 2), and a join whose
  two sides both contain c3's tuples produces a false pair.
- **Error names.** *COUNT error* (resp. SUM error): mean over entities of the per-entity relative error, for
  COUNT |m(e) − s(e)|/|e| (`operator_errors`, `COUNT_err`). *Total COUNT error*: Σ_e |m(e) − s(e)| (Corollary 3);
  the scale table and the Hospital chooser report it divided by n. *L1 error*: relative L1 error of a query's
  answer vector (Hard-multi pipeline, §VII-D/E).

## B. Operator error calculus [§III]

Lemmas 1, 3, 4 and 7 and Corollary 3 are proved in the paper. Proofs of Lemma 2, Lemma 6 and Corollaries 1
and 2 follow. Implementation and an independent execution path are in `exp_t1/contingency.py`; the check in
`exp_t1/run_ec_ea.py` (log `gpu_out/t1_ec_ea.txt`) found no mismatch in 160 (table, equivalence) cases for
Lemmas 1, 2, 3, 4 and 6. Both paths use the same representative-class rule, so the check covers the
arithmetic of the formulas, not the choice of c\*.

**Proof of Lemma 2 (SUM per entity).** Let A be numeric, of any sign. The reported answer for e is
Σ_{t ∈ c\*(e)} A(t) = Σ_{t ∈ c\*(e), e(t) = e} A(t) + Σ_{t ∈ c\*(e), e(t) ≠ e} A(t), and the truth is
Σ_{e(t) = e} A(t) = Σ_{t ∈ c\*(e), e(t) = e} A(t) + Σ_{e(t) = e, t ∉ c\*(e)} A(t). Subtracting, the error is the
A-mass of the foreign tuples in c\*(e) minus the A-mass of e's tuples outside c\*(e). No sign assumption is
used. If A ≥ 0, the first term is ≥ 0 and the second ≤ 0, so merges and splits push the error in opposite
directions. ∎

**Proof of Lemma 6 (Selection).** Let the query value q occur in r in a tuple t_q of entity e_q.
`WHERE K ≈ q` returns the class κ(t_q). If κ(t_q) = c\*(e_q), the returned tuples of other entities are the
foreign tuples of c\*(e_q), m(e_q) of them, and the tuples of e_q not returned are those outside c\*(e_q),
s(e_q) of them. ∎ (If κ(t_q) ≠ c\*(e_q), the same argument applies with κ(t_q) in place of c\*(e_q).)

**Proof of Corollary 1.** (a) Suppose e and e′ each fill one pure class (all their tuples, no foreign tuple).
Merging the two classes, with no other change, adds exactly the pairs (r, s) with {e(r), e(s)} = {e, e′},
i.e. |e|_R|e′|_S + |e′|_R|e|_S false pairs; without purity, pairs with the other entities in those classes
would be added as well. (b) Let σ_T(e) = s_T(e)/|e|_T, with c\*(e) taken on R ∪ S. A missed pair (r, s) of e
has κ(r) ≠ κ(s), so r and s are not both in c\*(e). The pairs of e with at least one end outside c\*(e) number
|e|_R|e|_S − n^R_{c\*,e} n^S_{c\*,e} = |e|_R|e|_S (1 − (1 − σ_R)(1 − σ_S)) ≤ |e|_R|e|_S (σ_R + σ_S), and the missed
pairs are a subset. No assumption on merges is needed. The common class matters: if σ_R and σ_S were measured
against different per-table representative classes, both could be 0 while every pair of e is missed. ∎

**Proof of Corollary 2.** Assume the hypotheses of Lemma 7: |e|_R, |e|_S > 0 and e's representative classes in
R and in S coincide, with signed ρ_T = (m_T − s_T)/|e|_T. Since that class is e's representative class in T
and |e|_T > 0, it holds at least one tuple of e in T, so s_T < |e|_T and ρ_T > −1. Expand
(1 + ρ_R)(1 + ρ_S) − 1 = ρ_R + ρ_S + ρ_Rρ_S. With ρ_R, ρ_S > 0 the product term is positive (super-additive).
With −1 < ρ_R, ρ_S ≤ 0, write a = −ρ_R, b = −ρ_S ∈ [0, 1); then the error is −(a + b − ab), whose magnitude is
at most a + b, and (1 − a)(1 − b) > 0 gives ρ_{R⋈S} > −1. ∎

**Proof of Corollary 3 (Profile bound).** By Lemma 1 the per-entity error is m(e) − s(e), and
|m − s| ≤ m + s, with equality iff min(m(e), s(e)) = 0; this holds for every e if κ is collapse-free (m ≡ 0)
or split-free (s ≡ 0). ∎

## C. Lemma 5 (re-join after decomposition) [§III, §V]

Assume e → Y holds on the clean data, so each entity e has a clean Y-value y(e). Decompose r by K → Y into
R1(κ, Y) and R2(κ, R ∖ Y). R2 holds one tuple per tuple t of r with κ(t) and t[R ∖ Y]. R1 holds one tuple
(c, π(c)) per class, where π(c)[B] is a most frequent value of B ∈ Y in c.
(a) The re-join R1 ⋈_κ R2 has exactly n tuples: each R2 tuple matches the single R1 tuple of its class.
(b) If κ is collapse-free and in each class and for each attribute B ∈ Y the clean B-value of its (single) entity is the strict plurality,
π(c) = y(e) for the class's entity e, so every tuple receives y(e(t)). The argument is per class, so an
entity spanning several classes is reconstructed in each.
(c) If a class contains entities e ≠ e′ with y(e)[B] ≠ y(e′)[B], π(c)[B] differs from at least one of them,
and all of that entity's tuples in c receive it.
(d) If R1 keeps every distinct (κ, Y)-tuple, a tuple of class c joins d(c) tuples (d(c) = number of distinct
Y-tuples in c), giving Σ_t (d(κ(t)) − 1) spurious tuples. ∎
K → Y need not hold on r: the representative replaces violating minority values.

**Splits and the re-join.** (b) needs the plurality condition in *every* class, including the fragments a
split creates. For collapse-free κ, a tuple receives a wrong Y-value iff the plurality Y-value of its class
is not clean (up to ties). A split can violate the condition: a fragment isolated from its entity's
representative class can have a dirty plurality. When the Context Veto uses an attribute of Y as context and
(i) of §E fails, the fragments the veto splits off can consist of tuples whose context value is dirty, and then
their plurality is dirty. (Counterexample to "exactly": a clean-context variant whose only key edges run to a
dirty-context tuple is also split off.) Example (Table I of the paper; exact matching, not the veto): exact matching isolates (IRNA, Teheran) in
its own class, and the re-join returns Teheran. The split cost of a re-join is therefore the total size of the fragments whose plurality Y-value is dirty
(a fragment with a clean plurality is repaired, one with a dirty plurality has all its tuples given the dirty value), and f_s = 0 holds only when every fragment's plurality Y-value is
clean. The asymmetry with merges remains: such a split leaves the isolated tuples as dirty as they were in the
input (it forgoes repair), whereas a merge (c) overwrites one entity's clean values with another's.

## D. Closure asymmetry [§IV]

The similarity graph G is over tuples: an edge joins two tuples whose key values are ε-similar (and, for the
Context Veto, whose context values are ε-similar). Similarities are computed once per distinct value; tuples
with equal key values are always adjacent (similarity 1) unless vetoed, so a value shared by several
entities yields inter-entity edges. "Intra-entity" refers to the two end tuples.

**Lemma 8.** Edges of G_in join tuples of one entity, so each G_in-component lies in one entity;
G's components are unions of G_in-components joined by inter-entity edges. ∎
**Theorem 1.** (a) Adding an edge between components merges them entirely (definition of connectivity).
(b) Deleting an edge increases the number of components iff it is a bridge (standard). Apply this to the
entity's subgraph G[e]: a non-bridge removal leaves e connected; a bridge removal separates G[e] into the two
sides of the cut, but in G the sides may remain connected through inter-entity paths, so the mass split off
is at most one side. (c) k-edge-connectivity means no set of fewer than k edges disconnects G[e]. ∎
Measured in `exp_t1/run_eb.py` (log `gpu_out/t1_eb.txt`). Its nodes are distinct (key value, entity) pairs
weighted by tuple counts, i.e. the tuple graph with tuples of equal value and entity contracted (which does not
change components). A value shared by several entities becomes several nodes joined by inter-entity edges of
similarity 1; building G_in therefore uses the ground truth to separate colliding values.

## E. Correctness conditions [§V]

**Theorem 2.** Let κ be collapse-free (X = 0). Then m(e) = 0 for every e, and m_T(e) = 0 in each table. Every error of Lemmas 1–8 is then one-sided (for SUM, if A ≥ 0).
Substituting into Lemma 1 gives an under-count of s(e); into Lemma 2, for A ≥ 0, an under-count by the A-mass
of e's s(e) tuples outside c\*(e); into Lemma 6, precision 1; into Lemma 7, ρ_T = −s_T/|e|_T ≤ 0. Every class meets one entity, so FP = 0 in
Lemma 4 (a false pair needs a class with two entities), the missed pairs obey Corollary 1(b), a selection from
any tuple returns only tuples of its entity, and in Lemma 3 each component of the bipartite graph has q_i = 1,
so COUNT(DISTINCT) over-counts by Σ_e (number of classes of e − 1). The re-join is exact under the plurality
condition of Lemma 5(b). ∎

**Zero merge mass alone.** If only m ≡ 0 is known: Lemma 1, Lemma 2 (A ≥ 0) and Lemma 7 (when m_R, m_S vanish)
are one-sided as above, and a selection from a tuple in c\*(e_q) has precision 1. Each c\*(e) contains only e,
so e ↦ c\*(e) is injective and |classes| − |entities| equals the number of classes that are no entity's
representative: COUNT(DISTINCT) still over-counts (or is exact), but a mixed class counts once although it
holds fragments of several entities. Join precision, selection from an arbitrary tuple, and the re-join need
X = 0; the example of §A has m ≡ 0 and a false join pair.

Let (i) say: the tuples of each entity are connected in the graph whose edges are the pairs with ε-similar
key values and ε-similar values on every context attribute. A sufficient, stronger form is that every pair of
tuples of one entity is ε-similar on the key and on every context attribute. Let (ii′) say: for every pair of
tuples of distinct entities whose key values are ε-similar, some context attribute is not ε-similar.
(ε-similar means similarity ≥ 1 − ε.)

**Proposition 1.** Under (i) and (ii′), the components of the Context Veto graph are exactly the entities.
*Proof.* An edge of the veto graph is a pair with sim^ctx = min(sim_K, min_B sim_B) ≥ 1 − ε. By (ii′), a pair of
distinct entities is never an edge (Corollary 4), so every component lies in one entity. The pairs in (i) are
edges of the veto graph, so each entity is connected and lies in one component. Hence components = entities.
Under the stronger pairwise form, each entity is moreover a clique with no edge to other entities, so
diameter-bounded (complete-linkage) grouping returns the entities as well. ∎

**Corollary 4.** Under (ii′) alone, a pair of tuples of distinct entities is never an edge: either its key
term is below 1 − ε, or (ii′) makes a context term, hence the min, below 1 − ε. Every edge joins one entity;
by induction on path length every component lies in one entity, i.e. κ is collapse-free (X = 0). (i) is not
used. ∎

## F. Embedding gate numbers cited in §I

From `e6_minilm.txt`, `e6_mpnet.txt`, `e6_bge.txt` (sentence-embedding cosine):

| Model | IRAN~IRNA | IRAN~IRAQ |
|---|---|---|
| all-MiniLM-L6-v2 | 0.281 | 0.674 |
| all-mpnet-base-v2 | 0.320 | 0.697 |
| bge-small-en-v1.5 | 0.586 | 0.715 |

Edit (SequenceMatcher) ratio: 0.75 for both pairs.

## G. Data, generators, witnesses

- **Hard-2000** (`hard_bench.py`): 12 countries with capital, continent, subregion, calling code and cities;
  a corrupted cell (probability = noise rate) receives a synonym with probability 1/2 when one exists, else
  one of seven typo operators. One `random.Random(seed)` drives the clean table and the rates in order.
- **Hard-multi** (`exp_t1/hard_multi.py`): countries dimension (12, clean), customers (400) and orders (5 000)
  with independently corrupted country and capital; Zipf(1.1) country popularity with a per-seed random rank
  order; amount log-normal(3, 1); calling code as witness. The calling code is **not corrupted** and is in
  one-to-one correspondence with the country, so where it is observed it equals the ground truth, and
  VetoQ-E(q = 1) coincides with the per-query oracle by construction (a sanity bound, not evidence). A noisy
  variant (`run_ed.py --wit-noise 0.1`) replaces 10% of the recorded codes by another entity's code. With it,
  VetoQ-E(0.3) has mean L1 error 0.084 (Q1–Q4: 0.051, 0.094, 0.153, 0.036; ties that matter in 6 of 60
  decisions, true mean error 0.073–0.087 over tied candidates), against 0.042 with the clean witness and 0.088
  for the best single candidate; VetoQ-E(0.1) 0.090 and VetoQ-E(1.0) 0.061. Witness-GB: 1.781 (q = 0.1),
  1.082 (q = 0.3), 0.068 (q = 1.0) (`gpu_out/t1_ed_witnoise0.1.txt`, `t1_ed_witnoise0.1_witness.csv`). This
  noise is the case in which a witness can hide a merge: a wrong code makes one entity's row look like
  another's.
- **Witness assumptions.** The witness is assumed missing completely at random. A COUNT(DISTINCT) estimated on
  a row sample is biased, because small fragments are under-sampled.
- **Hospital, Beers** (Raha collection): dirty/clean pairs; keys and entities as in Table IV.
- **MusicBrainz 20K/200K, Geographic Settlements, Affiliations** (Leipzig ER benchmarks, CC BY 4.0): loaded by
  `exp_t1/datasets.py`. MusicBrainz and Settlements are sampled by cluster for pairwise similarity (1 000 and
  400 clusters per seed). Settlements' context: two coordinates agree within 25 km; missing coordinates do not
  veto.

## H. Experiments, scripts, logs

| Exp. | Paper | Script | Log |
|---|---|---|---|
| E-C exactness | §VII-B | `exp_t1/run_ec_ea.py` | `gpu_out/t1_ec_ea.txt` |
| E-A metric vs operator | §VII-B | `exp_t1/run_ea.py` | `gpu_out/t1_ea.txt`, `.csv` |
| E-B closure asymmetry | §VII-C | `exp_t1/run_eb.py` | `gpu_out/t1_eb.txt`, `.csv` |
| E-D pipelines, E-E choosers (incl. Witness-GB) | §VII-D, §VII-E | `exp_t1/run_ed.py` | `gpu_out/t1_ed.txt`, `.csv`, `t1_ed_witness.csv` |
| E-E with noisy witness | §VII-E | `exp_t1/run_ed.py --wit-noise 0.1 --tag _witnoise0.1` | `gpu_out/t1_ed_witnoise0.1.txt`, `.csv`, `t1_ed_witnoise0.1_witness.csv` |
| E-E on Hospital | §VII-E | `exp_t1/run_ee_real.py` | `gpu_out/t1_ee_hospital.txt`, `.csv`, `t1_ee_hospital_witness.csv` |
| E-H off-the-shelf ER (Splink) | §VII-G | `exp_t1/run_eh.py`, `exp_t1/analyze_eh.py` | `gpu_out/t1_eh.txt`, `t1_eh.csv`, `t1_eh_analysis.txt` |
| E-E with Splink in the pool (Hard-multi) | §VII-E, §VII-G | `exp_t1/run_ee_ext.py` | `gpu_out/t1_ee_ext.txt`, `t1_ee_ext.csv` |
| E-H closure on a learned similarity | §IV | `exp_t1/run_eh_closure.py` | `gpu_out/t1_eh_closure.txt`, `t1_eh_closure.csv` |
| E-G LLM-based grouping | §VII-F | `exp_t1/run_eg.py --backend hf --batch 32`, `exp_t1/analyze_eg.py` | `gpu_out/t1_eg_hf.txt`, `t1_eg_hf_Qwen2.5-1.5B-Instruct.csv`, `t1_eg_analysis.txt`, answer cache `t1_eg_cache_hf_Qwen2.5-1.5B-Instruct.json` |
| E-F scale | §VII-H | `exp_t1/run_ef.py` | `gpu_out/t1_ef.txt`, `.csv` |

Candidate equivalences (`exp_t1/grouping.py`, `method_grid`): exact; normalized; {edit, embedding, hybrid} ×
{any, all} × ε ∈ {0.1, 0.2, 0.3}; hybrid ε = 0.2 × {any, all} × {hard, soft (θ = 0.5)} veto when a context
exists — 24 candidates (20 without context).

**Chooser tie rule (pre-specified).** VetoQ-E and F1-E pick the candidate with the best estimate; ties go to
the candidate with the smallest estimated merge mass Σ_e m(e)/n on the witnessed rows (merges are the error
closure amplifies), then to grid order (`contingency.choose`). The evaluation reports the range of true errors
over the tied candidates (`contingency.tie_range`). On Hard-multi (`gpu_out/t1_ed.txt`), VetoQ-E(0.3) has
mean error 0.042, and candidates of different true error tie on the estimate in 5 of its 60 (instance, query)
decisions, with true mean error 0.031–0.045 over the tied candidates; for q = 0.1: 0.052, 8 of 60, 0.049–0.087;
for q = 1.0: 0.026, no ties that matter. With the six Splink candidates in the pool (`gpu_out/t1_ee_ext.csv`,
recomputed with `contingency.choose` / `tie_range` on `est0.3_Qk`, `estM_0.3`), VetoQ-E(0.3) has mean error
0.039, ties matter in 5 of 60 decisions, and the tied range is 0.031–0.046. Its single Splink pick
(`Splink-em-key-0.5`, seed 0, noise 0.1, Q2, true error 0) is the unique minimum of the estimate, not a tie; in
the base pool, `edit-any-0.2`, `hybrid-any-0.2` and `hybrid-any-0.2+softveto` tie there on the estimate, and the
rule picks the soft veto (least estimated merge mass; true error 0.167, whereas `hybrid-any-0.2` would be 0).
The two tie rules (representative class, chooser) were implemented on 2026-10-24 and all T1 experiments were
rerun; the logs of the earlier version are archived in `gpu_out/v1.0_2026-10-23/`.

**Witness-GB(q).** Groups by the observed witness itself and propagates the majority witness to unwitnessed
rows with the same normalized key (a direct use of the identifier; implemented in `run_ed.py`). On Hard-multi
(`gpu_out/t1_ed.txt`, `t1_ed_witness.csv`) its mean L1 error is 1.776 (q = 0.1), 1.065 (q = 0.3; Q1–Q4: 0.017,
4.200, 0.024, 0.017) and 0.000 (q = 1.0, where the clean witness is the entity): it is accurate for the
per-entity and join queries, but unwitnessed key variants keep their own classes, which COUNT(DISTINCT) counts
as entities. On Hospital (`gpu_out/t1_ee_hospital.txt`, `t1_ee_hospital_witness.csv`; dirty phone witness) it
has 0.209, 0.224 and 0.248 for q = 0.1, 0.3, 1.0, against 0.198 for VetoQ-E(0.3) and 0.105 for the best single
candidate.

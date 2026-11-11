# ARGUS error audit: which component fails?

**Run set:** the six Judge v4.1 arms (A–F) on the 162 internal validation claims (SciFact train, seed 42). These are the runs described in [challenge-retrieval-v41-summary.md](challenge-retrieval-v41-summary.md). No new runs were made for this audit.

**Method:** each case was inspected by hand from `predictions.jsonl` traces. For each case I recorded:

- the claim and the SciFact answer key
- the papers each arm retrieved, and which agent found them
- the Verifier label and selected sentences
- the Judge's verdict and cited sentences
- my reading of what the abstract actually justifies

Each case gets one **primary** category, with secondary problems noted:

| Category | Meaning |
| --- | --- |
| Retrieval failure | No paper that could settle the claim reached the Verifier |
| Verifier error | A relevant paper arrived, but the Verifier gave it the wrong label |
| Judge error | The Verifier labels were usable, but the Judge reached the wrong verdict or wrote an unsupported rationale |
| Citation failure | The answer was rejected because a citation was malformed or out of range |
| Annotation ambiguity | The SciFact answer key is debatable, or incomplete, for the evidence retrieved |

**Caveat:** the "justified interpretation" column is my reading of the abstracts as a non-specialist, not a domain expert's judgement. SciFact's evidence annotations are not exhaustive (see `AGENTS.md` §5). A paper without a gold tag has not been judged irrelevant; it simply was never annotated.

Arm key: A no Challenge; B dense, claim + suffix (full ARGUS); C dense, claim, excluding Support's papers; D BM25, claim; E LLM counter-claim then dense; F hybrid, claim, excluding Support's papers.

## 1. Named cases

### Claim 494: answer key INSUFFICIENT

> "HSCT-T dosage may not be diagnostic if the onset of symptoms occurs less than 3 hours before AMI."

| Item | Detail |
| --- | --- |
| Gold evidence | None. The source paper 34873974 (Elecsys high-sensitive troponin T meta-analysis) is listed in `cited_doc_ids` but has no rationale sentences. |
| Arm answers | A INSUFFICIENT ✓; B–F CONTRADICT ✗ |
| Retrieval | Only the **Challenge** agent found 34873974 (B rank 4, C 2, D 1, E 4, F 1). Support never found it, and A never saw it. |
| Verifier | CONTRADICT on s14 in every arm that saw the paper |
| Judge | CONTRADICT, citing 34873974 s14 alone, in B–F |
| Cited sentence | s14: "…may not be appropriate for patients who present less than three hours after symptom onset." |
| Justified interpretation | s14 says the same thing as the claim: the test may not work within 3 hours of symptoms. If anything this is **SUPPORT**, not CONTRADICT. The key's INSUFFICIENT is defensible on wording: the claim's "dosage" and "before AMI" do not match the paper's "single baseline measurement" and "after symptom onset". |
| Category | **Verifier error** (direction flip on an agreeing sentence), plus **annotation ambiguity** |
| Effect of Challenge | Harmful. It brought in the only relevant paper, which the Verifier then misread. A was right only because it never saw the paper. |

### Claim 963: answer key CONTRADICT

> "Pre-mRNAs associated with spliceosomal components are less stable than unassociated splicing substrates."

| Item | Detail |
| --- | --- |
| Gold evidence | 4162857 s5 (CONTRADICT) |
| Arm answers | A, B, E INSUFFICIENT ✗; C, D, F SUPPORT ✗. No arm was correct. Standard and Iterative also answered INSUFFICIENT. |
| Retrieval | A, B, E never retrieved 4162857. C, D and F reached it only through **Challenge** (rank 2, 4 and 1). |
| Verifier (C, D, F) | SUPPORT on s2 and s5 |
| Judge (C, D, F) | SUPPORT, citing s2 |
| Cited sentence | s2: "Pol II-synthesized RNAs containing functional splice sites are protected from nuclear degradation, presumably because… the splicing machinery… [associates] over interactions with nucleases." Gold s5: "…guarantees an extended half-life and proper processing of nascent pre-mRNAs." |
| Justified interpretation | The paper says association with the splicing machinery makes pre-mRNAs **more** stable (protected, extended half-life). The claim says **less** stable. That is a clear **CONTRADICT**, matching the key. |
| Category | **Verifier error** (direction flip) in C, D, F; **retrieval failure** in A, B, E |
| Effect of Challenge | Challenge retrieval did what it was designed to do: three arms found the gold contradicting paper that Support missed. The gain was lost at the Verifier. |

### Claim 951: answer key SUPPORT

> "Piezo1 channels are sensors for cell migration in epithelial cells."

| Item | Detail |
| --- | --- |
| Gold evidence | 21414718 s3, s7, s8, s11 (SUPPORT) |
| Arm answers | A, B, C, E INSUFFICIENT ✗; D, F SUPPORT ✓ |
| Retrieval | Support retrieved 21414718 at rank 1 in **every** arm. |
| Verifier | INSUFFICIENT on the gold paper in every arm: it "discusses Piezo1's interaction with TFF1 and… gastric cancer cell mobility, but does not state… sensors for cell migration in epithelial cells." |
| Judge (D, F) | SUPPORT, citing only 4422723 s8, a non-gold paper found by **Challenge** |
| Cited sentence | 4422723 s8: "disruption of a stretch-activated channel, Piezo1, in zebrafish prevents extrusion and leads to the formation of epithelial cell masses." |
| Justified interpretation | The gold s11 states that Piezo1 "is important for TFF1-mediated cell migration". The cells are GES-1, a gastric epithelial line, so SUPPORT is reasonable, though the paper never says "sensor". 4422723 is about **extrusion**, not migration, so it gives only weak, indirect support. |
| Category | **Verifier error** (missed gold evidence; too strict) |
| Effect of Challenge | D and F were right for a weak reason: a loosely related Challenge paper, not the gold paper. |

### Claim 866: answer key INSUFFICIENT

> "Nuclear transfer from adult human fibroblasts to human oocytes can give rise to blastocysts containing expandable pluripotent cells."

| Item | Detail |
| --- | --- |
| Gold evidence | None. The source paper 37822406 ("Human somatic cell nuclear transfer using adult cells") has no rationale sentences. |
| Arm answers | A, B INSUFFICIENT ✓; C, D, E, F SUPPORT ✗. Standard answered SUPPORT; Iterative answered INSUFFICIENT. |
| Retrieval | Support found 4457834, 37822406 and 10546779 in every arm. |
| Verifier | SUPPORT on 4457834 s6 and 10546779 s1. On 37822406 s1 it said CONTRADICT in A, B, D, F but INSUFFICIENT in C and E. |
| Judge | A and B weighed SUPPORT against CONTRADICT and answered INSUFFICIENT. In C and E there was no conflicting label, so the Judge answered SUPPORT. D and F saw the CONTRADICT label, dropped it and answered SUPPORT. |
| Key sentences | 4457834 s6: "…we derived diploid pluripotent stem-cell lines from somatic cells of a newborn and, for the first time, an adult…" 37822406 s1: "…thus far has only been reported with fetal or infant somatic cells." 37822406 s2–s3: generation of human ESCs via SCNT "using dermal fibroblasts from 35- and 75-year-old males… demonstrates the applicability of SCNT for adult human cells." |
| Justified interpretation | Both 4457834 and the source paper itself appear to **support** the claim: adult human somatic or fibroblast nuclei, transferred into oocytes, gave blastocyst-derived pluripotent stem cells. 37822406 s1 is background that the paper then overturns, so labelling it CONTRADICT is a Verifier misreading. A and B were "right" only because of that misreading. |
| Secondary problems | 1. **Judge rationale:** in D and F, the Judge says 10546779 used "adult human fibroblasts", which that abstract does not state. 2. **Retracted paper:** 10546779 appears to be the 2004 cloned-blastocyst paper that was later retracted for fabrication. This should be checked; the corpus does not flag retractions. |
| Category | **Annotation ambiguity** (primary), plus a **Verifier error** on 37822406 s1 and an unsupported **Judge** rationale |
| Effect of Challenge | No new paper mattered. The differences between arms come from the inconsistent Verifier label on 37822406 and the Judge's handling of conflicting labels. |

### Claim 856: answer key CONTRADICT

> "Nonsteroidal antinflammatory drugs are ineffective as cancer treatments."

| Item | Detail |
| --- | --- |
| Gold evidence | 43334921 s6, s8, s10 (CONTRADICT; NSAID use linked to lower colorectal cancer risk) |
| Arm answers | A, B, C, E, F CONTRADICT ✓; D failed with no answer ✗ |
| Retrieval | No arm retrieved the gold paper 43334921. |
| Verifier | CONTRADICT on 6407356 (NSAIDs as chemopreventive agents) and 7975937 (COX ablation and tumour immunity), plus Challenge papers 21150010 (C) and 3866315 (E, F) |
| Judge | Cited 6407356 s0 and 7975937 s2 in every successful arm. **D's Judge cited 7975937 sentence 5, but that abstract has only sentences 0–4**, so the response was rejected as `invalid_response`. |
| Justified interpretation | CONTRADICT is the right verdict, but the evidence used is indirect: chemoprevention reviews and mouse COX-ablation studies, not NSAIDs used as a treatment. 3866315 s5, about acetaminophen and indomethacin lipid mediators, is irrelevant to the claim but was still cited in E and F. |
| Category | **Citation failure** in D; **retrieval failure** of the gold paper in every arm, masked by right answers from indirect evidence |
| Effect of Challenge | None on correctness. Challenge papers added irrelevant citations in E and F. |

## 2. Sample of INSUFFICIENT→CONTRADICT mistakes

Fifteen INSUFFICIENT claims were answered CONTRADICT by at least one arm. **Twelve of the 15 were wrong in every arm, including A**, which has no Challenge agent. This error is therefore mostly not caused by Challenge.

The sample below covers four all-arm cases (1026, 153, 602, 688), one partial case (876) and the only case unique to one Challenge arm (582).

| Claim | Arms wrong | Paper and sentence the verdict rested on | Found by | Verifier | Justified interpretation | Category |
| --- | --- | --- | --- | --- | --- | --- |
| **1026** "Reduced phosphorylation of PP2A increases HDAC4 dephosphorylation by enhancing PP2A-HDAC4 interaction." | A–F (Std, Iter correct) | 3113630 s7: "enhanced PP2A activity leads to HDAC4 dephosphorylation" (source paper) | Support | CON | s6–s7 say ATM phosphorylation *down*-regulates PP2A, so less phosphorylation gives more activity and more HDAC4 dephosphorylation. That is **consistent** with the claim. The "PP2A-HDAC4 interaction" mechanism is not addressed. INSUFFICIENT is justified. | **Verifier error** (false contradiction) |
| **153** "B3-Galectin decreases cell resistance to TKIs by engaging the alternate KRAS-RalB signaling complex downstream of EGFR." | A–F (Std, Iter correct) | 4702639 s1 and s3: integrin **αvβ3** recruits KRAS–RalB and drives erlotinib resistance (source paper) | Support | CON | The paper concerns a different molecule (integrin β3, not galectin). A different subject is "not about this", not a contradiction. INSUFFICIENT is justified. | **Verifier error** (entity mismatch read as contradiction) |
| **602** "Increase of p62 in prostate tumor stroma results in defective autophagy." | A–F (Std, Iter also CON) | 26735905 s1: "p62 levels were **reduced** in the stroma of several tumors…" (not the source paper) | Support | CON | This contradicts the claim's premise that p62 increases. It does not test the effect on autophagy. CONTRADICT is defensible, and the paper was never annotated, since SciFact only judged the source paper 3701541 (liver). | **Annotation ambiguity** (borderline) |
| **688** "Lamins have no effect on nuclear membrane structure maintenance." | A–F (Std, Iter also CON) | 22544171 s1: HGPS (lamin A mutation) fibroblasts "have severe morphological abnormalities in nuclear envelope structure"; 25036988 s0: lamin B1 is "one of the major structural proteins in the lamina mesh" | Support | CON | **CONTRADICT is scientifically justified.** The key is INSUFFICIENT only because the source paper 4452659 (lamin autophagy) does not address the claim directly. Secondary: the Verifier also labelled unrelated Challenge papers CONTRADICT, confusing **laminin** (10574949), **Lam6** (2947540) and lamin-free yeast (14446279) with lamins. The Judge did not cite those. | **Annotation ambiguity**, plus Verifier over-labelling |
| **876** "Obesity raises life quality." | A, B, C, D, F CON; E SUP | 195689316 s6 and s11: higher BMI gives higher mortality and shorter survival (source paper); 43220289 s7: bariatric surgery improves quality of life | Support | CON on 195689316; SUP on 43220289 | Mortality is not quality of life, so the source paper alone gives INSUFFICIENT. 43220289 gives indirect evidence against the claim. In E the Verifier labelled 43220289 SUPPORT while its own rationale says it supports "obesity **impairs** life quality", so the label was given against an inverted claim. | **Verifier error** (over-inference, plus a direction flip in E), plus annotation ambiguity |
| **582** "…daily subcutaneous injections of emtricitabine are **ineffective** at protecting against rectally transmitted SHIV." | A–E SUP; F CON (Std, Iter CON) | 14260013 s8: group 1 (daily subcutaneous FTC) had a "3.8… -fold lower" infection risk, p = 0.02 (source paper) | Support | **SUP** | The sentence shows the drug **was** protective, which contradicts "ineffective". The Verifier labelled whether the paper supports the drug's efficacy rather than whether it supports the claim. In F the Judge reversed this to CONTRADICT, which is scientifically justified but scored wrong. The INSUFFICIENT key looks questionable here. | **Verifier error** (direction flip on a negated claim), plus annotation ambiguity |

## 3. Tally

One primary category per case. In claim 963 the category differs by arm, so it is counted under its main failure, the Verifier.

| Primary category | Cases | Claims |
| --- | --- | --- |
| Verifier error | 7 | 494, 963, 951, 1026, 153, 876, 582 |
| Annotation ambiguity | 3 | 866, 602, 688 |
| Citation failure | 1 | 856 (D only) |
| Retrieval failure | 0 primary | Secondary in 963 (A, B, E) and 856 (gold paper never retrieved) |
| Judge error | 0 primary | Secondary in 866 (unsupported rationale) and 856 (irrelevant citation) |

The kinds of Verifier mistake seen:

| Verifier failure mode | Claims |
| --- | --- |
| Direction flip: an agreeing paper labelled CONTRADICT, or the reverse | 494, 963, 582, 876 (E) |
| Different subject or mechanism treated as contradiction | 153, 1026 |
| Too strict: real support labelled INSUFFICIENT | 951 |
| Name confusion (lamin, laminin, Lam6) and over-labelling | 688 |
| Background sentence read as the paper's finding | 866 (37822406 s1) |

## 4. What this shows

1. **The Verifier is the component that most needs improvement.** It is the primary failure in 7 of 11 cases, and its errors are of recognisable kinds: effect direction, negated claims ("ineffective", "no effect"), and entity mismatch.
2. **The Judge mostly follows the Verifier.** No case here is a Judge-first failure. This matches the full-run count: the Judge follows the Verifier in about 111–113 of about 115 claims where the Verifier labels point one way.
3. **Retrieval is not the bottleneck in these cases.** Challenge retrieval found the deciding paper in 494 and 963 that Support missed. The gain was then lost to Verifier errors.
4. **The INSUFFICIENT→CONTRADICT pattern is not a Challenge problem.** 12 of 15 cases fail in arm A too, and in 688 (and arguably 602 and 582) the "wrong" verdict is scientifically defensible. This makes INSUFFICIENT accuracy partly an artefact of SciFact's non-exhaustive annotation.
5. **Some "correct" answers are correct for weak reasons:** 951 in D and F, 856 in every arm, and 866 in A and B. These would not survive a groundedness check.

## 5. Observations to raise with the team (not acted on)

- **Verifier prompt:** it does not mention effect direction or negation, while the Judge prompt does. Of the hypotheses for the direction flips, this one is cheap to test on the internal split.
- **Inconsistent Verifier labels:** the Verifier labelled the same paper and claim (866, 37822406) differently across arms, even at temperature 0. The input context should be checked for differences between arms.
- **Language drift:** some Judge rationales end in Chinese tokens (856 E; 866 F). This is harmless for scoring but should be noted for groundedness review.
- **No retraction check:** retracted papers in the corpus (possibly 10546779) are not flagged.

Any Verifier change must be developed on the internal split and frozen before the single run on the official dev split (`AGENTS.md` §5).

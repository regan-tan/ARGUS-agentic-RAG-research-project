# Standard RAG offline error analysis

This analysis uses only the completed run `outputs/rag/20261017T115246Z/`, its saved top-five
rankings and responses, and the existing SciFact training annotations. No retrieval or generation
was rerun. The unit of retrieval comparison is the annotated gold **document ID**; sentence-level
observations use the preserved zero-based annotations.

## Error accounting

There were 54 classification errors among 162 attempts. The primary retrieval/error taxonomy is
applied to the 36 errors among the 101 claims with annotated SUPPORT or CONTRADICT evidence:

| Category | SUPPORT errors | CONTRADICT errors | Total | Share of 36 evidence-bearing errors |
| --- | ---: | ---: | ---: | ---: |
| Gold document retrieved, verdict incorrect | 24 | 7 | 31 | 86.1% |
| No annotated gold document retrieved | 2 | 1 | 3 | 8.3% |
| Inference or output-validation failure | 1 | 1 | 2 | 5.6% |

The corresponding class error rates were 27/66 (40.9%) for SUPPORT and 9/35 (25.7%) for
CONTRADICT. Category 3 takes precedence over retrieval status: both failed claims actually had a
gold document at rank 1, but no accepted verdict was produced.

### 1. Gold evidence retrieved but verdict incorrect

The notation `document@rank` records the exact overlap between annotated gold document IDs and the
saved top five. Every evidence-bearing error in this category is included below.

| Gold -> output | Claims and gold/top-five overlap |
| --- | --- |
| SUPPORT -> INSUFFICIENT (19) | 1035 `4547102@2`; 1048 `12486491@5`; 1084 `5691302@1`; 1159 `12584053@1`; 1224 `21932050@2` (other gold `34016987` absent); 1334 `11899391@1`, `13940200@4`; 1356 `13764090@1`; 155 `37549932@2`; 197 `2177022@3`; 305 `14797520@1`; 366 `13956305@1`; 506 `7433668@1`; 668 `25148216@1`; 709 `22442133@1`; 717 `17587795@1`; 79 `5099266@2`; 841 `15663829@1`; 846 `22696649@1`; 951 `21414718@1` |
| SUPPORT -> CONTRADICT (5) | 12 `33409100@1`; 607 `4506414@1`; 681 `9315213@1`; 76 `5531479@1`; 9 `44265107@1` |
| CONTRADICT -> INSUFFICIENT (5) | 1034 `4547102@2`; 1323 `19912367@1`; 245 `3430789@1`; 680 `9315213@1`; 764 `7552215@1` |
| CONTRADICT -> SUPPORT (2) | 571 `20333864@1`; 998 `16472469@1` |

Of these 31 errors, a gold document was first at rank 1 for 24, rank 2 for five, rank 3 for one,
and rank 5 for one. The 24 INSUFFICIENT outputs supplied no citations. All seven wrong-polarity
outputs cited a gold document, and six cited at least one exact annotated sentence. Thus, retrieval
presence alone is not evidence that the model used or understood the gold passage.

Representative cases expose distinct reasoning failures:

- **Claim 12 (SUPPORT -> CONTRADICT):** gold document 33409100 was rank 1. Its annotated sentence 12
  states that high-dose folic acid and B vitamins did not improve survival or vascular outcomes in
  advanced CKD. The response cited that sentence but speculated that lowered homocysteine could
  influence progression and reversed the annotated polarity.
- **Claims 1034/1035 (CONTRADICT/SUPPORT -> INSUFFICIENT):** the paired claims used the same gold
  document 4547102 at rank 2. Annotated sentence 3 reports that KDM4E overexpression improved
  blastocyst formation and SCNT cloning efficiency. The model recognized that the retrieved studies
  described improved efficiency, yet treated the wording “other demethylases” as too broad and
  returned INSUFFICIENT for both opposing claims.
- **Claim 9 (SUPPORT -> CONTRADICT):** gold document 44265107, annotated sentence 15, explicitly says
  that 32% of programs had discontinuation policies. The rationale appears to treat the sentence's
  phrase “contradict the evidence base” as contradicting the claim rather than describing those
  policies, a lexical-polarity error.
- **Claims 571 and 998 (CONTRADICT -> SUPPORT):** both responses cited the rank-1 gold document and
  their rationales explicitly concluded that the claim was contradicted, but the structured verdict
  was SUPPORT. These are rationale/verdict consistency failures that passed structural validation.
- **Claim 1224 (SUPPORT -> INSUFFICIENT):** one of two annotated documents, 21932050, was rank 2.
  Annotated sentence 3 discusses genes transcriptionally active during latent HCMV infection, but
  the response stated that the retrieved text did not address gene-expression changes and cited
  nothing. This separates document retrieval from sentence selection and entailment.

### 2. No annotated gold evidence retrieved

All three retrieval-miss errors returned INSUFFICIENT:

| Claim | Gold label and document IDs | Saved top-five document IDs |
| --- | --- | --- |
| 1109 | SUPPORT; `13770184`, `8582337` | `11748341`, `8529693`, `33740844`, `49429882`, `14803797` |
| 1311 | SUPPORT; `13763195` | `14050257`, `5519177`, `32324430`, `2380002`, `6828370` |
| 963 | CONTRADICT; `4162857` | `13384318`, `29828242`, `21330280`, `16929739`, `4067274` |

For claim 1109, the missing annotated sentences rank dietary risks or childhood undernutrition among
leading global health burdens; the retrieved abstracts discussed nutrition consequences without the
comparative ranking needed by “leading cause.” For claim 1311, the missing gold abstract directly
describes lincRNA-p21 as a translation modulator. For claim 963, the missing abstract links Pol II
transcription and splicing to extended pre-mRNA half-life. These are plausible targets for a second,
more discriminating query.

Across all 101 evidence-bearing claims, not only errors, five had no gold-document hit. Claims 1128
and 856 nevertheless received the annotation-derived correct verdict. This does not prove those
answers were grounded: SciFact gold documents are non-exhaustive, and a correct label can also arise
from non-gold evidence or guessing.

### 3. Inference or output-validation failures

| Claim | Gold label | Gold/top-five overlap | Failure |
| --- | --- | --- | --- |
| 1295 | CONTRADICT | `21239672@1` | Generated CONTRADICT with no citation; strict validation rejected it. |
| 953 | SUPPORT | `3355397@1` | Ollama endpoint timed out; no response or token metadata was returned. |

For claim 1295, annotated document 21239672 sentences 9-10 associate infant or maternal macrolide
use with infantile hypertrophic pyloric stenosis. The raw response instead discussed an unrelated
rank-5 myocardial-infarction document and emitted an empty evidence list. Claim 953's rank-1 gold
document reports an increased pancreatic-cancer hazard among pioglitazone users, but inference never
completed. Both failures therefore occurred after successful retrieval.

## Claims with no annotated evidence

The remaining 18 errors came from the 61 claims with empty SciFact evidence maps: 12 were predicted
CONTRADICT (1026, 1075, 1219, 1235, 1291, 582, 602, 638, 64, 688, 876, 978) and six were predicted
SUPPORT (1111, 1275, 1331, 609, 866, 89). The other 43 received the operational INSUFFICIENT label.

These 18 cases cannot be called retrieval misses because there is no annotated gold document to
retrieve, and their SUPPORT/CONTRADICT outputs should not automatically be described as scientifically
wrong. For example, claim 1275 cited document 4406819 sentences 4-5 about PAAR proteins extending and
sharpening the VgrG spike, which is facially relevant to its T6SS claim despite the empty annotation
map. Conversely, claim 1026 inferred contradiction from increased PP2A activity while its claim was
about PP2A phosphorylation and interaction, a causal distinction not established by the cited
sentence. These examples show why annotation absence must be analyzed separately from evidence
against a claim.

## Implications for later systems

1. **Iteration should be conditional.** Only 3/36 evidence-bearing errors were ordinary top-five
   retrieval misses. A Single-agent Iterative RAG system should reformulate queries when retrieved
   evidence lacks the claim's discriminating relation or qualifier, but additional retrieval alone
   is unlikely to resolve the larger 31-error reasoning group.
2. **Select and verify sentences before judging.** Rank-1 gold documents occurred in 24 of the 31
   retrieved-gold errors. A verifier should classify each claim-sentence pair, preserve direction and
   scope, and reject rationales that cite adjacent but non-annotated or non-entailing sentences.
3. **Check rationale-verdict consistency.** Claims 571 and 998 demonstrate structurally valid JSON in
   which the rationale and final label disagree. An independent Evidence Verifier or Judge should
   explicitly compare the claimed polarity, cited text, rationale, and structured verdict.
4. **Use the Challenge agent for counter-reading, not forced contradiction.** It should test causal
   direction, negation, comparison class, population, and overgeneralization, while allowing “no
   reliable counter-evidence found.” Claims 9, 12, 607, and 681 show that superficial polarity cues
   can invert otherwise direct evidence.
5. **Keep failure and annotation-absence policies explicit.** Bounded retry logic could recover a
   timeout or malformed citation without relabeling it INSUFFICIENT. Empty SciFact annotation maps
   must remain a separate, documented evaluation condition rather than being treated as exhaustive
   negative evidence.

This analysis is diagnostic, not a claim that any future agent design will improve the benchmark.
The official held-out development set remains necessary for the final comparison.

# Ion and channel bibliography

This file is the single source of truth for the `References` entries that
will appear in the NumPy-doc docstrings of the 155 public symbols under
`braincell/ion/` and `braincell/channel/`. Docstring `References` sections
in later tasks are copied **verbatim** from the `### Attribution` block of
the relevant key section below — no citation is typed fresh at docstring
time, and no citation is typed here without having been through the
verification steps (Tasks 2 and 3).

## How this file was built

- **Step 1 (key -> symbol inventory).** Every public class/function in
  `braincell/channel/*.py` and `braincell/ion/*.py` was scanned for a
  citation-key fragment embedded in its name (`[A-Za-z]{2}(?:19|20)\d{2}` or
  `[A-Z]{2}\d{2}`, e.g. `HM1992`, `MA2020`, `PC24`). Symbols whose name
  carries no such fragment are bucketed as `NO_KEY`.
- **Scope exclusion: package `__init__.py` files.** Step 1 scanned the
  *module* files `braincell/channel/*.py` and `braincell/ion/*.py`.
  Symbols defined in `braincell/channel/__init__.py` and
  `braincell/ion/__init__.py` are **outside this project's scope** and
  are deliberately absent from this file. Exactly one public symbol
  falls in that gap today:
  **`braincell/ion/__init__.py::build_placeholder_ions`**, which is in
  that module's `__all__` and is documented with an
  `.. autofunction::` directive in `docs/apis/braincell.ion.rst`. It
  is therefore a 156th public symbol by any whole-package count.
  **This is an exclusion, not an omission.** The figure **155** used
  throughout this file -- including in `## Unresolved attributions`
  item 11 and by any coverage check built from it -- counts module
  files only and does not include it. A later task that widens scope
  to `__init__.py` should add it deliberately and restate the count;
  until then, nobody should read 155 versus 156 as a missing record.
- **Step 2 (provenance harvest).** Every `.mod` file under
  `examples/neuron_compare/Cerebellum_mod/*/{channel,ion}/` was scanned for
  its `TITLE`/`COMMENT`/`Author`/`Ref`/`revis`/4-digit-year lines (first 25
  lines only). This is the NEURON source BrainCell's cerebellar channel
  suite was ported from. The mod-file "year code" in filenames
  (`MA20`, `MA24`, `MA25`, `RI21`, `SU15`, `ZH19`) is a 2-digit form of the
  same key used in the BrainCell class name.
- **Cell-type suffixes** (fixed by
  `examples/neuron_compare/Cerebellum_mod/README.md` and confirmed against
  directory layout): `BC` = basket cell, `DCN` = deep cerebellar nuclei,
  `GoC` = Golgi cell, `GrC` = granule cell, `IO` = inferior olive, `PC` =
  Purkinje cell, `SC` = stellate cell.

## READ BEFORE TRUSTING A `.mod` HEADER

The key embedded in a BrainCell class name is the name of the person(s)
BrainCell's `.mod` source file was imported *from* (i.e. whoever assembled
the multi-compartment cell model this channel was extracted from), **not**
necessarily the origin of the channel's equations or the author of the
paper that should be cited. Every cerebellar `.mod` file that carries an
explicit `Author:`/`CoAuthor:` line in this harvest names an author
*different* from the key's own literature-search target (e.g. Masoli et
al., Solinas/Forti/Rizza, etc.) — see the per-key `### Provenance evidence`
blocks below, and the summary table in
`.superpowers/sdd/task-1-report.md`. Do not resolve a citation from the key
name alone; read the harvested header text.

## Status of this file

- `### Verified record` and `### Attribution` blocks are filled in
  progressively. Task 2 (2026-08-15) completed the nine
  classical/thalamic keys -- `HH1952`, `TM1991`, `Ba2002`, `HM1992`,
  `Ya1989`, `Re1993`, `HP1992`, `De1994` and `IS2008`. Eight of them
  carry a published `.. [N]` entry; `IS2008` is recorded as NOT FILLED
  because its attribution did not close (see `## Unresolved
  attributions` item 6), and one symbol of the otherwise-verified
  `HM1992` key is carved out the same way (item 7).
- Task 3 (2026-08-15) completed the seven cerebellar-model keys --
  `MA2020`, `MA2024`, `MA2025`, `RI2021`, `SU2015`, `ZH2019` and
  `PC24`, covering **104** public symbols (not 103; see the note in
  `## Unresolved attributions` item 11). All seven model papers
  verified. Because the key names a *porter*, not an author, Task 3
  also verified 38 **origin-of-kinetics** papers, collected in
  `## Origin-of-kinetics records` below. **99 of the 104 symbols need
  a two-or-more-level citation** -- most of them more than two
  entries, not exactly two; see `## How to assemble a
  two-or-more-level citation`. The other 5 (the `SU2015` `Toy*`
  family) get no citation at all because they are BrainCell's own
  test fixtures.
- Task 3 follow-up (2026-08-15) closed the `NO_KEY` bucket, which had
  been assigned to no task. Three of its 32 symbols carry a real
  primary source and are now verified -- `ghk_flux` (Goldman 1943 /
  Hodgkin & Katz 1949), `KineticIon` (Hines & Carnevale 2000 / The
  NEURON book 2006) and `CalciumDetailed` (Destexhe 1994 / Bazhenov
  1998, plus Destexhe 1993 for its expository prose only; the
  deferred attribution checks are closed at the same time). The
  other **29 are recorded, symbol by symbol, as having no
  primary literature source** and are expected to ship without a
  `References` section; that list is in the bucket's
  `### Symbols with no primary literature source` block and is the
  source of truth for any later allowlist.
- **The symbol count, reconciled.** `__all__` across
  `braincell/channel/*.py` and `braincell/ion/*.py` holds **155**
  public symbols: 154 classes and one function (`ghk_flux`). The
  package `__init__.py` files are out of scope and their symbols are
  excluded -- see the scope exclusion under `## How this file was
  built`, which names the one symbol affected. They
  split as **32 `NO_KEY` + 123 keyed**, and the keyed half splits as
  **19 across Task 2's nine keys + 104 across Task 3's seven**. Every
  per-key heading in this file already carries the right number and
  they sum to 155 with no symbol listed twice and none missing -- this
  was checked mechanically, by parsing `__all__` and diffing it
  against this file's own bullet lists. The plan's figures of 103 and
  122 are each one short of 104 and 123; see `## Unresolved
  attributions` item 11.
- Nothing in this file's `### Provenance evidence` blocks is a citation.
  It is raw, unedited text copied from `.mod` file headers (typos,
  inconsistent spacing, and missing apostrophes preserved verbatim) plus
  structural notes about what was and was not found. Do not treat any of
  it as verified. **Task 1's harvest read only the first 25 lines of
  each `.mod` file and therefore missed real provenance text in five
  places** -- see `## Unresolved attributions` item 8. The raw blocks
  are left exactly as harvested; the corrections live in the
  `### Verified record` blocks.
- Task 3 added one structural block per key, `### Import deviations`,
  carrying the NEURON-to-BrainCell port deviations transcribed from
  `examples/neuron_compare/Cerebellum_mod/README.md`. This is not a
  second citation format -- it holds no citations. It exists so a
  module task can write a docstring `Notes` section without
  re-reading that file.

## Citation house style

Established by Task 2; **Task 3 and every module task must follow it.**
Entries are written in the reST form that goes straight into a NumPy-doc
`References` section, so they can be copied verbatim with no reformatting.

- Numbered `.. [N]` entries. Continuation lines are indented 7 spaces so
  they align under the text, not under the bracket. Keep every line
  under 79 characters.
- Authors: `Surname, A. B., & Surname, C. D.` -- initials with periods
  and spaces, `&` before the last author, all authors listed in full
  (no `et al.`), then `(YEAR).`
- Title: sentence case, ending in a period. Most of the journals cited
  here (The Journal of Neuroscience, Journal of Neurophysiology) print
  their titles in title case, so down-casing is the normal operation,
  not an exception -- five of the eight Task 2 entries required it;
  ``HH1952``, ``Re1993`` and ``De1994`` arrived from Crossref already
  in sentence case and needed none. Preserve proper nouns, species
  names, ion and chemical symbols (Ca2+, K+, GABAergic) and
  capitalised acronyms as published. What must never change is the
  *wording*: do not silently correct a publisher's singular/plural,
  spelling or hyphenation oddity -- reproduce it verbatim and record
  the discrepancy in the surrounding prose.
- Journal articles: `Journal Name, VOL(ISSUE), FIRST-LAST.` with the
  journal spelled out in title case and no italics markup. Page ranges
  use an ASCII hyphen, never an en dash.
- DOI last, on its own line, as `doi:10.xxxx/yyyy` -- bare prefix, no
  `https://doi.org/` URL. Omit the line only when no DOI exists.
- Books: `Author, A. B., & Author, C. D. (Year). Title of book.
  Publisher.` followed by the DOI line if one exists. No place of
  publication (APA 7 style). Record the ISBN in the surrounding prose,
  not in the `.. [N]` entry.
- Book chapters: `Author, A. B. (Year). Chapter title. In E. Editor &
  F. Editor (Eds.), Book title: Subtitle (pp. FIRST-LAST). Publisher.`
- Each `### Verified record` block opens with one or two sentences
  naming *what was checked and where* (PubMed PMID, DOI resolution,
  publisher page, PMC ID, catalogue record), dated, so the entry can be
  re-audited later. Then the `.. [N]` entry. Then any correction or
  ambiguity notes.
- Each `### Attribution` block names the symbols, cites the code
  location(s) read, names the external artefact the constants were
  compared against (paper section, abstract, or a specific reference
  `.mod` file with its ModelDB accession), lists the matching
  equations, and then lists parameter-default divergences separately as
  **caveats** -- a default that differs from the paper is not a citation
  error, but it must not be presented as the paper's value.
- A key that fails either check gets no `.. [N]` entry at all. Its
  `### Verified record` block says NOT FILLED and points at the
  numbered item in `## Unresolved attributions` that carries the
  evidence.

## How to assemble a two-or-more-level citation

Established by Task 3 for the cerebellar half of this file. **This is
not a second citation format** -- the `.. [N]` entries below are
written in exactly the house style above. What is new is only *which*
entries a docstring needs, and *how many*.

A cerebellar key names the group that assembled the multi-compartment
cell model BrainCell imported the mechanism from. It almost never names
whoever wrote the mechanism's equations. So a cerebellar docstring's
`References` section takes **two or more** entries, in this order:

- `.. [1]` (and `.. [2]`, `.. [3]` where the mapping row lists more
  than one) -- the **origin of the kinetics**: the paper or papers the
  mechanism's equations actually come from. Copy each verbatim from
  `## Origin-of-kinetics records` below, using the label(s) given in
  the key's `### Attribution` mapping table, **in the order the row
  lists them**.
- the **last** entry -- the **model BrainCell imported from**: the
  key's own paper. Copy it verbatim from that key's
  `### Verified record`.

**"Two" is the minimum, not the norm.** Of the 58 mapping rows in this
file, only **12 carry a single origin** (giving a two-entry
docstring). **32 carry two origins** (three entries) and **14 carry
three** (four entries). So 46 of 58 rows -- the clear majority --
need more than the two-entry shape, and a reader who skims this
section and writes two entries everywhere will truncate most of them.

Two failure modes, both bugs:

- **One entry where two are required.** Across all 104 cerebellar
  symbols, origin and model never coincide, so treat a single-entry
  cerebellar `References` block as a bug unless the mapping table says
  otherwise.
- **Two entries where a three-origin row requires four.** Eleven rows
  name three origins as `O-` keys: the `Kca3p1` family (`O-RC2006`,
  `O-BB1993`, `O-DV2000`, 3 rows), the `Nav1p6` family (`O-RB2001`
  kinetics, `O-KH2003`, `O-AK2006`, 4 rows) and the calcium
  buffer/pump family -- `CdpStC*`, `CdpCR*`, `CdpCAM*` -- (`O-AN2012`
  model, `O-SC2003` buffers, `O-MD1999` pump tuning, 4 rows). Three
  further rows reach three sources without three `O-` keys: the
  `Cav3p2_*` rows spell their first source as prose (Huguenard &
  McCormick 1992, see the `HM1992` Verified record) alongside
  `O-VI2005` and `O-CO1989`, so they need four entries as well.
  Fourteen rows in total, not eleven -- do not audit this by grepping
  for three `O-` keys. Dropping one of the three because "two-level"
  was read as a literal cap is the same class of bug as dropping the
  model paper, and it is easier to make. **Count the sources in the
  row, however they are spelled; the entry count is sources + 1.**

  The `Kv3p4_*` rows carry a single origin (`O-KH2003`) and
  `Kv3p3_MA2024_PC` carries two (`O-MT2007` fits, `O-AK2009` model);
  an earlier revision of this paragraph mislabelled the `Nav1p6`
  triple as a "`Kv3p4`/`Kv3p3` family" and the `Cdp*` triple as a
  "`Cav3p1`/`Cav3p2`/`Cav3p3` family". The per-key mapping tables
  were correct throughout and remain authoritative.

Renumber only the bracket digits when a docstring needs a different
order; never retype the entry text.

**One deliberate exception to the 79-column rule.** The per-key
mapping tables and `### Import deviations` tables are Markdown table
rows, which cannot be wrapped without breaking the table, and some
exceed 79 columns. The rule continues to bind without exception for
every `.. [N]` entry and its 7-space continuation lines, which is
where it matters -- those are the strings that get copied into
docstrings. No table row is ever copied into a docstring.

---

## Origin-of-kinetics records

Thirty-eight papers. **Thirty-six of them are the source of the
equations in one or more cerebellar mechanisms**; the remaining two --
`O-FO2006` and `O-SO2007b` -- are verified records that no mapping
table assigns to any symbol, and each is marked
**VERIFIED BUT UNASSIGNED** at its own heading below. Do not read
either as a gap in the mapping tables: they were verified because the
harvest surfaced them, and they are recorded so that a later task
neither re-verifies them nor assumes an assignment that was never
established. Each carries a stable label (`O-XXNNNN`) used by the
per-key `### Attribution` mapping tables. **Labels are internal to
this file and must never appear in a docstring** -- copy the `.. [N]`
entry text, not the label.

All thirty-eight were confirmed 2026-08-15 against at least two
independent sources: NCBI E-utilities (`efetch`, `db=pubmed`), the
Crossref REST API (`api.crossref.org/works/<doi>`), publisher-deposited
JATS XML via `efetch db=pmc`, live DOI resolution, and -- where the
mechanism file itself had to be inspected -- the ModelDB REST API and
the `github.com/ModelDBRepository` mirrors. Per-record deviations from
that baseline are noted inline.

### O-DA2001 -- cerebellar granule cell model (Pavia)

Source of the `CaHVA`, `Kv4p3`, `KM` and `Kir2p3` mechanisms.

.. [1] D'Angelo, E., Nieus, T., Maffei, A., Armano, S., Rossi, P.,
       Taglietti, V., Fontana, A., & Naldi, G. (2001). Theta-frequency
       bursting and resonance in cerebellar granule cells: experimental
       evidence and modeling of a slow K+-dependent mechanism. The
       Journal of Neuroscience, 21(3), 759-770.
       doi:10.1523/JNEUROSCI.21-03-00759.2001

PMID 11157062, PMCID PMC6762330. **Two corrections to the `.mod`
headers.** The credit "E.D'Angelo, T.Nieus, A. Fontana" names authors
1, 2 and 7 of an eight-author paper; Maffei, Armano, Rossi, Taglietti
and Naldi are omitted. The `Kir2p3` header's reference string
"Theta-Frequency Bursting and Resonance in Cerebellar Granule
Cells:Experimental" is the published title truncated mid-subtitle.
PubMed renders "k+" lowercase; Crossref and the publisher's own PMC
deposit both give capital "K+", which is used above.

### O-SO2007a -- cerebellar Golgi cell model, part I

.. [1] Solinas, S., Forti, L., Cesana, E., Mapelli, J., De Schutter,
       E., & D'Angelo, E. (2007). Computational reconstruction of
       pacemaking and intrinsic electroresponsiveness in cerebellar
       Golgi cells. Frontiers in Cellular Neuroscience, 1, 2.
       doi:10.3389/neuro.03.002.2007

PMID 18946520, PMCID PMC2525930. Article number 2; no issue number and
**no page range** -- the article number goes in the page slot.
**Discrepancy with the `.mod` header, resolved.** `Kca2p2_*.mod` cites
this paper as "(2008) ... Frontiers in Cellular Neuroscience 2:2".
Both fields are wrong: the year of record is 2007 and the volume is 1,
confirmed by PubMed, the PMC JATS deposit and the live publisher page.
The publisher's site also advertises a legacy slash-form DOI
(`10.3389/neuro.03/002.2007`) and a spurious `citation_firstpage` of
91; neither is canonical. Use the dot-form DOI and the article number.

### O-SO2007b -- cerebellar Golgi cell model, part II

**VERIFIED BUT UNASSIGNED.** No mapping table row cites this label and
no symbol takes it. The record below is correct and re-usable, but
nothing in this file establishes which mechanism, if any, draws its
equations from part II rather than part I (`O-SO2007a`, which *is*
assigned, to the `HCN1`/`HCN2` and `Kca2p2` rows). A module task must
not reach for this entry on the strength of the label's existence.

.. [1] Solinas, S., Forti, L., Cesana, E., Mapelli, J., De Schutter,
       E., & D'Angelo, E. (2007). Fast-reset of pacemaking and
       theta-frequency resonance patterns in cerebellar Golgi cells:
       Simulations of their impact in vivo. Frontiers in Cellular
       Neuroscience, 1, 4.
       doi:10.3389/neuro.03.004.2007

PMID 18946522, PMCID PMC2525929. Article number 4; same no-page-range
and legacy-DOI caveats as O-SO2007a (its spurious `citation_firstpage`
is 166). The post-colon "Simulations" is capitalised in the publisher
deposit and is reproduced as published.

### O-FO2006 -- Golgi cell autorhythmicity, experimental

.. [1] Forti, L., Cesana, E., Mapelli, J., & D'Angelo, E. (2006).
       Ionic mechanisms of autorhythmic firing in rat cerebellar Golgi
       cells. The Journal of Physiology, 574(3), 711-729.
       doi:10.1113/jphysiol.2006.110858

PMID 16690702, PMCID PMC1817727. **Both machine-readable deposits are
corrupt for this record**: Crossref and the PMC JATS both merge the
first two authors into one contributor (`given: "Lia Forti", family:
"Elisabetta Cesana"`), yielding three authors. PubMed is authoritative
and gives the four above, in that order. PubMed prints the issue as
"Pt 3"; Crossref as "3", which is used.

**VERIFIED BUT UNASSIGNED.** No mapping table row cites this label and
no symbol takes it. The lead that produced it is real but stops short
of an assignment: `GoC/channel/HCN1_MA20_GoC.mod` and
`HCN2_MA20_GoC.mod` carry `Author:L. Forti & S. Solinas` and
`Last revised: April 2006`, and Lia Forti is the first author here.
That places the mod files' author and this paper in the same year and
the same subject, which is why the record was verified. It does
**not** show that the HCN rate expressions were taken from this paper:
those files' own header credits their data to `Santoro et al. J
Neurosci. 2000` (`O-SA2000`), and the `HCN1`/`HCN2` mapping row
already assigns `O-SO2007a` for the kinetics and `O-SA2000` for the
data on the strength of direct constant comparison. This is an
experimental paper, not a modelling one, so a kinetics assignment
would need equation-level evidence that was not found. **Do not add it
to the `HCN1`/`HCN2` row without that evidence.**

### O-SA2000 -- HCN subunit data

Cited by `HCN1_MA20_GoC.mod` / `HCN2_MA20_GoC.mod` as "Data from".

.. [1] Santoro, B., Chen, S., Luthi, A., Pavlidis, P., Shumyatsky,
       G. P., Tibbs, G. R., & Siegelbaum, S. A. (2000). Molecular and
       functional heterogeneity of hyperpolarization-activated
       pacemaker channels in the mouse CNS. The Journal of
       Neuroscience, 20(14), 5264-5275.
       doi:10.1523/JNEUROSCI.20-14-05264.2000

PMID 10884310, PMCID PMC6772310. The third author is published as
**Lüthi**; the ASCII form is used above because the rest of this file
is ASCII. If the docstring pipeline is UTF-8-safe, write "Lüthi, A.".

### O-HI1998 -- SK2 gating data

.. [1] Hirschberg, B., Maylie, J., Adelman, J. P., & Marrion, N. V.
       (1998). Gating of recombinant small-conductance Ca-activated K+
       channels by calcium. The Journal of General Physiology, 111(4),
       565-581.
       doi:10.1085/jgp.111.4.565

PMID 9524139, PMCID PMC2217120. **Verbatim-wording flag:** the title
reads "**Ca**-activated", not "Ca2+-activated", even though the "K+"
in the same title is superscripted. That is a publisher inconsistency
in the original, identical in Crossref and the JGP PMC deposit, and it
is reproduced rather than corrected. Do not confuse this with the same
authors' 1999 Biophysical Journal 77(4), 1905-1913 SK paper.

### O-MG2006 -- granule cell sodium currents

.. [1] Magistretti, J., Castelli, L., Forti, L., & D'Angelo, E.
       (2006). Kinetic and functional analysis of transient, persistent
       and resurgent sodium currents in rat cerebellar granule cells in
       situ: an electrophysiological and modelling study. The Journal
       of Physiology, 573(1), 83-106.
       doi:10.1113/jphysiol.2006.106682

PMID 16527854, PMCID PMC1779707. Keep the British "modelling" and the
Latin "in situ" verbatim.

### O-RB2001 -- the 13-state Purkinje Na scheme

.. [1] Raman, I. M., & Bean, B. P. (2001). Inactivation and recovery
       of sodium currents in cerebellar Purkinje neurons: evidence for
       two mechanisms. Biophysical Journal, 80(2), 729-737.
       doi:10.1016/S0006-3495(01)76052-3

PMID 11159440, PMCID PMC1301271. The `.mod` headers' "Based om" is a
typo for "Based on", and their bare "80 (2001) 729" omits the closing
page 737 and the issue. Full given names come from Crossref only;
PubMed and PMC carry initials for this record.

### O-AN2012 -- OIST Purkinje Ca-buffering model

The "Current Model Reference" of the `Cav3p1`, `Cav2p1`, `Kca1p1` and
`Cdp*` mechanisms. ModelDB accession 138382.

.. [1] Anwar, H., Hong, S., & De Schutter, E. (2012). Controlling
       Ca2+-activated K+ channels with models of Ca2+ buffering in
       Purkinje cells. The Cerebellum, 11(3), 681-693.
       doi:10.1007/s12311-010-0224-3

PMID 20981513, PMCID PMC3411306. **Two corrections to the `.mod`
headers.** (1) The published title ends "Purkinje **cells**" (plural);
every header in this repository has the singular. (2) The headers date
it 2010; that is the online-first date, which is also why the DOI
carries `-010-`. The citable record is **2012**, 11(3), 681-693, per
both PubMed and Crossref, and ModelDB itself labels the entry
"(Anwar et al. 2012)". The journal's title of record is "The
Cerebellum" (Crossref journal record for ISSN 1473-4222); NLM indexes
it as "Cerebellum (London, England)".

### O-CX1997 -- BK/mslo allosteric gating parameters

.. [1] Cox, D. H., Cui, J., & Aldrich, R. W. (1997). Allosteric gating
       of a large conductance Ca-activated K+ channel. The Journal of
       General Physiology, 110(3), 257-281.
       doi:10.1085/jgp.110.3.257

PMID 9276753, PMCID PMC2229366. **The `.mod` headers' "Cox et al.
(1987)" is a typo for 1997** -- stated explicitly because the wrong
year is copy-pasted into five files in this repository. J Gen Physiol
volume 110 is September 1997, and pages 257-281 of that volume are
exactly this paper. The headers' "(patch 1)" points at a specific
recorded patch inside the paper, not a bibliographic field. Title
reads "Ca-activated", as published.

### O-SW2005 -- P-type (Cav2.1) recordings

.. [1] Swensen, A. M., & Bean, B. P. (2005). Robustness of burst
       firing in dissociated Purkinje neurons with acute or long-term
       reductions in sodium conductance. The Journal of Neuroscience,
       25(14), 3509-3520.
       doi:10.1523/JNEUROSCI.3929-04.2005

PMID 15814781, PMCID PMC6725377. The `.mod` headers write "purkinje"
lowercase, inherited from PubMed's own lowercasing; the publisher's
`<article-title>` in PMC6725377 capitalises it, and "Purkinje" is a
proper noun that survives sentence-casing. Headers also omit issue 14
and truncate the page range to 3509-20.

### O-IF2006 -- Cav3.1 temperature dependence

.. [1] Iftinca, M., McKay, B. E., Snutch, T. P., McRory, J. E.,
       Turner, R. W., & Zamponi, G. W. (2006). Temperature dependence
       of T-type calcium channel gating. Neuroscience, 142(4),
       1031-1042.
       doi:10.1016/j.neuroscience.2006.07.010

PMID 16935432; **no PMCID** (NCBI ID Converter returns an explicit
"not found in PMC", i.e. confirmed absent, not merely unfound). The
`.mod` headers capitalise "Calcium"; the published title is lowercase.
*Neuroscience* publishes initials only, so no source of record carries
full given names for these six authors.

### O-SC2003 -- Purkinje dendritic Ca buffering parameters

.. [1] Schmidt, H., Stiefel, K. M., Racay, P., Schwaller, B., &
       Eilers, J. (2003). Mutational analysis of dendritic Ca2+
       kinetics in rodent Purkinje cells: role of parvalbumin and
       calbindin D28k. The Journal of Physiology, 551(1), 13-32.
       doi:10.1113/jphysiol.2002.035824

PMID 12813159, PMCID PMC2343131. PubMed records the issue as "Pt 1",
Crossref as "1", which is used. Crossref's `given: "K. M"` is a
deposit artefact; PubMed's `ForeName=Klaus M` is authoritative.

### O-MD1999 -- Ca decay data used to tune the pump rate

.. [1] Maeda, H., Ellis-Davies, G. C. R., Ito, K., Miyashita, Y., &
       Kasai, H. (1999). Supralinear Ca2+ signaling by cooperative and
       mobile Ca2+ buffering in Purkinje neurons. Neuron, 24(4),
       989-1002.
       doi:10.1016/S0896-6273(00)81045-4

PMID 10624961; **no PMCID** (confirmed absent). Identified by
derivation, not guesswork: the `Cdp*` headers cite O-AN2012 as their
model reference, and the reference list of O-AN2012 as returned by
`efetch` contains exactly one Maeda-1999 entry, which is this one; it
was then re-verified independently against PubMed and Crossref.
**Caveat for the module task:** this is *not* a Ca2+-ATPase/PMCA
paper. It is a caged-Ca2+ uncaging study of endogenous buffering. The
headers' "pump rate was tuned according to data from Maeda et al.
1999" means the pump was tuned so the model reproduces Maeda's
measured Ca2+ decay -- do not write that Maeda characterised a pump.
Crossref gives "Graham C. R. Ellis-Davies"; PubMed truncates to "G C".

### O-XC2008 -- Cav3.3

.. [1] Xu, J., & Clancy, C. E. (2008). Ionic mechanisms of endogenous
       bursting in CA3 hippocampal pyramidal neurons: A model study.
       PLoS ONE, 3(4), e2056.
       doi:10.1371/journal.pone.0002056

PMID 18446231, PMCID PMC2323611. `e2056` is the article number and
occupies the page slot. Issue 4 is the only field the `.mod` headers
omit; their "3:e2056" is otherwise correct.

### O-VI2005 -- Cav3.2 recordings

.. [1] Vitko, I., Chen, Y., Arias, J. M., Shen, Y., Wu, X.-R., &
       Perez-Reyes, E. (2005). Functional characterization and
       neuronal modeling of the effects of childhood absence epilepsy
       variants of CACNA1H, a T-type calcium channel. The Journal of
       Neuroscience, 25(19), 4844-4855.
       doi:10.1523/JNEUROSCI.0847-05.2005

PMID 15888660, PMCID PMC6724770. The `.mod` header's "25(19)
:4844-4855, 2005" is correct in every field. Independently confirmed
by two separate verification passes in this task.

### O-CO1989 -- Q10 source for the low-threshold Ca current

.. [1] Coulter, D. A., Huguenard, J. R., & Prince, D. A. (1989).
       Calcium currents in rat thalamocortical relay neurones: kinetic
       properties of the transient, low-threshold current. The Journal
       of Physiology, 414(1), 587-604.
       doi:10.1113/jphysiol.1989.sp017705

PMID 2607443, PMCID PMC1189159. British "neurones" as published. The
issue number is uncertain in provenance -- PubMed records none,
Crossref supplies 1, and the classic in-print form is "414: 587-604"
with no issue; drop the `(1)` if matching the majority convention.
**Do not write that the paper prints "Q10 = 5 for m and 3 for h".** Its
abstract states only that all kinetic properties were temperature
sensitive "with Q10 ... values of greater than 2.5"; the specific 5/3
split is Destexhe's parameterisation derived from that data.

### O-AK2009 -- Kv1.1, Kv3.3 and Nav1.1 reference

.. [1] Akemann, W., Lundby, A., Mutoh, H., & Knopfel, T. (2009).
       Effect of voltage sensitive fluorescent proteins on neuronal
       excitability. Biophysical Journal, 96(10), 3959-3976.
       doi:10.1016/j.bpj.2009.02.046

PMID 19450468, PMCID PMC2712148. Published as **Knöpfel**; ASCII form
used here for file consistency. The title is printed "Voltage
Sensitive" with no hyphen -- non-standard, confirmed identically in
Crossref, PMC JATS and MEDLINE, and reproduced rather than corrected.
The `.mod` headers' "96: 3959-3976" is correct; only issue 10 is
missing.

### O-AK2006 -- resurgent Na (Nav1.6) reference

.. [1] Akemann, W., & Knopfel, T. (2006). Interaction of Kv3
       potassium channels and resurgent sodium current influences the
       rate of spontaneous firing of Purkinje neurons. The Journal of
       Neuroscience, 26(17), 4602-4612.
       doi:10.1523/JNEUROSCI.5204-05.2006

PMID 16641240, PMCID PMC6674064. The `.mod` headers' "Knoepfel" is an
ASCII transliteration of **Knöpfel**, and their "26 (2006) 4602" gives
only the first page; the range is 4602-4612, issue 17.

### O-KH2003 -- Purkinje resurgent Na and TEA-sensitive K

Origin of both the `Nav1p6` family (via O-AK2006) and, directly, the
`Kv3p4` family. ModelDB accession 48332.

.. [1] Khaliq, Z. M., Gouwens, N. W., & Raman, I. M. (2003). The
       contribution of resurgent sodium current to high-frequency
       firing in Purkinje neurons: an experimental and modeling study.
       The Journal of Neuroscience, 23(12), 4899-4912.
       doi:10.1523/JNEUROSCI.23-12-04899.2003

PMID 12832512, PMCID PMC6741194. The `.mod` headers' "23(2003)4899"
compresses volume, year and first page; the full locator is 23(12),
4899-4912.

**Verified against the deposited source, not just the citation.** The
`Kv3p4_*` files in this repository are ModelDB 48332's `kpkj.mod`
renamed. The deposited `kpkj.mod` opens with exactly the two lines
this repository's files carry -- ": HH TEA-sensitive Purkinje
potassium current" / ": Created 8/5/02 - nwg" -- and its parameters
`mivh = -24 mV`, `mik = 15.4`, `hiy0 = 0.31` reproduce the paper's
Table 1 row for **K fast** (m^3 h, V-half -24 mV, k 15.4 mV,
y0 = 0.31). "nwg" is Nathan W. Gouwens, the paper's second author. The
": Suffix from kpkj to Kv3_4" line is a BrainCell-local addition and is
not in the deposit. Sibling mechanisms in the same
accession are `kpkj2.mod` (low TEA-sensitive) and `kpkjslow.mod`
(TEA-insensitive); neither is imported here.

**Caveat that must reach the `Kv3p4` docstrings.** The paper calls
this current **K fast**, never a Kv subunit. Its only mention of Kv3 is
one Discussion sentence noting that the positive activation range is
"typical of the K_V3 family". The strings "Kv3.4", "Kv3.3" and "Kv3.1"
appear nowhere in it. **The ".4" in BrainCell's `Kv3p4` name is an
interpolation with no support in the cited paper.** A docstring may say
the kinetics are those of the TEA-sensitive fast K current of Khaliq
et al. (2003), which those authors associate with the Kv3 family; it
must not say the paper identifies a Kv3.4 subunit.

### O-ZE1998 -- human Kv1.1

.. [1] Zerr, P., Adelman, J. P., & Maylie, J. (1998). Episodic ataxia
       mutations in Kv1.1 alter potassium channel function by dominant
       negative effects or haploinsufficiency. The Journal of
       Neuroscience, 18(8), 2842-2848.
       doi:10.1523/JNEUROSCI.18-08-02842.1998

PMID 9526001, PMCID PMC6792579. The `.mod` headers' "18, 2842, 2848,
1998" is not garbled but mis-punctuated: volume 18, pages 2842**-**2848,
issue 8. **Disambiguation trap:** the same three authors published a
second 1998 paper on the same subject, FEBS Letters 431(3), 461-464
(PMID 9714564, doi:10.1016/s0014-5793(98)00814-x). The header's
volume/page pin the J Neurosci paper, but its descriptive gloss
("Human Kv1.1 expressed in xenopus oocytes") fits both.

### O-MT2007 -- Kv3.3 fitting data

.. [1] Martina, M., Metz, A. E., & Bean, B. P. (2007).
       Voltage-dependent potassium currents during fast spikes of rat
       cerebellar Purkinje neurons: inhibition by BDS-I toxin. Journal
       of Neurophysiology, 97(1), 563-571.
       doi:10.1152/jn.00269.2006

PMID 17065256; **no PMCID** (not deposited in PMC). **The `.mod`
header is wrong twice**: its page range "563-671" should be
**563-571**, and its unbalanced "(" before 563 is stray. The header
also omits the published subtitle ": inhibition by BDS-I toxin".

### O-AG2007 -- HCN1 (I_h) distribution

.. [1] Angelo, K., London, M., Christensen, S. R., & Hausser, M.
       (2007). Local and global effects of Ih distribution in
       dendrites of mammalian neurons. The Journal of Neuroscience,
       27(32), 8643-8653.
       doi:10.1523/JNEUROSCI.5284-06.2007

PMID 17687042, PMCID PMC6672943. Published as **Häusser**; ASCII form
used here. The published title sets I_h as an italic capital I with
subscript h -- PubMed's "I(h)" is an ASCII-flattening artefact and the
parentheses must not be copied. Christensen is published "Soren R."
with a plain o, despite the author's canonical "Søren". The `.mod`
`TITLE` line carries the author names but no volume, issue or pages.

### O-PO2003a -- CA1 pyramidal model, part I

Origin of `Cav2p3` (the `car.mod` mechanism). ModelDB accession 20212.

.. [1] Poirazi, P., Brannon, T., & Mel, B. W. (2003). Arithmetic of
       subthreshold synaptic summation in a model CA1 pyramidal cell.
       Neuron, 37(6), 977-987.
       doi:10.1016/S0896-6273(03)00148-X

PMID 12670426; no PMCID (Cell Press does not deposit in PMC).

### O-PO2003b -- CA1 pyramidal model, part II

.. [1] Poirazi, P., Brannon, T., & Mel, B. W. (2003). Pyramidal neuron
       as two-layer neural network. Neuron, 37(6), 989-999.
       doi:10.1016/S0896-6273(03)00149-1

PMID 12670427; no PMCID. **Verbatim-wording flag:** the title has no
article before either noun phrase ("as two-layer neural network"),
which is the publisher's wording and is reproduced unaltered.

**Which of the pair to cite.** ModelDB 20212 is one shared model
serving both companion papers plus their joint online supplement, and
its `readme.txt` and `model_paper` field list both. `car.mod` belongs
to the shared biophysics and cannot be assigned to one. Citing both is
the accurate choice; if only one is wanted, cite O-PO2003a, which is
listed first in both places and is the paper describing the model's
construction. The deposited `car.mod` header matches this repository's
file exactly apart from the BrainCell-local ": From car to Cav2_3"
line, and `calH.mod`, which the header references, is a sibling in the
same deposit.

### O-RC2006 -- Kca3.1 implementation

.. [1] Rubin, D. B., & Cleland, T. A. (2006). Dynamical mechanisms of
       odor processing in olfactory bulb mitral cells. Journal of
       Neurophysiology, 96(2), 555-568.
       doi:10.1152/jn.00264.2006

PMID 16707721; no PMCID (confirmed absent). The `.mod` header line
"Implemented in Rubin and Cleland (2006) J Neurophysiology" is correct.

### O-BB1993 -- Kca3.1 parameters

.. [1] Bhalla, U. S., & Bower, J. M. (1993). Exploring parameter space
       in detailed single neuron models: simulations of the mitral and
       granule cells of the olfactory bulb. Journal of
       Neurophysiology, 69(6), 1948-1965.
       doi:10.1152/jn.1993.69.6.1948

PMID 7688798; no PMCID. Header line correct; this is indeed the
parameter-fitting paper the header points at.

### O-DV2000 -- Kca3.1 mod-file author

.. [1] Davison, A. P., Feng, J., & Brown, D. (2000). A reduced
       compartmental model of the mitral cell for use in network
       models of the olfactory bulb. Brain Research Bulletin, 51(5),
       393-399.
       doi:10.1016/S0361-9230(99)00256-7

PMID 10715559; no PMCID. The header's affiliation claim is also
confirmed: PubMed's affiliation field reads "Laboratory of
Computational Neuroscience, The Babraham Institute, Babraham,
Cambridge, UK".

### O-FE1998 -- IKur data behind Kv1.5

.. [1] Feng, J., Xu, D., Wang, Z., & Nattel, S. (1998). Ultrarapid
       delayed rectifier current inactivation in human atrial
       myocytes: properties and consequences. American Journal of
       Physiology-Heart and Circulatory Physiology, 275(5),
       H1717-H1725.
       doi:10.1152/ajpheart.1998.275.5.H1717

PMID 9815079; no PMCID. Every field in the `.mod` header, including
the page range, is correct. In 1998 this appeared under the omnibus
title *American Journal of Physiology*, which is what PubMed indexes;
the sectioned title above is the publisher's own and matches the
`ajpheart` DOI namespace.

### O-SZ1998 -- Kv2.2 identification

.. [1] Schmalz, F., Kinsella, J., Koh, S. D., Vogalis, F., Schneider,
       A., Flynn, E. R. M., Kenyon, J. L., & Horowitz, B. (1998).
       Molecular identification of a component of delayed rectifier
       current in gastrointestinal smooth muscles. American Journal of
       Physiology-Gastrointestinal and Liver Physiology, 274(5),
       G901-G911.
       doi:10.1152/ajpgi.1998.274.5.G901

PMID 9612272; no PMCID. Every field of the `.mod` header's
":Reference :" line is correct; it merely abbreviates the end page.
PubMed gives the sixth author as "Flynn E R"; the publisher deposit in
Crossref gives "Elaine R. M. Flynn", which is used. This is the only
field in the record where two authoritative sources disagree.

### O-RA2011 -- the toolchain that generated the Kv2.2 file

.. [1] Ranjan, R., Khazen, G., Gambazzi, L., Ramaswamy, S., Hill,
       S. L., Schurmann, F., & Markram, H. (2011). Channelpedia: an
       integrative and interactive database for ion channels.
       Frontiers in Neuroinformatics, 5, 36.
       doi:10.3389/fninf.2011.00036

PMID 22232598, PMCID PMC3248699. Published as **Schürmann**; ASCII
form used here. Article number 36, no page range.

**Why this record exists.** `Kv2p2_0010_MA20_GrC.mod` is not
hand-written. Its SVN keywords name the generator
(`.../IonChannel/xmlTomod/CreateMOD.c`, EPFL Blue Brain) and its
`BBiD = 10` is the Channelpedia ion-channel ID for Kv2.2 (gene
*KCNB2*), HH model 24. The Open Source Brain NeuroML2 re-derivation of
Channelpedia model 10/24 carries the identical reference string and
annotates `identifiers.org/pubmed/9612272`, and its gating equations
match constant for constant -- so O-SZ1998 is the actual parameter
source, not merely a bibliography line. The `.mod` header's
"$Author: rajnish $" is SVN noise, but it does identify Rajnish
Ranjan, first author above. There is **no standalone ModelDB accession**
for this mechanism; it is redistributed inside larger cell models,
here ModelDB 265584 (see the `MA2020` record).

### O-EV2013 -- Cav1.2 / Cav1.3 GENESIS kinetics

.. [1] Evans, R. C., Maniar, Y. M., & Blackwell, K. T. (2013).
       Dynamic modulation of spike timing-dependent calcium influx
       during corticostriatal upstates. Journal of Neurophysiology,
       110(7), 1631-1645.
       doi:10.1152/jn.00232.2013

PMID 23843436, PMCID PMC4042418. ModelDB accession 150912 (GENESIS).
This is a **striatal medium spiny neuron** paper, not a dentate
granule cell paper -- the `.mod` header's phrasing invites that
misreading. Identification is certain, not merely likely: O-BE2017's
Methods state verbatim that "The Cav1.2 and Cav1.3 (L-type) Ca2+
channel models were taken from (Evans et al., 2013) and transferred
from GENESIS to NEURON", with this exact DOI as its `bib62`, and the
GENESIS sources `CaL12CDI.g` / `CaL13CDI.g` match the ported `.mod`
files parameter for parameter.

### O-BE2017 -- the GENESIS-to-NEURON transfer

.. [1] Beining, M., Mongiat, L. A., Schwarzacher, S. W., Cuntz, H., &
       Jedlicka, P. (2017). T2N as a new tool for robust
       electrophysiological modeling demonstrated for mature and
       adult-born dentate granule cells. eLife, 6, e26517.
       doi:10.7554/eLife.26517

PMID 29165247, PMCID PMC5737656. **The `.mod` header is wrong on both
of the fields it gives.** (1) Year: it says 2016; the paper was
published 22 November 2017. (2) Title: its "A novel comprehensive and
consistent electrophysiologcal model of dentate granule cells" (typo
in the original) corresponds to **no published paper or preprint** --
bioRxiv author search, Crossref preprint search, OpenAlex and Europe
PMC exact-phrase search all return nothing, and the eLife article
declares no preprint relation. Best reading: a pre-publication working
title that was never posted. A withdrawn posting cannot be formally
excluded. The discrepancy is internal to the upstream deposit --
Beining's own repository README attributes the code to the 2017 eLife
paper while still shipping the stale "(2016)" header.

**Explicitly ruled out:** this is *not* the group's Brain Structure
and Function paper (Beining et al., 2017, 222(3), 1427-1446,
doi:10.1007/s00429-016-1285-y, PMID 27514866), which is a
developmental morphology paper containing no channel kinetics. Its
`-016-` DOI suffix is online-first dating and is a likely source of
the year confusion.

### O-ST2011 -- the GENESIS DCN model

Origin of every DCN channel and calcium pool. ModelDB accession 136175.

.. [1] Steuber, V., Schultheiss, N. W., Silver, R. A., De Schutter,
       E., & Jaeger, D. (2011). Determinants of synaptic integration
       and heterogeneity in rebound firing explored with data-driven
       models of deep cerebellar nucleus cells. Journal of
       Computational Neuroscience, 30(3), 633-658.
       doi:10.1007/s10827-010-0282-z

PMID 21052805, PMCID PMC3108018.

### O-LU2011 -- the NEURON translation of that model

.. [1] Luthman, J., Hoebeek, F. E., Maex, R., Davey, N., Adams, R.,
       De Zeeuw, C. I., & Steuber, V. (2011). STD-dependent and
       independent encoding of input irregularity as spike rate in a
       computational model of a cerebellar nucleus neuron. The
       Cerebellum, 10(4), 667-682.
       doi:10.1007/s12311-011-0295-9

PMID 21761198, PMCID PMC3215884. **Note the author list**: the last
author is Steuber and **De Schutter is not on this paper at all** -- a
natural but wrong assumption given the surrounding lineage. Its
Methods state verbatim that the model, "originally implemented in
GENESIS", "was translated to NEURON", citing O-ST2011.

### O-SW1999 -- inferior olive compartmental model

Origin of the IO `Na`, `Kdr` and `HCN` mechanisms.

.. [1] Schweighofer, N., Doya, K., & Kawato, M. (1999).
       Electrophysiological properties of inferior olive neurons: A
       compartmental model. Journal of Neurophysiology, 82(2),
       804-817.
       doi:10.1152/jn.1999.82.2.804

PMID 10444678; **no PMCID**. Free readability was **not** verified --
there is no PMC copy and the publisher paywall was not tested.

### O-MN1997 -- inferior olive low-amplitude oscillations

Origin of the IO `Ca` mechanism.

.. [1] Manor, Y., Rinzel, J., Segev, I., & Yarom, Y. (1997).
       Low-amplitude oscillations in the inferior olive: A model based
       on electrical coupling of neurons with heterogeneous channel
       densities. Journal of Neurophysiology, 77(5), 2736-2752.
       doi:10.1152/jn.1997.77.5.2736

PMID 9163389; **no PMCID**. Free readability not verified, same
reason. The `.mod` header's "Manor (Rinzel, Segev, Yarom) 1997"
parenthesises the co-authors; all four are authors of record.

### O-TN2012 -- the NEURON port of the IO channels

.. [1] Torben-Nielsen, B., Segev, I., & Yarom, Y. (2012). The
       generation of phase differences and frequency changes in a
       network model of inferior olive subthreshold oscillations. PLOS
       Computational Biology, 8(7), e1002580.
       doi:10.1371/journal.pcbi.1002580

PMID 22792054, PMCID PMC3390386. ModelDB accession 144502. This is the
"B. Torben-Nielsen @ HUJI, 2010" credit in the IO `.mod` headers --
the deposit from which the `Ca`, `Kdr` and `Na` files were ported
before the `ZH2019` model reused them.

---

## NO_KEY  (32 symbols)

### Symbols

- `braincell/channel/_base.py::ghk_flux`
- `braincell/channel/_base.py::Gate`
- `braincell/channel/_base.py::Transition`
- `braincell/channel/_base.py::HH`
- `braincell/channel/_base.py::OhmicHH`
- `braincell/channel/_base.py::Markov`
- `braincell/channel/leaky.py::LeakageChannel`
- `braincell/channel/leaky.py::IL`
- `braincell/channel/potassium.py::K_Leak`
- `braincell/channel/potassium.py::K_Kv_test`
- `braincell/ion/_base.py::Factor`
- `braincell/ion/_base.py::Species`
- `braincell/ion/_base.py::Reaction`
- `braincell/ion/_base.py::Source`
- `braincell/ion/_base.py::Conserve`
- `braincell/ion/_base.py::FixedIon`
- `braincell/ion/_base.py::InitNernstIon`
- `braincell/ion/_base.py::DynamicNernstIon`
- `braincell/ion/_base.py::KineticIon`
- `braincell/ion/calcium.py::Calcium`
- `braincell/ion/calcium.py::CalciumFixed`
- `braincell/ion/calcium.py::CalciumInitNernst`
- `braincell/ion/calcium.py::CalciumDetailed`
- `braincell/ion/calcium.py::CalciumFirstOrder`
- `braincell/ion/nonspecific.py::NonSpecific`
- `braincell/ion/nonspecific.py::NonSpecificFixed`
- `braincell/ion/potassium.py::Potassium`
- `braincell/ion/potassium.py::PotassiumFixed`
- `braincell/ion/potassium.py::PotassiumInitNernst`
- `braincell/ion/sodium.py::Sodium`
- `braincell/ion/sodium.py::SodiumFixed`
- `braincell/ion/sodium.py::SodiumInitNernst`

### Provenance evidence

No `.mod` file provenance applies to this bucket. These are BrainCell's own
template/base classes (`Gate`, `Transition`, `HH`, `OhmicHH`, `Markov`,
GHK helper), a generic leak channel, a test-only stub (`K_Kv_test`), and
the abstract ion-state container hierarchy (`Factor`, `Species`,
`Reaction`, `Source`, `Conserve`, fixed/Nernst/kinetic ion mixins). They are
not literature-derived channel models and carry no citation key by
construction. Nothing further to verify for most of these; any docstring
math should cite the standard HH (1952) formalism or the relevant ion
model paper only where the *implementation* (not the class scaffolding)
draws on it.

> **Correction (Task 3 follow-up, 2026-08-15).** "Nothing further to
> verify for most of these" is right about 29 of the 32 symbols and
> wrong about three. `ghk_flux` implements a named published equation,
> `KineticIon` reproduces a documented NEURON language feature by name,
> and `CalciumDetailed` already ships a `References` section in the
> source tree. Those three are verified below. The remaining 29 are
> confirmed as having no primary literature source, and that
> confirmation is now written out symbol by symbol rather than left
> implicit -- see `### Symbols with no primary literature source`.

### Verified record

**Three of the 32 symbols carry a real primary source; 29 do not.**
The three are verified here. The 29 are enumerated, with reasons, in
`### Symbols with no primary literature source` below -- that list is
the complete determination, not a summary of one, and a later task
builds its allowlist from it.

All records below were confirmed 2026-08-15 against at least two
independent sources: the Crossref REST API, NCBI E-utilities
(`efetch`, `db=pubmed` and `db=pmc`), Europe PMC, dblp, Semantic
Scholar, OpenLibrary, and -- where a rendered publisher page was
unreachable -- the publisher-deposited JATS XML or the authors' own
extended preprints. Sources are named per record.

**N-GHK -- `braincell/channel/_base.py::ghk_flux`.** Two entries, in
this order:

.. [1] Goldman, D. E. (1943). Potential, impedance, and rectification
       in membranes. The Journal of General Physiology, 27(1), 37-60.
       doi:10.1085/jgp.27.1.37

.. [2] Hodgkin, A. L., & Katz, B. (1949). The effect of sodium ions on
       the electrical activity of the giant axon of the squid. The
       Journal of Physiology, 108(1), 37-77.
       doi:10.1113/jphysiol.1949.sp004310

Goldman: PMID 19873371, PMCID PMC2142582; Crossref, PubMed, Europe PMC
and the RUPress JATS deposit all agree on 27(1), 37-60. **The printed
title is set in full capitals** (`POTENTIAL, IMPEDANCE, AND
RECTIFICATION IN MEMBRANES`) -- that is the journal's 1943 house style
for every article, so down-casing it to sentence case is the house
rule's normal operation, not a silent edit to a publisher's wording.

Hodgkin & Katz: PMID 18128147, PMCID PMC1392331. **One source
disagreement, resolved.** PubMed and Europe PMC print "... electrical
activity of giant axon of the squid", dropping the definite article;
Crossref and the Physiological Society's own JATS deposit both print
"... of **the** giant axon of the squid". Two publisher-sourced
records outweigh one cataloguer-sourced one, so "the giant axon" is
used above. The article is free-to-read but not open access, and its
PMC deposit is a scanned PDF with no text layer.

**Ordering is not cosmetic here.** Goldman 1943 and Hodgkin & Katz
1949 are the two halves of the name "GHK", but they are the primary
sources for two *different* equations, and `ghk_flux` computes only
one of them -- see the attribution check below. Goldman is the primary
source for the constant-field current/flux equation and therefore
takes `.. [1]`. A docstring that cites Hodgkin & Katz alone, or that
puts it first, is citing the constant-field *voltage* equation, which
this function does not compute.

**N-NRN -- `braincell/ion/_base.py::KineticIon`.** Two entries:

.. [1] Hines, M. L., & Carnevale, N. T. (2000). Expanding NEURON's
       repertoire of mechanisms with NMODL. Neural Computation, 12(5),
       995-1007.
       doi:10.1162/089976600300015475

.. [2] Carnevale, N. T., & Hines, M. L. (2006). The NEURON book.
       Cambridge University Press.
       doi:10.1017/CBO9780511541612

Hines & Carnevale 2000: PMID 10905805, MIT Press, ISSN 0899-7667.
Confirmed against Crossref, PubMed, dblp, Semantic Scholar, the
official NEURON "Publications about NEURON" list and the authors' own
Yale page -- six sources, all agreeing on 12(5), 995-1007. **One
disagreement, resolved.** The authors' extended preprint
(`nmodl400.pdf`) carries "Neural Computation 12:839-851, 2000" on its
own cover page; that is a stale pre-publication estimate contradicted
by all six sources above, and third-party summaries still propagate
it. Use 995-1007. The published title is set in title case by MIT
Press (`Expanding NEURON's Repertoire of Mechanisms with NMODL`);
down-casing to sentence case is the house rule, and `NEURON` and
`NMODL` are preserved as capitalised acronyms.

The NEURON book: ISBN-13 9780521843218 (hardback, ISBN-10
0521843219), 9780521115636 (paperback), 9780511541612 (online); 478
pages; LCCN 2006277066. Per the house style, the ISBN is recorded here
and not in the `.. [N]` entry, and no place of publication is given.
Chapter 9, "How to expand NEURON's library of mechanisms", pp.
207-264, is the relevant chapter (chapter DOI
10.1017/CBO9780511541612.010). Confirmed against Crossref (book record
plus all sixteen chapter records) and OpenLibrary. Cambridge Core
itself returned HTTP 403, so the chapter page range rests on the
Crossref chapter deposit.

**Why the 1997 paper is not the citation here, and would be wrong.**
The obvious candidate is Hines & Carnevale (1997), "The NEURON
simulation environment", Neural Computation 9(6), 1179-1209,
doi:10.1162/neco.1997.9.6.1179. Three independent findings rule it
out, all from a full-text search of the authors' own extended preprint
(re-run 2026-08-15):

1. **It contains no NMODL block keyword of any kind.** Not just
   `KINETIC`, `COMPARTMENT` and `CONSERVE`: the count is **zero** for
   every one of `SOLVE`, `BREAKPOINT`, `DERIVATIVE`, `PARAMETER`,
   `SUFFIX`, `USEION`, `NONLINEAR`, `LINEAR` and `PROCEDURE` as well.
   The paper does not exhibit the language it would be cited for.
2. **It contains no `.mod` listing.** The string `.mod` occurs zero
   times; there is no mechanism source anywhere in the paper.
3. **It explicitly forward-references its successor.** Verbatim: *"In
   a future publication we will examine how the NMODL translator is
   used to define new membrane channels and calculate ionic
   concentration changes."* The 2000 paper is that future publication.

The paper also says as much in passing -- *"An extensive discussion of
NMODL is beyond the scope of this article, but its major advantages
can be listed succinctly."* -- but finding 3 is the decisive one,
because it names the successor rather than merely declining the topic.
Citing the 1997 paper for `KINETIC`/`COMPARTMENT` points the reader at
a paper that defers the subject to another. It remains the right
citation for NEURON the simulator in general; it is the wrong one for
this class.

**A precision the earlier revision got wrong.** The keyword counts
above are for the **uppercase NMODL block keywords** as they appear in
mechanism source. Lowercase prose *does* discuss the concept: the
phrase "kinetic schemes" occurs three times in the 1997 preprint
(e.g. "allows the expression of models in terms of kinetic schemes",
"Mechanisms described by kinetic schemes are written with a syntax in
which the reactions are clearly apparent"). A claim of "zero
occurrences of `KINETIC`" is therefore only true case-sensitively, and
must be stated that way. **The verdict is unaffected** -- naming the
concept in one clause is not documenting the language construct -- but
a case-insensitive re-audit would otherwise appear to refute this
block.

Also ruled out, by full-text search returning zero hits for the
uppercase keywords: Hines & Carnevale (2001) in *The Neuroscientist*,
and Awile et al. (2022), doi:10.3389/fninf.2022.884046, which covers
the build system and transpiler rather than language semantics.

**N-CAD -- `braincell/ion/calcium.py::CalciumDetailed`.** Three
entries, ordered by the file's own `.. [1]` = origin-of-kinetics /
`.. [2]` = model-imported-from rule (see `## How to assemble a
two-or-more-level citation`). Entries [2] and [3] supersede the two that the
source tree already carries at `braincell/ion/calcium.py:227-233`; the
existing pair has a title error and no DOIs. See `## Corrections to
pre-existing in-code citations` items 1 and 2, whose *record* checks
these confirm independently, and whose deferred *attribution* checks
are closed below.

.. [1] Destexhe, A., Contreras, D., Sejnowski, T. J., & Steriade, M.
       (1994). A model of spindle rhythmicity in the isolated thalamic
       reticular nucleus. Journal of Neurophysiology, 72(2), 803-818.
       doi:10.1152/jn.1994.72.2.803

.. [2] Bazhenov, M., Timofeev, I., Steriade, M., & Sejnowski, T. J.
       (1998). Cellular and network models for intrathalamic
       augmenting responses during 10-Hz stimulation. Journal of
       Neurophysiology, 79(5), 2730-2748.
       doi:10.1152/jn.1998.79.5.2730

.. [3] Destexhe, A., Babloyantz, A., & Sejnowski, T. J. (1993). Ionic
       mechanisms for intrinsic slow oscillations in thalamic relay
       neurons. Biophysical Journal, 65(4), 1538-1552.
       doi:10.1016/S0006-3495(93)81190-1

**Why entry [1] exists: Bazhenov is not the origin of this model.**
An earlier revision of this block treated Bazhenov et al. (1998) as
the origin of the first-order calcium model and gave `CalciumDetailed`
only a two-entry block that did not follow this file's `[1]` = origin
/ `[2]` = model rule. Bazhenov's own Appendix credits it: equation
(A5) is introduced with the words "the Ca 2+ dynamics is described by
a simple first-order model (Destexhe et al. 1994a)", and the paper's
reference list resolves `1994a` to "DESTEXHE, A., CONTRERAS, D.,
SEJNOWSKI, T. J., AND STERIADE, M. A model of spindle rhythmicity in
the isolated thalamic reticular nucleus. J. Neurophysiol. 72:
803-818, 1994a." Read directly from the author-hosted PDF of the
Bazhenov paper, 2026-08-15. That is **the same paper already verified
in this file under the `De1994` key** (`## De1994`, PMID 7527077); the
entry above is copied verbatim from that key's `### Verified record`
and is not a fresh citation. Entry [2] remains the right *model*
citation -- it is the deposit BrainCell's parameter defaults come
from, and it is what the class's existing docstring names.

Destexhe: PMID 8274647, PMCID PMC1225880. Re-confirmed here by direct
`efetch db=pubmed`, which returns the title ending in "thalamic relay
**neurons**" -- plural, as Task 2 found; the in-code entry's "neuron"
is wrong. Bazhenov: PMID 9582241; the same `efetch` call returns
79(5), 2730-2748, and the publisher-typeset PDF's embedded metadata
(`Subject: J. Neurophysiol. 79: 2730-2748, 1998`) independently
confirms the same range across 19 pages. Both records agree with the
corrected forms Task 2 published, from a separate query.

**Caveat on open-access status, for anyone re-auditing.** OpenAlex,
Unpaywall and Semantic Scholar all report the Bazhenov paper as closed
with zero open-access locations, while an author-hosted copy is in
fact reachable. That is a simultaneous false negative in three
aggregators; do not use them to decide whether a reference here is
obtainable. The companion paper -- Bazhenov, Timofeev, Steriade &
Sejnowski (1998), "Computational models of thalamocortical augmenting
responses", *The Journal of Neuroscience* 18(16), 6444-6465, PMID
9698334 -- *is* in PMC as PMC6793176, if a docstring ever needs it.

### Attribution

**Attribution check: PASSED for `ghk_flux` and `KineticIon`. PASSED
with a correction for `CalciumDetailed`, whose two entries are both
verified but are not both descriptions of what the class computes.**

**`ghk_flux` -- checked against the implementation, not the name.**
Code read: `braincell/channel/_base.py:33-40`. The function is five
lines and computes, with `zeta = z*F*V/(R*T)`:

```
regular_branch = z * zeta * F * (ci - co*exp(-zeta)) / (1 - exp(-zeta))
```

Substituting `zeta` gives `z^2 F^2 V / (R T) * (ci - co e^{-zeta}) /
(1 - e^{-zeta})`, which is the Goldman constant-field **current**
equation divided through by the permeability. The permeability is
supplied by the caller: `braincell/channel/calcium.py:1002-1013`
(`Cav2p1_RI2021_SC.current`) returns `-g_max * conductance_factor(V,
Ca) * drive`, so `g_max` occupies the `P_s` slot. The function takes
`V` as an *input* and returns a flux; the constant-field **voltage**
equation takes concentrations and permeabilities and returns a
potential. The two cannot be confused once the signature is read, and
this is the check the coordinator asked for: **`ghk_flux` computes the
flux/current equation, so Goldman 1943 is its primary source.**

That Goldman 1943 is the source of the current equation is verified
from the paper's own OCR'd full text in the PMC deposit, which gives
it as equation (11), `J_i = (u_i F / a) dV (n_i^0 e^{-z_i theta dV} -
n_i)/(e^{-z_i theta dV} - 1)`, generalised at equation (17). That
Hodgkin & Katz 1949 is the source of the voltage equation is verified
from a peer-reviewed secondary source rather than the primary text,
which is not retrievable: Alvarez, R., & Latorre, R. (2017), "The
enduring legacy of the 'constant-field equation' in membrane ion
transport", *The Journal of General Physiology* 149(10), 911,
PMC5688357. **Recorded as a limitation:** the Hodgkin & Katz half of
this attribution rests on that review, not on the 1949 paper's text.
The review is a JGP centenary tribute written about the Goldman paper
by domain experts, and it cross-checks against the primary Goldman
text (its claim that the voltage equation is "formally equivalent to
Eq. 18 in Goldman's 1943 paper" matches Goldman's equation (18) as
read directly), so the limitation is narrow. It is nevertheless not a
primary-source confirmation and must not be presented as one.

*Caveat, and it belongs in the docstring.* The `small_branch` return
value, `z*F*(ci - co*e^{-zeta})*(1 + zeta/2)`, is **not** from either
paper. It is the first-order Taylor truncation of `zeta/(1 -
e^{-zeta}) = 1 + zeta/2 + zeta^2/12 + ...`, selected by
`where(|1 - exp(-zeta)| <= 1e-6, ...)` to keep the removable
singularity at `V = 0` finite and differentiable under JAX. It is a
BrainCell numerical-stability addition. The same applies to the two
private variants in `braincell/channel/calcium.py:78-96`,
`_cav3p1_nmodl_ghk_flux` and `_cav3p3_nmodl_ghk_flux`, which differ
from the public helper only in substituting the Faraday, gas-constant
and Kelvin-offset literals that the corresponding NMODL sources use in
place of the `brainunit` constants; their expansions
(`1 + zeta/2` and `1 - w/2` respectively) were both checked and are
correct to first order. Those two are private and not in `__all__`, so
they need no `References` section of their own.

**`KineticIon` -- checked against the NMODL constructs it names.**
Code read: `braincell/ion/_base.py:335-351` (the class and its
`Notes`, which state the semantics "matching NEURON's
``KINETIC``/``COMPARTMENT`` behavior") and `43-147` (the five frozen
declaration dataclasses it consumes). The mapping is one-for-one, and
each element was matched against verbatim text in the 2000 paper's
extended preprint and in the public draft of NEURON Book chapter 9:

| BrainCell declaration | NMODL construct | Evidence |
|---|---|---|
| `Reaction(lhs, rhs, forward, backward)` | `~ A + B <-> C (kf, kb)` | "The first character of a reaction statement is the tilde `~`"; worked example `~ cai + pump <-> capump (k1,k2)` |
| `Factor(name, value)` + `Species(..., factor=)` | `COMPARTMENT vol {states}` | "where the STATEs named in the braces have the same compartment volume given by the volume expression after the COMPARTMENT keyword. The volume merely multiplies the dSTATE/dt left hand side" |
| `Conserve(species, algebraic, total)` | `CONSERVE` | "the user is allowed to explicitly specify conservation equations with the CONSERVE statement" |
| `Species(name, init, factor)` | `STATE` | kinetic-scheme unknowns: "The NMODL translator converts the kinetic scheme into a family of ODEs whose unknowns are the STATEs" |

The `COMPARTMENT` semantics quoted above is exactly what
`KineticIon`'s `Notes` describes: the factor multiplies the derivative
side only, which is why the class stores species in visible units and
converts only temporarily inside conservation and derivative mapping.
Corroboration that this is still current NEURON semantics and not a
2000-vintage description: the live official NMODL language reference
at `nrn.readthedocs.io` carries the `COMPARTMENT` and `CONSERVE`
paragraphs **word for word** identical to the 2000 text. *Caveat:*
that documentation page carries no citation of its own -- zero
occurrences of "Hines", "Carnevale" or "Neural Comput" -- so it
corroborates the semantics, not the attribution.

Two limitations, both recorded rather than papered over. First, the
published 13-page article could not be read (MIT Press and the ACM
Digital Library both returned HTTP 403); the `KINETIC`/`COMPARTMENT`
text was verified in the authors' own much longer extended preprint of
it, and the published abstract calls itself "a summary". That is why
the book is cited alongside as `.. [2]` rather than optionally: NEURON
Book chapter 9 was verified directly from the authors' public draft
and carries the same `COMPARTMENT` passage and the same worked
examples, including `COMPARTMENT (1e10)*parea {pump pumpca}` -- the
very idiom `braincell/ion/calcium.py:1317-1318` describes when
explaining an imported mechanism. Second, do not quote a numbered
section such as "9.10.1" from the book; the draft uses example
numbering and Cambridge Core was unreachable.

*Scope caveat.* This reference belongs on `KineticIon`, the template
that implements the semantics. The five declaration dataclasses are
inert records with no behaviour, so they ship without a `References`
section and point at `KineticIon` through `See Also` instead. That is
a judgement call, not a finding -- flagged as item 15 below.

**`CalciumDetailed` -- the correction.** Code read:
`braincell/ion/calcium.py:127-269` (the 106-line docstring, the
constructor and `derivative`) and `61-81` (the `Calcium` base). The
class's entire dynamics is four lines:

```
drive = total_current / (2 * u.faraday_constant * self.d)
drive = u.math.maximum(drive, u.math.zeros_like(drive))
return drive + (self.C_rest - Ci) / self.tau
```

*Entries [1] and [2], Destexhe et al. 1994 / Bazhenov et al. 1998:
attribution PASSED.* That expression is term for term the first-order
model the docstring's own section 2 attributes to Bazhenov -- influx
`I_Ca/(zFd)` with `z = 2` hard-coded as the literal `2`, plus
relaxation `([Ca]_rest - [Ca])/tau`. The constructor exposes exactly
and only that model's three parameters: `d`, `tau`, `C_rest`. Entry
[1] is the model's origin and entry [2] the deposit the defaults come
from; a module task should carry both, in that order. One shape
difference worth knowing and not worth flagging as a divergence:
Bazhenov writes the influx term with a single lumped constant,
`A = 5.18e-5 mM cm2/(ms uA)`, where BrainCell (like the docstring's
own section 2) writes it out as `1/(zFd)`. The two are the same term
parameterised differently, which is why `d` is a BrainCell parameter
and not a Bazhenov one.

*Entry [3], Destexhe et al. 1993: the paper is correctly identified
and correctly recorded, but the class does not implement it.* The
docstring's section 1 sets out the ATP-driven pump, its kinetic scheme
`Ca_i + P <-> CaP -> P + Ca_o`, and its Michaelis-Menten reduction
`d[Ca]_i/dt = -K_T [Ca]_i/([Ca]_i + K_d)` with `K_T = 1e-4 mM/ms` and
`K_d = 1e-4 mM`. **None of it is implemented.** A grep of the whole
module for `K_T`, `Kd`, `1e-4` and `Michaelis` returns exactly one hit
-- the word "Michaelis-Menten" in that docstring's prose. There is no
saturating term anywhere in `derivative`, and no pump parameter in the
constructor. Task 2 deferred this check and, in doing so, recorded the
assumption that the abstract's "included Ca2+ diffusion" was
"consistent with the Michaelis-Menten Ca pump the class implements";
**that assumption is wrong and is corrected here.**

The consequence for the module task is narrow and specific: keep
entry `.. [3]` only for as long as the docstring keeps its expository
section 1. `.. [3]` is a correct citation *for the text it supports*.
It must not be presented as the source of
`CalciumDetailed.derivative`, and if the docstring is ever trimmed to
what the class computes, `.. [3]` goes with the prose it belongs to
and entries [1] and [2] -- Destexhe et al. 1994 and Bazhenov et al.
1998 -- become the whole reference block.

*Parameter caveats, none of them citation errors.* `d = 1.0 um` and
the `Calcium` base's `default_Co = 2.0 mM` and `default_valence = 2`
match the docstring's stated values.

**`C_rest` and `tau` are Bazhenov's own values, and the docstring's
"0.05 uM" is the error.** An earlier revision of this block had this
exactly backwards -- it recorded `C_rest = 2.4e-4 mM` as "BrainCell's
choice rather than the paper's value" and `tau = 5.0 ms` as
unconfirmed. Both statements were false, and shipping either into a
docstring would have published a false claim about the paper.
Bazhenov et al. (1998), Appendix, equation (A5) reads verbatim:

```
For both the RE and TC cells, the Ca 2+ dynamics is described
by a simple first-order model (Destexhe et al. 1994a)
  d[Ca]/dt = -A I_T - ([Ca] - [Ca]_inf)/tau            (A5)
where [Ca]_inf = 2.4e-4 mM is equilibrium Ca 2+ concentration,
A = 5.18e-5 mM cm2/(ms uA) and tau = 5 ms.
```

Read from the author-hosted PDF of the paper, 2026-08-15. The code at
`braincell/ion/calcium.py:245-246` defaults `tau = 5.0 * u.ms` and
`C_rest = 2.4e-4 * u.mM` (and `Ci_initializer` to the same
`2.4e-4 mM`). **Both are verbatim the paper's values and may be
presented as such.** Neither is a divergence and neither needs a
caveat.

*What the module task must fix instead.* The error is in the existing
docstring **prose**, not in the defaults. `braincell/ion/calcium.py:193`
states the resting concentration as `.05 uM`, which is neither the
paper's `2.4e-4 mM` (= 0.24 uM) nor the code's default. That number is
wrong and must be **corrected to 2.4e-4 mM (0.24 uM), or removed**, by
whichever module task rewrites this docstring. Do not preserve it, and
do not "reconcile" it by relabelling the correct defaults as
BrainCell's own -- that is the inversion this paragraph replaces.

The docstring quotes `F = 96489 C/mol` and `R = 8.31441 J/(mol K)` from
the paper while the code uses `u.faraday_constant` and, through
`DynamicNernstIon`, `u.gas_constant` -- the CODATA values. The default
temperature `u.celsius2kelvin(36.0)` is 309.15 K, matching the
docstring. Finally, `maximum(drive, 0)` rectifies the influx term so
that only inward calcium current raises `Ci`; no such clamp appears in
either paper, and it is a BrainCell addition on the same footing as
the `ghk_flux` small-`zeta` branch.

*What was not closed on the paper side.* Three items about the
Destexhe paper's own numbers were sought and not resolved: an apparent
factor-of-100 inconsistency between `k = 0.1` and `k = 10` in its
equation (7); a disagreement between the 1 um shell depth stated in
the paper and the 0.1 um used by the widely circulated `cad.mod`
implementation of it; and the absence of individual values for the
rate constants `c1`, `c2`, `c3`, which the paper gives only through
the lumped `K_T` and `K_d`. **None of the three affects the verdict**,
which rests on the implementation containing no pump at all, but they
are recorded as item 14 below so nobody re-derives them.

### Symbols with no primary literature source

**The remaining 29 symbols have no primary literature source and are
expected to ship without a `References` section.** This is a
determination, not an omission: each was inspected and the reason is
given. A later task builds its allowlist from exactly this list, so it
is written out symbol by symbol.

| # | Symbol | Why no reference |
|---|---|---|
| 1 | `braincell/channel/_base.py::Gate` | Declaration dataclass for one HH gate (name, power, q10, temp_ref). No equations, no constants. |
| 2 | `braincell/channel/_base.py::Transition` | Declaration dataclass for one Markov transition. Same. |
| 3 | `braincell/channel/_base.py::HH` | Generic gate-template base. Carries no rate expressions and no parameters; subclasses supply both. |
| 4 | `braincell/channel/_base.py::OhmicHH` | `HH` plus the ohmic current law `g_max * f * (E - V)`. Textbook, parameterless. |
| 5 | `braincell/channel/_base.py::Markov` | Generic state-transition template. No scheme of its own. |
| 6 | `braincell/channel/leaky.py::LeakageChannel` | Abstract base; every method is `pass` or `NotImplementedError`. |
| 7 | `braincell/channel/leaky.py::IL` | `g_max * (E - V)` with conventional defaults `0.1 mS/cm2`, `-70 mV`. No source model. |
| 8 | `braincell/channel/potassium.py::K_Leak` | Same, `0.005 mS/cm2`, reversal taken from the ion object. |
| 9 | `braincell/channel/potassium.py::K_Kv_test` | Scratch/template fixture: `g_max` defaults to zero, `Q10_n` to 1.0. See the caveat below. |
| 10 | `braincell/ion/_base.py::Factor` | Frozen declaration dataclass. See `KineticIon`. |
| 11 | `braincell/ion/_base.py::Species` | Frozen declaration dataclass. See `KineticIon`. |
| 12 | `braincell/ion/_base.py::Reaction` | Frozen declaration dataclass. See `KineticIon`. |
| 13 | `braincell/ion/_base.py::Source` | Frozen declaration dataclass. See `KineticIon`. |
| 14 | `braincell/ion/_base.py::Conserve` | Frozen declaration dataclass. See `KineticIon`. |
| 15 | `braincell/ion/_base.py::FixedIon` | Mixin: stores `Ci`/`Co`/`E` as constants. No model. |
| 16 | `braincell/ion/_base.py::InitNernstIon` | Mixin: computes `E` once from the Nernst equation. See the Nernst note below. |
| 17 | `braincell/ion/_base.py::DynamicNernstIon` | Mixin: recomputes `E` from a dynamic `Ci`. Same note. |
| 18 | `braincell/ion/calcium.py::Calcium` | Species container. Holds `default_Co = 2.0 mM`, `default_valence = 2`. |
| 19 | `braincell/ion/calcium.py::CalciumFixed` | Container + `FixedIon`. |
| 20 | `braincell/ion/calcium.py::CalciumInitNernst` | Container + `InitNernstIon`. |
| 21 | `braincell/ion/calcium.py::CalciumFirstOrder` | `Ca' = max(alpha*I_Ca, 0) - beta*Ca` with `alpha = 0.13`, `beta = 0.075`. Generic first-order form; no paper identified. See below. |
| 22 | `braincell/ion/nonspecific.py::NonSpecific` | Container. `default_Ci = default_Co = 1.0 mM`, `valence = 1` -- placeholders, not measurements. |
| 23 | `braincell/ion/nonspecific.py::NonSpecificFixed` | Container + `FixedIon`. |
| 24 | `braincell/ion/potassium.py::Potassium` | Container. `default_Ci = 54.4 mM`, `default_Co = 2.5 mM`. |
| 25 | `braincell/ion/potassium.py::PotassiumFixed` | Container + `FixedIon`. |
| 26 | `braincell/ion/potassium.py::PotassiumInitNernst` | Container + `InitNernstIon`. |
| 27 | `braincell/ion/sodium.py::Sodium` | Container. `default_Ci = 10.0 mM`, `default_Co = 140.0 mM`. |
| 28 | `braincell/ion/sodium.py::SodiumFixed` | Container + `FixedIon`. |
| 29 | `braincell/ion/sodium.py::SodiumInitNernst` | Container + `InitNernstIon`. |

Four notes, so that the list is unambiguous rather than merely short:

1. **`HH` and `OhmicHH` do not take `HH1952`.** The temptation is
   obvious and should be resisted. These are parameterless templates,
   not the squid-axon model; the squid-axon model is already covered
   by the `HH1952` key, which owns `K_HH1952` and `Na_HH1952`. A
   module task may name the Hodgkin-Huxley formalism in `Notes` prose,
   but the `References` section stays absent.
2. **The Nernst equation is not cited.** `InitNernstIon`,
   `DynamicNernstIon` and `KineticIon.E` all evaluate
   `E = (RT/zF) ln(Co/Ci)`. It is a nineteenth-century textbook
   result; house policy for textbook results is no citation, and
   applying it here keeps `KineticIon`'s `References` section about
   the NMODL semantics that are actually distinctive.
3. **The default concentrations are conventions, not data.** `54.4 /
   2.5 mM` for potassium, `10 / 140 mM` for sodium and `2.0 mM`
   external calcium are the values conventional in this literature
   -- potassium's pair yields `E_K` near -82 mV at 36 C -- but no
   single paper is their source, and they are constructor defaults
   that any caller overrides. Recorded explicitly so that no later
   task asserts a source for them.
4. **Two symbols are judgement calls rather than clean determinations,
   and are flagged as such.** `K_Kv_test`'s rate form,
   `Ra*(V - V12)/(1 - exp(-(V - V12)/q))`, is the generic `vtrap`
   alpha/beta idiom that recurs across dozens of unrelated NEURON
   `kv.mod` files; its name, its zero default conductance and its unit
   `Q10` mark it as a fixture, and no specific source was pursued.
   `CalciumFirstOrder`'s `alpha = 0.13`, `beta = 0.075` are likewise
   not traceable to a paper from the code alone. Both are recorded as
   no-source; see item 16 below.

---

## MA2020  (32 symbols)

### Symbols

- `braincell/channel/calcium.py::Cav1p2_MA2020_GoC`
- `braincell/channel/calcium.py::Cav1p3_MA2020_GoC`
- `braincell/channel/calcium.py::Cav3p1_MA2020_GoC`
- `braincell/channel/calcium.py::Cav3p1_MA2020_GoC_Frozen`
- `braincell/channel/calcium.py::CaHVA_MA2020_GoC`
- `braincell/channel/calcium.py::CaHVA_MA2020_GrC`
- `braincell/channel/calcium.py::Cav2p3_MA2020_GoC`
- `braincell/channel/hyperpolarization_activated.py::HCN1_MA2020_GoC`
- `braincell/channel/hyperpolarization_activated.py::HCN2_MA2020_GoC`
- `braincell/channel/potassium.py::KM_MA2020_GoC`
- `braincell/channel/potassium.py::Kv1p1_MA2020_GoC`
- `braincell/channel/potassium.py::Kv3p4_MA2020_GoC`
- `braincell/channel/potassium.py::Kv4p3_MA2020_GoC`
- `braincell/channel/potassium.py::KM_MA2020_GrC`
- `braincell/channel/potassium.py::Kir2p3_MA2020_GrC`
- `braincell/channel/potassium.py::Kv1p1_MA2020_GrC`
- `braincell/channel/potassium.py::Kv2p2_0010_MA2020_GrC`
- `braincell/channel/potassium.py::Kv3p4_MA2020_GrC`
- `braincell/channel/potassium.py::Kv4p3_MA2020_GrC`
- `braincell/channel/potassium_calcium.py::Kca3p1_MA2020_GoC`
- `braincell/channel/potassium_calcium.py::Kca2p2_MA2020_GoC`
- `braincell/channel/potassium_calcium.py::Kca2p2_MA2020_GrC`
- `braincell/channel/potassium_calcium.py::Kca1p1_MA2020_GoC`
- `braincell/channel/potassium_calcium.py::Kca1p1_MA2020_GrC`
- `braincell/channel/potassium_sodium.py::Kv1p5_MA2020_GrC`
- `braincell/channel/sodium.py::Nav1p6_MA2020_GoC`
- `braincell/channel/sodium.py::Nav_MA2020_GrC`
- `braincell/channel/sodium.py::NaFHF_MA2020_GrC`
- `braincell/ion/calcium.py::CdpStC_CAMOnly_MA2020_GoC`
- `braincell/ion/calcium.py::CdpStC_NoCAM_MA2020_GoC`
- `braincell/ion/calcium.py::CdpStC_MA2020_GoC`
- `braincell/ion/calcium.py::CdpCR_MA2020_GrC`

Mod-file year code: `MA20`. Cell types: `GoC` (Golgi cell), `GrC` (granule
cell). Every symbol above maps 1:1 onto a `<mechanism>_MA20_<GoC|GrC>.mod`
file **except** `CdpStC_NoCAM_MA2020_GoC`, for which no matching `.mod`
file (`CdpStC_NoCAM_MA20_GoC.mod`) exists anywhere under
`examples/neuron_compare/Cerebellum_mod` — see "no provenance evidence"
note below. There is also an unclaimed mod file with no BrainCell symbol:
`GoC/ion/CdpStC_CAMOnly_MA20_GoC.mod` *is* claimed
(`CdpStC_CAMOnly_MA2020_GoC`); no extra unclaimed files were found in this
bucket.

### Provenance evidence

Raw header text (first 25 lines, filtered to
`TITLE|COMMENT|Author|Ref|revis|[0-9]{4}` lines), verbatim, one block per
`.mod` file:

```
=== GoC/channel/CaHVA_MA20_GoC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Author: E.D'Angelo, T.Nieus, A. Fontana
	Last revised: 8.5.2000
ENDCOMMENT

=== GoC/channel/Cav1p2_MA20_GoC.mod
: model from Evans et al 2013, transferred from GENESIS to NEURON by Beining et al (2016), "A novel comprehensive and consistent electrophysiologcal model of dentate granule cells"

=== GoC/channel/Cav1p3_MA20_GoC.mod
: model from Evans et al 2013, transferred from GENESIS to NEURON by Beining et al (2016), "A novel comprehensive and consistent electrophysiologcal model of dentate granule cells"

=== GoC/channel/Cav2p3_MA20_GoC.mod
TITLE Ca R-type channel with medium threshold for activation
(no further TITLE/COMMENT/Author/Ref/revis/year lines in first 25 lines)

=== GoC/channel/Cav3p1_MA20_GoC.mod
TITLE Low threshold calcium current Cerebellum Purkinje Cell Model
COMMENT
Kinetics adapted to fit the Cav3.1 Iftinca et al 2006, Temperature dependence of T-type Calcium channel gating, NEUROSCIENCE
Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.
ENDCOMMENT

=== GoC/channel/HCN1_MA20_GoC.mod
TITLE Cerebellum Golgi Cell Model
COMMENT
Author:L. Forti & S. Solinas
Data from: Santoro et al. J Neurosci. 2000
Last revised: April 2006
ENDCOMMENT

=== GoC/channel/HCN2_MA20_GoC.mod
TITLE Cerebellum Golgi Cell Model
COMMENT
Author:L. Forti & S. Solinas
Data from: Santoro et al. J Neurosci. 2000
Last revised: April 2006
ENDCOMMENT

=== GoC/channel/Kca1p1_MA20_GoC.mod
TITLE Large conductance Ca2+ activated K+ channel mslo
COMMENT
Parameters from Cox et al. (1987) J Gen Physiol 110:257-81 (patch 1).
Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Sungho Hong, Okinawa Institute of Science and Technology, March 2009.
ENDCOMMENT

=== GoC/channel/Kca2p2_MA20_GoC.mod
TITLE SK2 multi-state model Cerebellum Golgi Cell Model
COMMENT
Author:Sergio Solinas, Lia Forti, Egidio DAngelo
Based on data from: Hirschberg, Maylie, Adelman, Marrion J Gen Physiol 1998
Last revised: May 2007
             Jonathan Mapelli, Erik De Schutter and Egidio D`Angelo (2008)
ENDCOMMENT

=== GoC/channel/Kca3p1_MA20_GoC.mod
TITLE Calcium dependent potassium channel
: Implemented in Rubin and Cleland (2006) J Neurophysiology
: Parameters from Bhalla and Bower (1993) J Neurophysiology
:   by Andrew Davison, The Babraham Institute  [Brain Res Bulletin, 2000]

=== GoC/channel/KM_MA20_GoC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Author: A. Fontana
	CoAuthor: T.Nieus Last revised: 20.11.99
ENDCOMMENT

=== GoC/channel/Kv1p1_MA20_GoC.mod
TITLE Voltage-gated low threshold potassium current from Kv1 subunits
COMMENT
Human Kv1.1 expressed in xenopus oocytes: Zerr et al., J Neurosci 18, 2842, 2848, 1998
Reference: Akemann et al., Biophys. J. (2009) 96: 3959-3976

=== GoC/channel/Kv3p4_MA20_GoC.mod
(no TITLE/COMMENT/Author/Ref/revis/year lines in first 25 lines)

=== GoC/channel/Kv4p3_MA20_GoC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Author: E.D'Angelo, T.Nieus, A. Fontana
	Last revised: Egidio 3.12.2003
ENDCOMMENT

=== GoC/channel/Nav1p6_MA20_GoC.mod
TITLE resurgent sodium channel
COMMENT
Based om updated kinetic parameters from Raman and Bean, Biophys.J. 80 (2001) 729
Modified from Khaliq et al., J.Neurosci. 23(2003)4899
Reference: Akemann and Knoepfel, J.Neurosci. 26 (2006) 4602
Date of Implementation: May 2005
ENDCOMMENT

=== GoC/ion/CdpStC_CAMOnly_MA20_GoC.mod
TITLE Calcium accumulation with calmodulin-only subnetwork in Golgi cell model
    Nannuli = 10.9495 (1)

=== GoC/ion/CdpStC_MA20_GoC.mod
COMMENT
1) Extended using parameters from Schmidt et al. 2003.
2) Pump rate was tuned according to data from Maeda et al. 1999
Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.
ENDCOMMENT

=== GrC/channel/CaHVA_MA20_GrC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Author: E.D'Angelo, T.Nieus, A. Fontana
	Last revised: 8.5.2000
ENDCOMMENT

=== GrC/channel/Kca1p1_MA20_GrC.mod
TITLE Large conductance Ca2+ activated K+ channel mslo
COMMENT
Parameters from Cox et al. (1987) J Gen Physiol 110:257-81 (patch 1).
Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Sungho Hong, Okinawa Institute of Science and Technology, March 2009.
ENDCOMMENT

=== GrC/channel/Kca2p2_MA20_GrC.mod
TITLE SK2 multi-state model Cerebellum Golgi Cell Model
COMMENT
Author:Sergio Solinas, Lia Forti, Egidio DAngelo
Based on data from: Hirschberg, Maylie, Adelman, Marrion J Gen Physiol 1998
Last revised: May 2007
             Jonathan Mapelli, Erik De Schutter and Egidio D`Angelo (2008)
ENDCOMMENT

=== GrC/channel/Kir2p3_MA20_GrC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Reference: Theta-Frequency Bursting and Resonance in Cerebellar Granule Cells:Experimental
ENDCOMMENT

=== GrC/channel/KM_MA20_GrC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Author: A. Fontana
	CoAuthor: T.Nieus Last revised: 20.11.99
ENDCOMMENT

=== GrC/channel/Kv1p1_MA20_GrC.mod
TITLE Voltage-gated low threshold potassium current from Kv1 subunits
COMMENT
Human Kv1.1 expressed in xenopus oocytes: Zerr et al., J Neurosci 18, 2842, 2848, 1998
Reference: Akemann et al., Biophys. J. (2009) 96: 3959-3976

=== GrC/channel/Kv1p5_MA20_GrC.mod
TITLE Cardiac IKur  current & nonspec cation current with identical kinetics
: Hodgkin - Huxley type channels, modified to fit IKur data from Feng et al Am J Physiol 1998 275:H1717 - H 1725
	 gKur=0.13195e-3 (S/cm2) <0,1e9>

=== GrC/channel/Kv2p2_0010_MA20_GrC.mod
:[$Revision: 1367 $]
:[$Date: 2010-03-26 15:17:59 +0200 (Fri, 26 Mar 2010) $]
:[$Author: rajnish $]
:Comment :
:Reference :Molecular identification of a component of delayed rectifier current in gastrointestinal smooth muscles. Am. J. Physiol., 1998, 274, G901-11
	SUFFIX Kv2p2_0010_MA20_GrC
	gKv2_2bar = 0.00001 (S/cm2)

=== GrC/channel/Kv3p4_MA20_GrC.mod
(no TITLE/COMMENT/Author/Ref/revis/year lines in first 25 lines)

=== GrC/channel/Kv4p3_MA20_GrC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Author: E.D'Angelo, T.Nieus, A. Fontana
	Last revised: Egidio 3.12.2003
ENDCOMMENT

=== GrC/channel/NaFHF_MA20_GrC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
ENDCOMMENT

=== GrC/channel/Nav_MA20_GrC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
Based on Raman 13 state model. Adapted from Magistretti et al, 2006.
ENDCOMMENT

=== GrC/ion/CdpCR_MA20_GrC.mod
COMMENT
1) Extended using parameters from Schmidt et al. 2003.
2) Pump rate was tuned according to data from Maeda et al. 1999
Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.
Modified by Stefano Masoli, Department Brain and Behavioral Sciences, University of Pavia, 2015
1) Buffer for Granule cell model 2015, without Parvalbumin and Calretinin instead of Calbindin.
ENDCOMMENT
```

**No `.mod` file found** for `CdpStC_NoCAM_MA2020_GoC` (expected filename
`CdpStC_NoCAM_MA20_GoC.mod` does not exist under
`examples/neuron_compare/Cerebellum_mod/GoC/ion/`; only
`CdpStC_CAMOnly_MA20_GoC.mod` and `CdpStC_MA20_GoC.mod` exist there). This
symbol's provenance is unresolved by this harvest — see "Unresolved
attributions" below.

**Inconsistent-author cases in this bucket** (header names an author who
is not the `MA2020`/Masoli-family search target):
`CaHVA_MA20_GoC.mod`, `CaHVA_MA20_GrC.mod`, `Kv4p3_MA20_GoC.mod`,
`Kv4p3_MA20_GrC.mod` (all "Author: E.D'Angelo, T.Nieus, A. Fontana");
`HCN1_MA20_GoC.mod`, `HCN2_MA20_GoC.mod` ("Author:L. Forti & S. Solinas");
`KM_MA20_GoC.mod`, `KM_MA20_GrC.mod` ("Author: A. Fontana" /
"CoAuthor: T.Nieus"); `Kca2p2_MA20_GoC.mod`, `Kca2p2_MA20_GrC.mod`
("Author:Sergio Solinas, Lia Forti, Egidio DAngelo");
`Kv2p2_0010_MA20_GrC.mod` (SVN keyword "Author: rajnish", clearly a
repository-tooling artifact, not a scientific author).

### Verified record

> ## !!! STOP -- `[1]` AND `[2]` DO NOT MEAN WHAT THEY MEAN ELSEWHERE
>
> **In this `MA2020` block only, `.. [1]` = the Golgi paper and
> `.. [2]` = the granule paper. They are two alternatives, and a
> given symbol takes exactly ONE of them.**
>
> Everywhere else in this file -- and in every docstring produced
> from it -- `.. [1]` means *origin of the kinetics* and `.. [2]`
> means *the model BrainCell imported from*, and a docstring carries
> **both**.
>
> **Never copy this block's `[1]` and `[2]` into a docstring as a
> pair.** Doing so would cite the granule-cell paper on a Golgi-cell
> class, or vice versa, and would drop the origin entry entirely.
>
> The correct docstring shape for an `MA2020` symbol is: the
> origin entry or entries from the mapping table as `.. [1]`
> (`.. [2]`, `.. [3]` ...), then **whichever one** of the two papers
> below matches the symbol's cell-type suffix as the final entry,
> renumbered.
>
> - `*_MA2020_GoC` -> the Golgi paper (labelled `[1]` below)
> - `*_MA2020_GrC` -> the granule paper (labelled `[2]` below)
>
> **This hazard is live across all 32 `MA2020` symbols.** It is the
> only place in this file where the bracket digits carry a local
> meaning, and the two labels are not distinguishable by eye once
> copied out of context. Re-read this box before touching any
> `MA2020` docstring.

Confirmed 2026-08-15 against PubMed (`efetch`, `db=pubmed`), the
Crossref REST API, the PMC article HTML for each paper, and the ModelDB
REST API for the accessions. **`MA2020` requires two papers, not one.**
The Task 1 hypothesis of a single Masoli et al. (2020) covering both
cell types is wrong: the Golgi and granule models were published
separately, in different journals, with different co-author sets.
Neither paper covers the other's cell type.

Golgi cell (`GoC` symbols), ModelDB accession 266806:

.. [1] Masoli, S., Ottaviani, A., Casali, S., & D'Angelo, E. (2020).
       Cerebellar Golgi cell models predict dendritic processing and
       mechanisms of synaptic plasticity. PLOS Computational Biology,
       16(12), e1007937.
       doi:10.1371/journal.pcbi.1007937

PMID 33378395, PMCID PMC7837495, fully open access. The accession
appears both in the ModelDB API record and in the paper's own Data
Availability statement.

Granule cell (`GrC` symbols), ModelDB accession 265584:

.. [2] Masoli, S., Tognolina, M., Laforenza, U., Moccia, F., &
       D'Angelo, E. (2020). Parameter tuning differentiates granule
       cell subtypes enriching transmission properties at the
       cerebellum input stage. Communications Biology, 3(1), 222.
       doi:10.1038/s42003-020-0953-x

PMID 32385389, PMCID PMC7210112, fully open access. `222` is the
article number and occupies the page slot; *Communications Biology*
places every article of a volume-year in issue 1, so `3(1)` is
nominal. **Provenance caveat:** unlike the Golgi paper, this one's
Data Availability statement cites only the EBRAINS Knowledge Graph and
an HBP live paper, not ModelDB. Accession 265584 is confirmed from the
ModelDB API record, which names this paper, not from the article.

**Which paper a given symbol takes.** Split strictly on the cell-type
suffix: every `*_MA2020_GoC` symbol takes entry [1], every
`*_MA2020_GrC` symbol takes entry [2]. Seven mechanisms
(`CaHVA`, `KM`, `Kv1p1`, `Kv3p4`, `Kv4p3`, `Kca1p1`, `Kca2p2`) exist in
both buckets with identical kinetics; they still take different model
papers, because they were imported from different deposits.

### Attribution

**Attribution check: PASSED for all 32 symbols**, with three caveats
recorded below that must reach the docstrings.

**Method, applied uniformly across all seven Task 3 keys.** Every
symbol's `.mod` counterpart is present in this repository, so the
`code -> .mod` half of the check was done exhaustively rather than by
sampling: every numeric literal in each BrainCell class was extracted
and set-differenced against every numeric literal in the corresponding
`.mod` file, with NMODL comments and `COMMENT` blocks stripped. Every
rate-function constant matches. The only literals that appear in a
`.mod` file and not in its BrainCell class fall into seven benign
classes, verified individually: (a) GHK/unit-conversion constants
(`8.3145`, `96485`, `1e-6`, `1e3`) handled by `braincell.channel.
_base.ghk_flux`; (b) Kelvin offsets (`273.15`/`273.19`/`273.14`)
handled by `u.celsius2kelvin`; (c) reversal potentials (`ek = -84.69`,
`eca = 140`) supplied by the ion object, not the channel; (d) the Q10
exponent divisor `10` absorbed into `Gate(q10=, temp_ref=)`; (e) NMODL
range annotations such as `<0,1e9>`; (f) `gbar` in `mho/cm2` or
`S/cm2` against BrainCell's `mS/cm2` -- e.g. `KM_MA20_GoC.mod`'s
`gkbar = 0.00025 mho/cm2` is BrainCell's `g_max = 0.25 mS/cm2`
exactly; and (g) the documented NMODL default-precision rewrites (see
`### Import deviations`). The `.mod -> paper` half was done per
mechanism family, against the origin records cited in the mapping
table.

Classes that report a zero literal overlap in that scan are BrainCell
subclasses whose constants live in a base class (e.g.
`Kca1p1_MA2020_GrC(Kca1p1_MA2020_GoC)`); this was confirmed by reading
each `class` line and is a scan artefact, not a mismatch.

**Mapping table.** The `Origin` column gives the origin record(s),
which become `.. [1]` (and `.. [2]`, `.. [3]` where the row lists
more than one). The **final** entry is the `MA2020` model paper.

> **Reminder -- the labels above are local.** The model entry is
> **one** of the two papers in the `### Verified record` block, not
> both: the Golgi paper (labelled `[1]` there) for every `*_GoC`
> symbol, the granule paper (labelled `[2]` there) for every `*_GrC`
> symbol. In the finished docstring it is renumbered to whatever
> follows the origin entries. Copying "`[1]` and `[2]`" out of that
> block as a pair is the specific mistake this warning exists to
> prevent.

| Symbols | Origin `.. [1]` |
|---|---|
| `CaHVA_MA2020_GoC`, `CaHVA_MA2020_GrC`, `Kv4p3_MA2020_GoC`, `Kv4p3_MA2020_GrC`, `KM_MA2020_GoC`, `KM_MA2020_GrC`, `Kir2p3_MA2020_GrC` | O-DA2001 |
| `HCN1_MA2020_GoC`, `HCN2_MA2020_GoC` | O-SO2007a (kinetics), O-SA2000 (data) |
| `Kca2p2_MA2020_GoC`, `Kca2p2_MA2020_GrC` | O-HI1998 (data), O-SO2007a (model) |
| `Kca1p1_MA2020_GoC`, `Kca1p1_MA2020_GrC` | O-CX1997 (parameters), O-AN2012 (model) |
| `Kca3p1_MA2020_GoC` | O-RC2006, O-BB1993, O-DV2000 |
| `Cav1p2_MA2020_GoC`, `Cav1p3_MA2020_GoC` | O-EV2013 (kinetics), O-BE2017 (port) |
| `Cav3p1_MA2020_GoC`, `Cav3p1_MA2020_GoC_Frozen` | O-IF2006 (kinetics), O-AN2012 (model) |
| `Cav2p3_MA2020_GoC` | O-PO2003a (and O-PO2003b) |
| `Kv1p1_MA2020_GoC`, `Kv1p1_MA2020_GrC` | O-ZE1998 (data), O-AK2009 (model) |
| `Kv3p4_MA2020_GoC`, `Kv3p4_MA2020_GrC` | O-KH2003 |
| `Kv1p5_MA2020_GrC` | O-FE1998 |
| `Kv2p2_0010_MA2020_GrC` | O-SZ1998 (data), O-RA2011 (toolchain) |
| `Nav1p6_MA2020_GoC` | O-RB2001 (kinetics), O-KH2003, O-AK2006 |
| `Nav_MA2020_GrC`, `NaFHF_MA2020_GrC` | O-MG2006 (kinetics), O-RB2001 (scheme) |
| `CdpStC_MA2020_GoC`, `CdpStC_CAMOnly_MA2020_GoC`, `CdpStC_NoCAM_MA2020_GoC`, `CdpCR_MA2020_GrC` | O-AN2012 (model), O-SC2003 (buffer parameters), O-MD1999 (pump tuning) |

**Fingerprints checked beyond the literal scan.**

- `Kca1p1_*` (`braincell/channel/potassium_calcium.py`) carries the
  Horrigan-Aldrich allosteric parameter set `Qo = 0.73`,
  `Qc = -0.67`, `L0 = 1806`, `Kc = 11.0e-3 mM`, `Ko = 1.1e-3 mM`,
  `k1 = 1.0e3 /mM`, and the `pf0..pf4` / `pb0..pb4` opening and
  closing rate ladders. These are the BK/mslo parameters of O-CX1997
  patch 1, as the header states -- with the year corrected from 1987
  to 1997.
- `Kca2p2_*` implements the six-state SK2 scheme of O-HI1998 with
  `invc1 = invc2 = 80e-3`, `invc3 = 200e-3`, `invo1 = 1`,
  `invo2 = 100e-3`, `diro1 = 160e-3`, `diro2 = 1.2` (Ca-independent)
  and `dirc2 = 200`, `dirc3 = 160`, `dirc4 = 80` /ms-mM
  (Ca-dependent), plus `diff = 3`.
- `Nav_MA2020_GrC` and `NaFHF_MA2020_GrC` are the same 13-state
  Raman-style Markov scheme. **Correction (Task 13):** an earlier
  revision of this bullet gave one shared constant set for both. That
  is wrong. `ACon` and `AOoff` differ between them, verified by
  reading both constructors:
  `NaFHF_MA2020_GrC` ships `ACon = 0.025` and `AOoff = 0.002`
  (`braincell/channel/sodium.py:1818,1821`), while `Nav_MA2020_GrC`
  ships `ACon = 0.005` and `AOoff = 0.005`
  (`braincell/channel/sodium.py:1592,1595`). `ACoff = 0.5` and
  `AOon = 0.75` /ms are shared, as are the derived
  `a = (Oon/Con)^0.25`, `b = (Ooff/Coff)^0.25`. `NaFHF` adds the
  `Lon`/`Loff` blocked-state ladder (`L3..L6`) on top of the same
  `C1..C5`/`I1..I6`/`O` topology -- it is the same mechanism with the
  slow-blocked branch enabled, not an independent model. This is why
  `NaFHF_MA20_GrC.mod`'s empty `COMMENT` block is not a provenance
  gap: it inherits `Nav_MA20_GrC.mod`'s "Based on Raman 13 state
  model. Adapted from Magistretti et al, 2006."

**Caveat 1 -- `Kv3p4` is not established as a Kv3.4.** See the
paragraph in O-KH2003. The cited paper calls this current "K fast" and
never names a Kv3 subunit. Do not write a docstring sentence claiming
the paper identifies Kv3.4. This caveat applies identically to
`Kv3p4_MA2024_PC`, `Kv3p4_MA2025_BC` and `Kv3p4_RI2021_SC`.

**Caveat 2 -- `Cav1p2` / `Cav1p3` may carry a unit-scale defect
inherited from upstream.** The GENESIS originals (`CaL12CDI.g`,
`CaL13CDI.g`) evaluate the `mTau` linoid in volts and return seconds,
whereas the NEURON ports apply the same numeric coefficients in mV and
declare `mTau (ms)`. Worked at V = 0 mV for Cav1.3, GENESIS gives
tau ~ 0.283 ms and the `.mod` file gives tau ~ 2.9e-5 ms -- about
1e4 times smaller, i.e. effectively instantaneous activation. Cav1.2
shows the same pattern and O-BE2017's Methods mention no intentional
rescaling. **This is derived arithmetic, not a fetched claim, and it
was not confirmed against a NEURON run.** It is not a citation error:
BrainCell faithfully reproduces the `.mod` file. Record it as an open
question for the module task rather than asserting either reading.
Applies identically to `Cav1p2_MA2025_BC` and `Cav1p3_MA2025_BC`.

**Caveat 3 -- `CdpStC_NoCAM_MA2020_GoC` has no `.mod` file of its own,
and this is deliberate.** Task 1 flagged the missing
`CdpStC_NoCAM_MA20_GoC.mod` as unresolved (`## Unresolved
attributions` item 1). It is now resolved: the class is a
BrainCell-factored base, not a port of a file that went missing. Its
literal set matches `BC/ion/CdpStC_MA25_BC.mod` and
`SC/ion/CdpStC_RI21_SC.mod` (21 of 26 literals; the five unmatched are
unit conversions and NMODL range annotations), which are exactly the
GoC `CdpStC` mechanism with the CAM subnetwork commented out -- as
`examples/neuron_compare/Cerebellum_mod/README.md` states in its
"Ion_dyn inherited variants" table. Its citation is therefore
O-AN2012 / O-SC2003 / O-MD1999 plus the `MA2020` GoC paper, the same
as `CdpStC_MA2020_GoC`, and a docstring should say the CAM reactions
were removed rather than implying a distinct published mechanism.

**Parameter-default caveats (not citation errors).** BrainCell's
`g_max` defaults were read against each `.mod` file's `gbar`/`gkbar`
and agree after the `mho/cm2` -> `mS/cm2` conversion wherever both
exist. Where a docstring wants to state a conductance as "the
published value", it must not: these are the *cell-model* deposit's
tuned values, not values from the origin paper.

### Import deviations

Transcribed from `examples/neuron_compare/Cerebellum_mod/README.md`
(tables "TABLE status summary", "Integration method status", "Rate
update placement status", "NMODL numeric default precision"). Put
these in the docstring `Notes` section; do not re-read that file.

**These deviations are already applied to the `.mod` files in this
repository.** The README's status column reads `已连续化` ("now
continuous") with the former range under `原` ("formerly"), so opening
a shipped `.mod` file shows `cnexp` and no `TABLE`. That is the
deviation having been made, not evidence against it. Do not read the
shipped file as refuting a row below.

**`TABLE` removed, replaced by per-call evaluation of the continuous
formula.** The former interpolation range is given because NEURON
clamps to the boundary value outside it, so any recorded BrainCell/
NEURON divergence outside the range is expected.

| Symbol | Former `TABLE` range | Tabulated |
|---|---|---|
| `Kv4p3_MA2020_GoC`, `Kv4p3_MA2020_GrC` | `[-100, 30]` (clamped) | `a_inf`, `tau_a`, `b_inf`, `tau_b` |
| `KM_MA2020_GoC`, `KM_MA2020_GrC` | `[-100, 30]` (clamped) | `n_inf`, `tau_n` |
| `CaHVA_MA2020_GoC`, `CaHVA_MA2020_GrC` | `[-100, 30]` (clamped) | `s_inf`, `tau_s`, `u_inf`, `tau_u` |
| `HCN1_MA2020_GoC`, `HCN2_MA2020_GoC` | `[-100, 30]` (clamped) | `o_fast_inf`, `o_slow_inf`, `tau_f`, `tau_s` |
| `Kir2p3_MA2020_GrC` | `[-100, 100]` | `d_inf`, `tau_d` |
| `Cav2p3_MA2020_GoC` | `[-100, 100]` | indexed `inf`/`tau` |
| `Kca3p1_MA2020_GoC` | V `[-100, 100]`; `cai` `[0, 0.01]` (clamped) | `Yvdep`, `Yconcdep` |

The `Kca3p1` concentration table is the one to watch: `cai` above
0.01 mM was previously clamped to the boundary value.

**`derivimplicit` -> `cnexp`.** `KM_MA2020_GoC`, `KM_MA2020_GrC`,
`Kir2p3_MA2020_GrC`, `Kv1p5_MA2020_GrC`, `Kv4p3_MA2020_GoC`,
`Kv4p3_MA2020_GrC`, `CaHVA_MA2020_GrC`. Each gate ODE is independent,
so the substitution is exact. `CaHVA_MA2020_GoC` was already `cnexp`
upstream and is **not** a substitution.

**Rate-refresh relocation.** `Cav1p2_MA2020_GoC` and
`Cav1p3_MA2020_GoC`: the `rates()` call moved from `BREAKPOINT` into
`DERIVATIVE state`, so `inf`/`tau` are refreshed before the `cnexp`
state update rather than after.

**NMODL default-precision rewrites.** NEURON's generated C writes some
`PARAMETER`/global defaults to about six significant figures, and
BrainCell aligns with the compiled values rather than the `.mod`
source text:

| Symbol | Name | `.mod` source | BrainCell |
|---|---|---|---|
| `Kv4p3_MA2020_GoC`, `Kv4p3_MA2020_GrC` | `Kalpha_a` | `-23.32708` | `-23.3271` |
| " | `Kbeta_a` | `19.47175` | `19.4718` |
| " | `V0beta_a` | `-18.27914` | `-18.2791` |
| " | `V0alpha_b` | `-111.33209` | `-111.332` |
| `HCN1_MA2020_GoC` | `tEf`, `tEs` | `2.302585092` | `2.30259` |
| `CaHVA_MA2020_GoC`, `CaHVA_MA2020_GrC` | `Kalpha_s` | `15.87301587302` | `15.873` |

Ordinary in-formula literals are **not** subject to this rewrite and
keep their source values.

---

## MA2024  (19 symbols)

### Symbols

- `braincell/channel/calcium.py::Cav3p1_MA2024_PC`
- `braincell/channel/calcium.py::Cav3p1_MA2024_PC_Frozen`
- `braincell/channel/calcium.py::Cav2p1_MA2024_PC`
- `braincell/channel/calcium.py::Cav2p1_MA2024_PC_Frozen`
- `braincell/channel/calcium.py::Cav3p3_MA2024_PC_Frozen`
- `braincell/channel/calcium.py::Cav3p2_MA2024_PC`
- `braincell/channel/calcium.py::Cav3p3_MA2024_PC`
- `braincell/channel/hyperpolarization_activated.py::HCN1_MA2024_PC`
- `braincell/channel/potassium.py::Kir2p3_MA2024_PC`
- `braincell/channel/potassium.py::Kv1p1_MA2024_PC`
- `braincell/channel/potassium.py::Kv1p5_MA2024_PC`
- `braincell/channel/potassium.py::Kv3p3_MA2024_PC`
- `braincell/channel/potassium.py::Kv3p4_MA2024_PC`
- `braincell/channel/potassium.py::Kv4p3_MA2024_PC`
- `braincell/channel/potassium_calcium.py::Kca3p1_MA2024_PC`
- `braincell/channel/potassium_calcium.py::Kca2p2_MA2024_PC`
- `braincell/channel/potassium_calcium.py::Kca1p1_MA2024_PC`
- `braincell/channel/sodium.py::Nav1p6_MA2024_PC`
- `braincell/ion/calcium.py::CdpCAM_MA2024_PC`

Mod-file year code: `MA24`. Cell type: `PC` (Purkinje cell). All 19 symbols
map 1:1 onto a `<mechanism>_MA24_PC.mod` file. Two extra `.mod` files exist
in `PC/channel/` that are **not** claimed by any `MA2024` symbol —
`Cav3_1_test.mod` and `Cav3_1_test2.mod` — see the `PC24` key section
below; the `Cav3p1Test_PC24` symbol's own docstring already identifies
`Cav3_1_test.mod` as its source.

### Provenance evidence

```
=== PC/channel/Cav2p1_MA24_PC.mod
TITLE P-type calcium channel
COMMENT
Reference: Swensen AM and Bean BP (2005) Robustness of burst firing in dissociated purkinje neurons with acute or long-term reductions in sodium conductance. J Neurosci 25:3509-20
Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Sungho Hong, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2009.
ENDCOMMENT

=== PC/channel/Cav3p1_MA24_PC.mod
TITLE Low threshold calcium current Cerebellum Purkinje Cell Model
COMMENT
Kinetics adapted to fit the Cav3.1 Iftinca et al 2006, Temperature dependence of T-type Calcium channel gating, NEUROSCIENCE
Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.
ENDCOMMENT

=== PC/channel/Cav3p2_MA24_PC.mod
TITLE Low threshold calcium current
:   Model of Huguenard & McCormick, J Neurophysiol 68: 1373-1383, 1992.
:   Written by Alain Destexhe, Salk Institute, Sept 18, 1992
:    - see Vitko et al., J. Neurosci 25(19) :4844-4855, 2005

=== PC/channel/Cav3p3_MA24_PC.mod
TITLE CaV 3.3 CA3 hippocampal neuron
COMMENT
    Xu J, Clancy CE (2008) Ionic mechanisms of endogenous bursting in CA3 hippocampal pyramidal neurons:
        a model study. PLoS ONE 3:e2056- [PubMed]
ENDCOMMENT

=== PC/channel/HCN1_MA24_PC.mod
TITLE I-h HCN1 channel from Kamilla Angelo, Michael London,Soren R. Christensen, and Michael Hausser 2007 J. of Neurosci.
COMMENT
We call it HCN1 as PC express only HCN1 Santoro et al. 2000
ENDCOMMENT

=== PC/channel/Kca1p1_MA24_PC.mod
TITLE Large conductance Ca2+ activated K+ channel mslo
COMMENT
Parameters from Cox et al. (1987) J Gen Physiol 110:257-81 (patch 1).
Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Sungho Hong, Okinawa Institute of Science and Technology, March 2009.
ENDCOMMENT

=== PC/channel/Kca2p2_MA24_PC.mod
TITLE SK2 multi-state model Cerebellum Golgi Cell Model
COMMENT
Author:Sergio Solinas, Lia Forti, Egidio DAngelo
Based on data from: Hirschberg, Maylie, Adelman, Marrion J Gen Physiol 1998
Last revised: May 2007
             Jonathan Mapelli, Erik De Schutter and Egidio D`Angelo (2008)
ENDCOMMENT

=== PC/channel/Kca3p1_MA24_PC.mod
TITLE Calcium dependent potassium channel
: Implemented in Rubin and Cleland (2006) J Neurophysiology
: Parameters from Bhalla and Bower (1993) J Neurophysiology
:   by Andrew Davison, The Babraham Institute  [Brain Res Bulletin, 2000]

=== PC/channel/Kir2p3_MA24_PC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Reference: Theta-Frequency Bursting and Resonance in Cerebellar Granule Cells:Experimental
ENDCOMMENT

=== PC/channel/Kv1p1_MA24_PC.mod
TITLE Voltage-gated low threshold potassium current from Kv1 subunits
COMMENT
Human Kv1.1 expressed in xenopus oocytes: Zerr et al., J Neurosci 18, 2842, 2848, 1998
Reference: Akemann et al., Biophys. J. (2009) 96: 3959-3976

=== PC/channel/Kv1p5_MA24_PC.mod
TITLE Cardiac IKur  current & nonspec cation current with identical kinetics
: Hodgkin - Huxley type channels, modified to fit IKur data from Feng et al Am J Physiol 1998 275:H1717 - H 1725
	 gKur=0.13195e-3 (S/cm2) <0,1e9>

=== PC/channel/Kv3p3_MA24_PC.mod
TITLE Voltage-gated potassium channel from Kv3 subunits
COMMENT
Values derive from least-square fits to experimental data of G/Gmax(v) and taun(v) in Martina et al. J Neurophys. 97 (563-671, 2007.
Reference: Akemann et al., Biophys. J. (2009) 96: 3959-3976
Date of Implementation: April 2007

=== PC/channel/Kv3p4_MA24_PC.mod
(no TITLE/COMMENT/Author/Ref/revis/year lines in first 25 lines)

=== PC/channel/Kv4p3_MA24_PC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Author: E.D'Angelo, T.Nieus, A. Fontana
	Last revised: Egidio 3.12.2003
ENDCOMMENT

=== PC/channel/Nav1p6_MA24_PC.mod
TITLE resurgent sodium channel
COMMENT
Based om updated kinetic parameters from Raman and Bean, Biophys.J. 80 (2001) 729
Modified from Khaliq et al., J.Neurosci. 23(2003)4899
Reference: Akemann and Knoepfel, J.Neurosci. 26 (2006) 4602
Date of Implementation: May 2005
ENDCOMMENT

=== PC/ion/CdpCAM_MA24_PC.mod
COMMENT
1) Extended using parameters from Schmidt et al. 2003.
2) Pump rate was tuned according to data from Maeda et al. 1999
Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.
ENDCOMMENT
```

**Inconsistent-author cases in this bucket:** `Kca2p2_MA24_PC.mod`
("Author:Sergio Solinas, Lia Forti, Egidio DAngelo") and `Kv4p3_MA24_PC.mod`
("Author: E.D'Angelo, T.Nieus, A. Fontana") both name an author unrelated
to the `MA2024`/Masoli-family search target — same pattern as the `MA2020`
bucket (these two mechanisms carry the identical header text copy-pasted
across all cell-type ports).

### Verified record

Confirmed 2026-08-15 against PubMed (`efetch`), the Crossref REST API,
the PMC article HTML (PMC10761885, 299 KB retrieved), and the ModelDB
REST API. The Task 1 hypothesis is confirmed unchanged. ModelDB
accession 267694, which the paper's own Code Availability statement
gives as `https://modeldb.science/267694`.

.. [1] Masoli, S., Sanchez-Ponce, D., Vrieler, N., Abu-Haya, K.,
       Lerner, V., Shahar, T., Nedelescu, H., Rizza, M. F.,
       Benavides-Piccione, R., DeFelipe, J., Yarom, Y., Munoz, A., &
       D'Angelo, E. (2024). Human Purkinje cells outperform mouse
       Purkinje cells in dendritic complexity and computational
       capacity. Communications Biology, 7(1), 5.
       doi:10.1038/s42003-023-05689-y

PMID 38168772, PMCID PMC10761885, fully open access. All thirteen
authors are listed in the published order, confirmed identically by
PubMed and Crossref. `5` is the article number and occupies the page
slot; *Communications Biology* assigns every article of a volume-year
to issue 1, so `7(1)` is nominal, not a real issue.

**Name-form note.** In this paper the second author is published
"Sanchez-Ponce" and the twelfth "Munoz", without accents, in both
PubMed and Crossref -- whereas the *same two people* appear as
"Sanchez-Ponce"/"Munoz" here but with accents in the `RI2021` record.
That inconsistency is real and inter-paper, not a transcription error;
each record reproduces its own publisher's form.

### Attribution

**Attribution check: PASSED for all 19 symbols**, with the `Kv3p4`
caveat from the `MA2020` block carried over.

Method as documented in the `MA2020` `### Attribution` block: the
exhaustive literal set-difference of every BrainCell class against its
`PC/channel/*_MA24_PC.mod` or `PC/ion/*_MA24_PC.mod` counterpart, with
the same seven benign-difference classes. All 19 map 1:1 onto a `.mod`
file and every rate-function constant matches. The zero-overlap
classes in that scan (`Cav2p1_MA2024_PC`, `Cav3p1_MA2024_PC`,
`Cav3p2_MA2024_PC`, `Cav3p3_MA2024_PC`, `Kca1p1_MA2024_PC`,
`Kca2p2_MA2024_PC`, `Kca3p1_MA2024_PC`, `Nav1p6_MA2024_PC`) are
subclasses that inherit their constants -- e.g. `Cav2p1_MA2024_PC`
derives from `Cav2p1_RI2021_SC`, `Kca3p1_MA2024_PC` from
`Kca3p1_MA2020_GoC` -- confirmed by reading each `class` line.

**Mapping table.** `.. [1]` is the origin record; `.. [2]` is the
`MA2024` entry above.

| Symbols | Origin `.. [1]` |
|---|---|
| `Cav2p1_MA2024_PC`, `Cav2p1_MA2024_PC_Frozen` | O-SW2005 (recordings), O-AN2012 (model) |
| `Cav3p1_MA2024_PC`, `Cav3p1_MA2024_PC_Frozen` | O-IF2006 (kinetics), O-AN2012 (model) |
| `Cav3p2_MA2024_PC` | Huguenard & McCormick (1992) -- see the `HM1992` Verified record -- plus O-VI2005 and O-CO1989 (Q10) |
| `Cav3p3_MA2024_PC`, `Cav3p3_MA2024_PC_Frozen` | O-XC2008 |
| `HCN1_MA2024_PC` | O-AG2007 (kinetics), O-SA2000 (subunit identity) |
| `Kir2p3_MA2024_PC`, `Kv4p3_MA2024_PC` | O-DA2001 |
| `Kv1p1_MA2024_PC` | O-ZE1998 (data), O-AK2009 (model) |
| `Kv1p5_MA2024_PC` | O-FE1998 |
| `Kv3p3_MA2024_PC` | O-MT2007 (fits), O-AK2009 (model) |
| `Kv3p4_MA2024_PC` | O-KH2003 |
| `Kca1p1_MA2024_PC` | O-CX1997 (parameters), O-AN2012 (model) |
| `Kca2p2_MA2024_PC` | O-HI1998 (data), O-SO2007a (model) |
| `Kca3p1_MA2024_PC` | O-RC2006, O-BB1993, O-DV2000 |
| `Nav1p6_MA2024_PC` | O-RB2001 (kinetics), O-KH2003, O-AK2006 |
| `CdpCAM_MA2024_PC` | O-AN2012 (model), O-SC2003 (buffers), O-MD1999 (pump tuning) |

**Fingerprints checked beyond the literal scan.** `Cav3p1_MA24_PC.mod`
carries the named Boltzmann/tau parameter block
`v0_m_inf = -52 mV`, `v0_h_inf = -72 mV`, `k_m_inf = -5 mV`,
`k_h_inf = 7 mV`, `C_tau_m = 1`, `v0_tau_m1 = -40 mV`,
`v0_tau_m2 = -102 mV`, `k_tau_m1 = 9 mV`, `k_tau_m2 = -18 mV`,
`C_tau_h = 15`, `v0_tau_h1 = -32 mV`, `k_tau_h1 = 7 mV`, with
`qt = 3^((celsius - 37)/10)` -- Anwar's `CaT3_1.mod` from ModelDB
138382, whose header names O-IF2006 as the kinetic fit. The
`Kca1p1`/`Kca2p2` fingerprints are the same as in the `MA2020` block
(identical mechanism, different port). `CdpCAM_MA24_PC.mod` is the
`CdpStC` scaffold with the CB subnetwork enabled and both CB and CAM
states placed in the cytosolic compartment, per the README's
"Ion_dyn implementation notes" table.

**Caveats.** The `Kv3p4` subunit-naming caveat (see O-KH2003) applies
to `Kv3p4_MA2024_PC`. `Kv1p5_MA2024_PC` is the one PC mechanism whose
`.mod` file computes `ino` as a RANGE variable with its
`USEION no WRITE ino` line commented out; BrainCell converts only the
default `ik` path. `Cav3p2_MA2024_PC` shares Destexhe's 1992
implementation of the Huguenard & McCormick low-threshold current;
that key is already verified in this file and must be cited from the
`HM1992` `### Verified record` block, not retyped.

### Import deviations

Transcribed from `examples/neuron_compare/Cerebellum_mod/README.md`.

**These deviations are already applied to the `.mod` files in this
repository.** The README's status column reads `已连续化` ("now
continuous") with the former range under `原` ("formerly"), so opening
a shipped `.mod` file shows `cnexp` and no `TABLE`. That is the
deviation having been made, not evidence against it. Do not read the
shipped file as refuting a row below.

**`TABLE` removed, replaced by per-call evaluation.**

| Symbol | Former `TABLE` range | Tabulated |
|---|---|---|
| `Kv4p3_MA2024_PC` | `[-100, 30]` (clamped) | `a_inf`, `tau_a`, `b_inf`, `tau_b` |
| `Kir2p3_MA2024_PC` | `[-100, 100]` | `d_inf`, `tau_d` |
| `Kca3p1_MA2024_PC` | V `[-100, 100]`; `cai` `[0, 0.01]` (clamped) | `Yvdep`, `Yconcdep` |

**`derivimplicit` -> `cnexp`.** `Kir2p3_MA2024_PC`,
`Kv1p5_MA2024_PC`, `Kv4p3_MA2024_PC`.

**Rate-refresh relocation.** None for `PC`.

**NMODL default-precision rewrites.** `Kv4p3_MA2024_PC` only, same
four parameters and values as in the `MA2020` table:
`Kalpha_a` `-23.32708` -> `-23.3271`; `Kbeta_a` `19.47175` ->
`19.4718`; `V0beta_a` `-18.27914` -> `-18.2791`; `V0alpha_b`
`-111.33209` -> `-111.332`.

---

## SU2015  (16 symbols)

### Symbols

- `braincell/channel/calcium.py::CaHVA_SU2015_DCN`
- `braincell/channel/calcium.py::CaL_SU2015_DCN`
- `braincell/channel/calcium.py::CaLVA_SU2015_DCN`
- `braincell/channel/hyperpolarization_activated.py::HCN_SU2015_DCN`
- `braincell/channel/potassium.py::fKdr_SU2015_DCN`
- `braincell/channel/potassium.py::sKdr_SU2015_DCN`
- `braincell/channel/potassium_calcium.py::SK_SU2015_DCN`
- `braincell/channel/sodium.py::NaF_SU2015_DCN`
- `braincell/channel/sodium.py::NaP_SU2015_DCN`
- `braincell/ion/calcium.py::ToyCaBindingKinetic_SU2015_DCN`
- `braincell/ion/calcium.py::ToyCaBindingSourceKinetic_SU2015_DCN`
- `braincell/ion/calcium.py::ToyCaBindingIcaSourceKinetic_SU2015_DCN`
- `braincell/ion/calcium.py::ToyCaPumpFactorKinetic_SU2015_DCN`
- `braincell/ion/calcium.py::ToyDiamFactorKinetic_SU2015_DCN`
- `braincell/ion/calcium.py::CdpHVA_SU2015_DCN`
- `braincell/ion/calcium.py::CdpLVA_SU2015_DCN`

Mod-file year code: `SU15`. Cell type: `DCN` (deep cerebellar nuclei). All
16 symbols map 1:1 onto a `<mechanism>_SU15_DCN.mod` file. One extra `.mod`
file, `DCN/ion/ToyStoich3ABtoCKinetic_SU15_DCN.mod`, exists but is **not**
claimed by any braincell symbol — it was apparently not ported. None of
the 11 `DCN` `.mod` files harvested here carry an `Author:`/`Ref:` line at
all — their headers are bare `TITLE ... (DCN) neuron` / empty `COMMENT`
blocks, so this bucket has essentially no in-repo textual provenance
beyond the mechanism name and species (DCN neuron).

> **TWO CORRECTIONS TO THE PARAGRAPH ABOVE (Task 3, 2026-08-15).** It is
> left unedited because `### Provenance evidence` blocks are raw harvest
> output, but both of its factual claims are wrong.
>
> 1. **The file count is 17, not 11.** `DCN/channel/` holds 9 `.mod`
>    files and `DCN/ion/` holds 8, verified by `find`. Sixteen are
>    claimed by `SU2015` symbols and one
>    (`ToyStoich3ABtoCKinetic_SU15_DCN.mod`) is unclaimed. The count 11
>    is Task 1's harvest window, not the directory. (The 11 in the
>    README's "Cell totals" table is a different, correct number: it
>    counts only the shipped DCN mechanisms and excludes the six `Toy*`
>    fixtures.) `DCN/` also holds `other/` and `synapse/`
>    subdirectories with 6 further `.mod` files, outside this bucket's
>    scope.
> 2. **These files are NOT provenance-free.** Every one of the 11
>    shipped DCN mechanisms carries, inside its `COMMENT` block below
>    the harvest's 25-line window, the line:
>
>        Translated from GENESIS by Johannes Luthman and Volker Steuber.
>
>    That is a complete provenance lead, and it resolves the bucket. See
>    the `### Verified record` block below.

### Provenance evidence

```
=== DCN/channel/CaHVA_SU15_DCN.mod
TITLE High voltage activated calcium current (CaHVA) of deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/channel/CaL_SU15_DCN.mod
TITLE LVA calcium current (CaLVA) of deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/channel/CaLVA_SU15_DCN.mod
TITLE Low voltage activated calcium current (CaLVA) of deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/channel/fKdr_SU15_DCN.mod
TITLE Fast delayed rectifier (fKdr) of deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/channel/HCN_SU15_DCN.mod
TITLE h current of deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/channel/NaF_SU15_DCN.mod
TITLE Fast sodium current (NaF) of deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/channel/NaP_SU15_DCN.mod
TITLE Persistent sodium current (NaP) of deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/channel/sKdr_SU15_DCN.mod
TITLE Slow delayed rectifier (sKdr) of deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/channel/SK_SU15_DCN.mod
TITLE Small conductance calcium dependent potassium current (SK) of deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/ion/CdpHVA_SU15_DCN.mod
TITLE Intracellular calcium concentration in deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/ion/CdpLVA_SU15_DCN.mod
TITLE Intracellular calcium concentration from the CaLVA channel in deep cerebellar nucleus (DCN) neuron
COMMENT
ENDCOMMENT

=== DCN/ion/ToyCaBindingIcaSourceKinetic_SU15_DCN.mod
TITLE Minimal reversible calcium-binding kinetic toy with ica-driven source in deep cerebellar nucleus (DCN)
COMMENT
ENDCOMMENT

=== DCN/ion/ToyCaBindingKinetic_SU15_DCN.mod
TITLE Minimal reversible calcium-binding kinetic toy in deep cerebellar nucleus (DCN)
COMMENT
ENDCOMMENT

=== DCN/ion/ToyCaBindingSourceKinetic_SU15_DCN.mod
TITLE Minimal reversible calcium-binding kinetic toy with constant source in deep cerebellar nucleus (DCN)
COMMENT
ENDCOMMENT

=== DCN/ion/ToyCaPumpFactorKinetic_SU15_DCN.mod
TITLE Minimal factor-crossing calcium pump toy in deep cerebellar nucleus (DCN)
COMMENT
ENDCOMMENT

=== DCN/ion/ToyDiamFactorKinetic_SU15_DCN.mod
TITLE Minimal diameter-driven factor reaction toy in deep cerebellar nucleus (DCN)
COMMENT
ENDCOMMENT
```

No `Author:`/`Ref:`-bearing header found anywhere in this bucket — flagged
in "Unresolved attributions" below.

### Verified record

Confirmed 2026-08-15 against PubMed (`efetch`), the Crossref REST API,
the PMC article HTML (PMC4668013, 231 KB retrieved and searched), the
ModelDB REST API, and the `github.com/ModelDBRepository/185513` file
tree. ModelDB accession 185513, which the paper's own Data
Availability statement gives.

.. [1] Sudhakar, S. K., Torben-Nielsen, B., & De Schutter, E. (2015).
       Cerebellar nuclear neurons use time and rate coding to transmit
       Purkinje neuron pauses. PLOS Computational Biology, 11(12),
       e1004641.
       doi:10.1371/journal.pcbi.1004641

PMID 26630202, PMCID PMC4668013, fully open access. `e1004641` is the
article number and occupies the page slot.

**Title case: down-cased deliberately -- do not "fix" it back.**
*PLOS Computational Biology* prints this title in title case, and both
Crossref and PubMed return it that way: "Cerebellar Nuclear Neurons
Use Time and Rate Coding to Transmit Purkinje Neuron Pauses". The
entry above is in sentence case because that is the house rule (see
`## Citation house style`, "Title: sentence case"), under which
down-casing a title-case journal is the normal operation, not an
exception. "Cerebellar" and "Purkinje" are kept capitalised as a
sentence opener and a proper noun respectively. **The wording is
untouched.** Anyone re-auditing this record against Crossref will see
a case mismatch; it is intentional and is not a transcription error.

**Author-list correction to the Task 1 brief.** The hypothesised
"Sudhakar, Hong, Raikov, ... De Schutter" is a *different* paper --
Sudhakar, Hong, Raikov, Publio, Lang, Close, Guo, Negrello & De
Schutter (2017), *Spatiotemporal network coding of physiological mossy
fiber inputs by the cerebellar granular layer*, PLOS Computational
Biology 13(9), e1005754, PMID 28934196 (record independently confirmed
here, and **not** the source of anything in this bucket). The 2015 DCN
paper has three authors.

**The origin of the DCN kinetics is Steuber et al. (2011), reached
through Luthman et al. (2011).** Task 1 recorded this bucket as having
no in-repo provenance; that was an artefact of its harvest pattern,
which matched on `Author`/`Ref`-style keywords and so never saw a
credit phrased as free text (see `## Unresolved attributions` item 8).
**9 of the 11 shipped DCN `.mod` files carry "Translated from GENESIS
by Johannes Luthman and Volker Steuber." inside their `COMMENT`
block** -- every `DCN/channel/` file except `CaLVA_SU15_DCN.mod`, plus
`DCN/ion/CdpHVA_SU15_DCN.mod`. The two that do not,
`CaLVA_SU15_DCN.mod` and `CdpLVA_SU15_DCN.mod`, carry a `COMMENT`
about their shared GHK coupling and name no author at all; they are
part of the same deposit and the chain below is unaffected by their
silence. Three steps, each verified:

1. **O-ST2011** is the original GENESIS DCN model (ModelDB 136175) and
   the origin of the channel kinetics. `SU2015`'s own Methods cite it
   verbatim: "The CN neuron model used in our study is based on a
   previously published model [21]", where [21] is Steuber et al.
   2011, repeated throughout (its m1/m2/m3 reproduce "Neuron 1/2/3 of
   [21]").
2. **O-LU2011** is the GENESIS-to-NEURON translation. Its Methods say
   verbatim that the model, "originally implemented in GENESIS", "was
   translated to NEURON to simplify the modelling of STD", citing
   Steuber et al. 2011. Corroborating detail: `GammaStim.mod`, which
   Luthman et al. describe as their custom `NetStim` variant, is
   present in the `SU2015` deposit -- so this file set demonstrably
   passed through that translation.
3. **`SU2015`** reused the NEURON translation but cites only step 1.
   **"Luthman" appears zero times in the Sudhakar et al. (2015) full
   text, reference list included.** The `.mod` files are the only
   evidence of step 2, which is why it must be recorded here rather
   than inferred from the paper.

No GENESIS ancestor earlier than O-ST2011 is cited by either paper.

**One further correction, recorded because a docstring could easily
get it wrong.** Six of the mechanism names in this bucket appear only
in the deposited code, never in the article. Searching the PMC full
text, only `CaLVA`, `NaP`, `HCN` and `SK` occur as strings; `CaHVA`,
`CaL`, `NaF`, `fKdr`, `sKdr` and the phrase "calcium pool" occur zero
times. A docstring may say the mechanism is part of the model
published as `SU2015`; it must not say the paper names or describes
that mechanism.

### Attribution

**Attribution check: PASSED for 11 of 16 symbols. The other 5 get NO
CITATION -- they are not literature-derived at all.**

**The five `Toy*` symbols are BrainCell's own test fixtures.**
`ToyCaBindingKinetic_SU2015_DCN`,
`ToyCaBindingSourceKinetic_SU2015_DCN`,
`ToyCaBindingIcaSourceKinetic_SU2015_DCN`,
`ToyCaPumpFactorKinetic_SU2015_DCN` and
`ToyDiamFactorKinetic_SU2015_DCN` carry the `SU15_DCN` suffix by naming
convention only. Their `.mod` files say so explicitly -- e.g.
`ToyCaBindingKinetic_SU15_DCN.mod`: "This deliberately minimal
mechanism exists only to validate the BrainCell `KineticIon` import
path against a small NMODL `KINETIC` example before attempting larger
DCN or GoC calcium-pool mechanisms." They model one reversible
buffering step (`cai + b <-> bc` with `CONSERVE b + bc = Btot`,
`kf = 2 /ms mM`, `kb = 0.5 /ms`, `Btot = 1 mM`) and its variants with
a constant source, an `ica`-driven source, an explicit pump
compartment, and geometry-derived factors. **They must ship no
`References` section.** Their docstrings should state that they are
import-path fixtures, not models of a published mechanism. The
unclaimed sixth file, `ToyStoich3ABtoCKinetic_SU15_DCN.mod` (`3a + b
<-> c`, "exists to validate higher-order stoichiometry handling"), is
the same thing left unported -- closing `## Unresolved attributions`
item 4.

**Mapping table for the 11 real symbols.** `.. [1]` is O-ST2011
(origin of the kinetics), `.. [2]` is the `SU2015` entry above.
O-LU2011 is the NEURON translation and belongs in `Notes`, not in
`References`, unless the docstring discusses the port itself.

| Symbols | Origin `.. [1]` |
|---|---|
| `CaHVA_SU2015_DCN`, `CaL_SU2015_DCN`, `CaLVA_SU2015_DCN`, `HCN_SU2015_DCN`, `fKdr_SU2015_DCN`, `sKdr_SU2015_DCN`, `SK_SU2015_DCN`, `NaF_SU2015_DCN`, `NaP_SU2015_DCN`, `CdpHVA_SU2015_DCN`, `CdpLVA_SU2015_DCN` | O-ST2011 (via O-LU2011) |

**Code-versus-mod check.** Method as documented in the `MA2020`
`### Attribution` block. All 11 match exhaustively; the unmatched
literals are `gbar = 1e-5 siemens/cm2` (BrainCell `g_max =
0.01 mS/cm2`, an exact match after conversion) and the GHK Kelvin
offset `273.15`. Equations confirmed term for term against the `.mod`
files, for example:

- `NaF_SU2015_DCN` (`braincell/channel/sodium.py:194`):
  `m_inf = 1/(1 + exp((V+45)/-7.3))`,
  `tau_m = 5.83/(exp((V-6.4)/-9) + exp((V+97)/17)) + 0.025`,
  `h_inf = 1/(1 + exp((V+42)/5.9))`,
  `tau_h = 16.67/(exp((V-8.3)/-29) + exp((V+66)/9)) + 0.2`, gating
  `m^3 h`, all divided by `qdeltat`.
- `NaP_SU2015_DCN`: `m_inf = 1/(1 + exp((V+70)/-4.1))`, `tau_m = 50`,
  `h_inf = 1/(1 + exp((V+80)/4))`,
  `tau_h = 1750/(1 + exp((V+65)/-8)) + 250`.
- `fKdr_SU2015_DCN`: `m_inf = 1/(1 + exp((V+40)/-7.8))`,
  `tau_m = 13.9/(exp((V+40)/12) + exp((V+40)/-13)) + 0.1`.
- `sKdr_SU2015_DCN`: `m_inf = 1/(1 + exp((V+50)/-9.1))`,
  `tau_m = 14.95/(exp((V+50)/21.74) + exp((V+50)/-13.91)) + 0.05`,
  gating `m^4`.
- `HCN_SU2015_DCN`: `m_inf = 1/(1 + exp((V+80)/5))`, and a **constant**
  `tau_m = 400/qdeltat` ms. Earlier drafts of this line read "no tau",
  meaning only that no *voltage-dependent* tau expression appears (the
  `.mod` file's `TABLE` directive tabulates `minf` alone, because
  `taum` does not depend on `V`). That wording misled a module task
  into briefing the gate as instantaneous, which it is not:
  `braincell/channel/hyperpolarization_activated.py::HCN_SU2015_DCN.f_m_tau`
  returns a real, finite `400.0 / self.qdeltat`.
- `SK_SU2015_DCN`: `z_inf = cai^4/(cai^4 + 8.1e-15)` (i.e. a Hill
  coefficient of 4 at `[Ca] = 3e-4 mM`), with the piecewise
  `tau_z = 1 - 186.67 cai` below `cai = 0.005 mM` and `0.0667` above.
- `CaHVA_SU2015_DCN` and `CaLVA_SU2015_DCN` are the only two DCN
  channels using GHK; both carry the `4.47814e6` and
  `-23.20764929` GHK constants, and `CaLVA` reads a separate `cal`
  ion so that `CdpLVA` can track it independently of the `CaHVA` pool
  that gates `SK`.

**What could NOT be established, and must not be papered over.** The
individual DCN mechanisms have no per-mechanism attribution anywhere
in this chain. O-ST2011 is the origin of the *set*, established from
the `.mod` translation credit plus `SU2015`'s own "based on a
previously published model" sentence -- **not** from a per-mechanism
statement in any paper, and **not** by comparing these constants
against numbers printed in O-ST2011, whose parameter tables were not
read. Six of the nine channel names do not appear in the `SU2015` text
at all (above). So a docstring may say: "kinetics from the deep
cerebellar nucleus model of Steuber et al. (2011), translated from
GENESIS to NEURON by Luthman et al. (2011) and used in Sudhakar et
al. (2015)". It may **not** say that any of these papers prints the
Boltzmann constants quoted above. Recorded as `## Unresolved
attributions` item 9.

### Import deviations

Transcribed from `examples/neuron_compare/Cerebellum_mod/README.md`.

**These deviations are already applied to the `.mod` files in this
repository.** The README's status column reads `已连续化` ("now
continuous") with the former range under `原` ("formerly"), so opening
a shipped `.mod` file shows `cnexp` and no `TABLE`. That is the
deviation having been made, not evidence against it. Do not read the
shipped file as refuting a row below.

**`TABLE` removed, replaced by per-call evaluation.** The DCN files
used a wider voltage window than the other cell types, so boundary
clamping was much less likely to bite; the concentration table in
`SK` is the exception.

| Symbol | Former `TABLE` range | Tabulated |
|---|---|---|
| `CaHVA_SU2015_DCN` | `[-150, 100]` | `minf`, `taum`, plus a `DEPEND T` table |
| `CaLVA_SU2015_DCN` | `[-150, 100]` | `minf`, `taum`, `hinf`, `tauh`, plus a `DEPEND T` table |
| `CaL_SU2015_DCN` | `[-150, 100]` | `minf`, `taum`, `hinf`, `tauh` |
| `HCN_SU2015_DCN` | `[-150, 100]` | `minf` |
| `NaF_SU2015_DCN` | `[-150, 100]` | `minf`, `taum`, `hinf`, `tauh` |
| `NaP_SU2015_DCN` | `[-150, 100]` | `minf`, `hinf`, `tauh` |
| `fKdr_SU2015_DCN` | `[-150, 100]` | `minf`, `taum` |
| `sKdr_SU2015_DCN` | `[-150, 100]` | `minf`, `taum` |
| `SK_SU2015_DCN` | `cai` `[0, 0.01]` (clamped) | `zinf`, `tauz` |

**`derivimplicit` -> `cnexp`.** None. `DCN` does not appear in the
README's integration-method table; these mechanisms were already
`cnexp` (or `sparse`, for the kinetic pools) upstream.

**Rate-refresh relocation.** None for `DCN`.

**NMODL default-precision rewrites -- one documented EXCEPTION.**
`ToyDiamFactorKinetic_SU2015_DCN`'s `pump_area` and `cyto` are
`62.83185307179586` in the `.mod` source and `62.8319` in NEURON's
compiled output. BrainCell does **not** substitute the compiled
constant: it derives both at runtime from geometry, as
`pi * diam_mid` and `pi * diam_mid * depth`. The README marks this
`例外` (exception). No other DCN mechanism is affected. (This entry is
recorded for completeness; the symbol it concerns is one of the five
citation-less `Toy*` fixtures.)

---

## MA2025  (16 symbols)

### Symbols

- `braincell/channel/calcium.py::Cav1p2_MA2025_BC`
- `braincell/channel/calcium.py::Cav1p3_MA2025_BC`
- `braincell/channel/calcium.py::Cav2p1_MA2025_BC`
- `braincell/channel/calcium.py::Cav2p1_MA2025_BC_Frozen`
- `braincell/channel/calcium.py::Cav3p2_MA2025_BC`
- `braincell/channel/hyperpolarization_activated.py::HCN1_MA2025_BC`
- `braincell/channel/potassium.py::Kir2p3_MA2025_BC`
- `braincell/channel/potassium.py::Kv1p1_MA2025_BC`
- `braincell/channel/potassium.py::Kv3p4_MA2025_BC`
- `braincell/channel/potassium.py::Kv4p3_MA2025_BC`
- `braincell/channel/potassium_calcium.py::Kca3p1_MA2025_BC`
- `braincell/channel/potassium_calcium.py::Kca2p2_MA2025_BC`
- `braincell/channel/potassium_calcium.py::Kca1p1_MA2025_BC`
- `braincell/channel/sodium.py::Nav1p6_MA2025_BC`
- `braincell/channel/sodium.py::Nav1p1_MA2025_BC`
- `braincell/ion/calcium.py::CdpStC_MA2025_BC`

Mod-file year code: `MA25`. Cell type: `BC` (basket cell). All 16 symbols
map 1:1 onto a `<mechanism>_MA25_BC.mod` file.

### Provenance evidence

```
=== BC/channel/Cav1p2_MA25_BC.mod
: model from Evans et al 2013, transferred from GENESIS to NEURON by Beining et al (2016), "A novel comprehensive and consistent electrophysiologcal model of dentate granule cells"

=== BC/channel/Cav1p3_MA25_BC.mod
: model from Evans et al 2013, transferred from GENESIS to NEURON by Beining et al (2016), "A novel comprehensive and consistent electrophysiologcal model of dentate granule cells"

=== BC/channel/Cav2p1_MA25_BC.mod
TITLE P-type calcium channel
COMMENT
Reference: Swensen AM and Bean BP (2005) Robustness of burst firing in dissociated purkinje neurons with acute or long-term reductions in sodium conductance. J Neurosci 25:3509-20
Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Sungho Hong, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2009.
ENDCOMMENT

=== BC/channel/Cav3p2_MA25_BC.mod
TITLE Low threshold calcium current
:   Model of Huguenard & McCormick, J Neurophysiol 68: 1373-1383, 1992.
:   Written by Alain Destexhe, Salk Institute, Sept 18, 1992
:    - see Vitko et al., J. Neurosci 25(19) :4844-4855, 2005

=== BC/channel/HCN1_MA25_BC.mod
TITLE I-h HCN1 channel from Kamilla Angelo, Michael London,Soren R. Christensen, and Michael Hausser 2007 J. of Neurosci.
COMMENT
We call it HCN1 as PC express only HCN1 Santoro et al. 2000
ENDCOMMENT

=== BC/channel/Kca1p1_MA25_BC.mod
TITLE Large conductance Ca2+ activated K+ channel mslo
COMMENT
Parameters from Cox et al. (1987) J Gen Physiol 110:257-81 (patch 1).
Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Sungho Hong, Okinawa Institute of Science and Technology, March 2009.
ENDCOMMENT

=== BC/channel/Kca2p2_MA25_BC.mod
TITLE SK2 multi-state model Cerebellum Golgi Cell Model
COMMENT
Author:Sergio Solinas, Lia Forti, Egidio DAngelo
Based on data from: Hirschberg, Maylie, Adelman, Marrion J Gen Physiol 1998
Last revised: May 2007
             Jonathan Mapelli, Erik De Schutter and Egidio D`Angelo (2008)
ENDCOMMENT

=== BC/channel/Kca3p1_MA25_BC.mod
TITLE Calcium dependent potassium channel
: Implemented in Rubin and Cleland (2006) J Neurophysiology
: Parameters from Bhalla and Bower (1993) J Neurophysiology
:   by Andrew Davison, The Babraham Institute  [Brain Res Bulletin, 2000]

=== BC/channel/Kir2p3_MA25_BC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Reference: Theta-Frequency Bursting and Resonance in Cerebellar Granule Cells:Experimental
ENDCOMMENT

=== BC/channel/Kv1p1_MA25_BC.mod
TITLE Voltage-gated low threshold potassium current from Kv1 subunits
COMMENT
Human Kv1.1 expressed in xenopus oocytes: Zerr et al., J Neurosci 18, 2842, 2848, 1998
Reference: Akemann et al., Biophys. J. (2009) 96: 3959-3976

=== BC/channel/Kv3p4_MA25_BC.mod
(no TITLE/COMMENT/Author/Ref/revis/year lines in first 25 lines)

=== BC/channel/Kv4p3_MA25_BC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Author: E.D'Angelo, T.Nieus, A. Fontana
	Last revised: Egidio 3.12.2003
ENDCOMMENT

=== BC/channel/Nav1p1_MA25_BC.mod
TITLE Non-resurgent sodium channel in Purkinje cells
COMMENT
This channel was derived from the Narsg channel of Khaliq et al., J. Neurosci. 23(2003)4899
Reference: Akemann et al. Biophys. J. (2009) 96: 3959-3976
Date of Implementation: April 2007
ENDCOMMENT

=== BC/channel/Nav1p6_MA25_BC.mod
TITLE resurgent sodium channel
COMMENT
Based om updated kinetic parameters from Raman and Bean, Biophys.J. 80 (2001) 729
Modified from Khaliq et al., J.Neurosci. 23(2003)4899
Reference: Akemann and Knoepfel, J.Neurosci. 26 (2006) 4602
Date of Implementation: May 2005
ENDCOMMENT

=== BC/ion/CdpStC_MA25_BC.mod
COMMENT
1) Extended using parameters from Schmidt et al. 2003.
2) Pump rate was tuned according to data from Maeda et al. 1999
Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.
ENDCOMMENT
```

**Inconsistent-author cases in this bucket:** `Kca2p2_MA25_BC.mod`
("Author:Sergio Solinas, Lia Forti, Egidio DAngelo") and `Kv4p3_MA25_BC.mod`
("Author: E.D'Angelo, T.Nieus, A. Fontana") — same header text
copy-pasted from the `MA2020`/`MA2024` ports, again naming an author
unrelated to the `MA2025` search target.

### Verified record

Confirmed 2026-08-15 against PubMed (`efetch`), the Crossref REST API,
the PMC article HTML (PMC12255734, 275 KB retrieved), and the ModelDB
REST API. ModelDB accession 2018018, given in the paper's own Data
Availability statement. **It is not a preprint** -- the Task 1 brief
allowed for that possibility; the paper is peer-reviewed and appeared
12 July 2025.

.. [1] Masoli, S., Rizza, M. F., Soda, T., Sanchez-Ponce, D., Munoz,
       A., Prestori, F., & D'Angelo, E. (2025). Cerebellar basket cell
       filtering of Purkinje cell responses elicited by low frequency
       parallel fibre transmission. Scientific Reports, 15(1), 25192.
       doi:10.1038/s41598-025-09964-2

PMID 40652073, PMCID PMC12255734, fully open access. `25192` is the
article number and occupies the page slot; *Scientific Reports*
assigns every article of a volume-year to issue 1, so `15(1)` is
nominal. Keep the British spelling **"fibre"** as published.

**Name-form note, corrected.** An earlier revision of this record
claimed that, as in `MA2024`, both "Sanchez-Ponce" and "Munoz" are
published here without accents. That is **half wrong**, and the wrong
half is the fourth author. Re-checked 2026-08-15 against both sources:
Crossref returns `family: "Sánchez-Ponce"` and PubMed's `efetch`
returns `<LastName>S&#xe1;nchez-Ponce</LastName>` -- i.e. **this paper
prints Sánchez-Ponce with the accent.** Only the fifth author, Munoz,
is unaccented here (both sources agree). The `.. [1]` entry above uses
the ASCII form for both, deliberately and for the same reason as
`Lüthi`, `Knöpfel`, `Häusser` and `Schürmann` elsewhere in this file:
the entry text is kept ASCII. **The ASCII "Sanchez-Ponce" above is a
transliteration, not a reproduction of the publisher's form.** If the
docstring pipeline is UTF-8-safe, write "Sánchez-Ponce, D." here and
leave "Munoz, A." plain.

### Attribution

**Attribution check: PASSED for all 16 symbols**, with the `Kv3p4` and
`Cav1p2`/`Cav1p3` caveats from the `MA2020` block carried over.

Method as documented in the `MA2020` `### Attribution` block. All 16
map 1:1 onto a `BC/channel/*_MA25_BC.mod` or `BC/ion/*_MA25_BC.mod`
file and every rate-function constant matches. The zero-overlap
classes in that scan (`Cav1p2_MA2025_BC`, `Cav1p3_MA2025_BC`,
`Cav2p1_MA2025_BC`, `Cav2p1_MA2025_BC_Frozen`, `Cav3p2_MA2025_BC`,
`Kca1p1_MA2025_BC`, `Kca2p2_MA2025_BC`, `Kca3p1_MA2025_BC`,
`Nav1p6_MA2025_BC`, `CdpStC_MA2025_BC`) are subclasses inheriting
their constants from a `MA2020`, `MA2024` or `RI2021` base --
confirmed by reading each `class` line. `CdpStC_MA2025_BC` derives
from `CdpStC_NoCAM_MA2020_GoC`, and that base's literal set was
re-checked directly against `BC/ion/CdpStC_MA25_BC.mod`: 21 of 26
literals match, the five unmatched being unit conversions and NMODL
range annotations.

**Mapping table.** `.. [1]` is the origin record; `.. [2]` is the
`MA2025` entry above.

| Symbols | Origin `.. [1]` |
|---|---|
| `Cav1p2_MA2025_BC`, `Cav1p3_MA2025_BC` | O-EV2013 (kinetics), O-BE2017 (port) |
| `Cav2p1_MA2025_BC`, `Cav2p1_MA2025_BC_Frozen` | O-SW2005 (recordings), O-AN2012 (model) |
| `Cav3p2_MA2025_BC` | Huguenard & McCormick (1992) -- see the `HM1992` Verified record -- plus O-VI2005 and O-CO1989 (Q10) |
| `HCN1_MA2025_BC` | O-AG2007 (kinetics), O-SA2000 (subunit identity) |
| `Kir2p3_MA2025_BC`, `Kv4p3_MA2025_BC` | O-DA2001 |
| `Kv1p1_MA2025_BC` | O-ZE1998 (data), O-AK2009 (model) |
| `Kv3p4_MA2025_BC` | O-KH2003 |
| `Kca1p1_MA2025_BC` | O-CX1997 (parameters), O-AN2012 (model) |
| `Kca2p2_MA2025_BC` | O-HI1998 (data), O-SO2007a (model) |
| `Kca3p1_MA2025_BC` | O-RC2006, O-BB1993, O-DV2000 |
| `Nav1p1_MA2025_BC` | O-KH2003 (`Narsg` derivation), O-AK2009 |
| `Nav1p6_MA2025_BC` | O-RB2001 (kinetics), O-KH2003, O-AK2006 |
| `CdpStC_MA2025_BC` | O-AN2012 (model), O-SC2003 (buffers), O-MD1999 (pump tuning) |

**Caveat specific to this bucket.** `HCN1_MA25_BC.mod` inherits the
Purkinje-cell comment "We call it HCN1 as PC express only HCN1 Santoro
et al. 2000" verbatim from the `MA2024` port, and sets
`rec_temp = 23 (deg)` with the header note that Angelo et al. "forogot
tp mention the recording temperature" (typos in the original). A
basket-cell docstring must not repeat the Purkinje-only claim as
though it were about basket cells; the 23 degC is the porter's
assumption, not a value from O-AG2007. This applies equally to
`HCN1_MA2024_PC` and `HCN1_RI2021_SC`.

`CdpStC_MA25_BC.mod` is the GoC `CdpStC` mechanism with its CAM block
commented out, per the README's "Ion_dyn inherited variants" table --
the same non-CAM pump/PV network as `CdpStC_RI2021_SC`.

### Import deviations

Transcribed from `examples/neuron_compare/Cerebellum_mod/README.md`.

**These deviations are already applied to the `.mod` files in this
repository.** The README's status column reads `已连续化` ("now
continuous") with the former range under `原` ("formerly"), so opening
a shipped `.mod` file shows `cnexp` and no `TABLE`. That is the
deviation having been made, not evidence against it. Do not read the
shipped file as refuting a row below.

**`TABLE` removed, replaced by per-call evaluation.**

| Symbol | Former `TABLE` range | Tabulated |
|---|---|---|
| `Kv4p3_MA2025_BC` | `[-100, 30]` (clamped) | `a_inf`, `tau_a`, `b_inf`, `tau_b` |
| `Kir2p3_MA2025_BC` | `[-100, 100]` | `d_inf`, `tau_d` |
| `Kca3p1_MA2025_BC` | V `[-100, 100]`; `cai` `[0, 0.01]` (clamped) | `Yvdep`, `Yconcdep` |

**`derivimplicit` -> `cnexp`.** `Kir2p3_MA2025_BC`,
`Kv4p3_MA2025_BC`.

**Rate-refresh relocation.** `Cav1p2_MA2025_BC` and
`Cav1p3_MA2025_BC`: `rates()` moved from `BREAKPOINT` into
`DERIVATIVE state`, so `inf`/`tau` are refreshed before the `cnexp`
state update.

**NMODL default-precision rewrites.** `Kv4p3_MA2025_BC` only, same
four parameters and values as in the `MA2020` table.

---

## RI2021  (15 symbols)

### Symbols

- `braincell/channel/calcium.py::Cav2p1_RI2021_SC`
- `braincell/channel/calcium.py::Cav2p1_RI2021_SC_Frozen`
- `braincell/channel/calcium.py::Cav3p2_RI2021_SC`
- `braincell/channel/calcium.py::Cav3p3_RI2021_SC`
- `braincell/channel/hyperpolarization_activated.py::HCN1_RI2021_SC`
- `braincell/channel/potassium.py::KM_RI2021_SC`
- `braincell/channel/potassium.py::Kir2p3_RI2021_SC`
- `braincell/channel/potassium.py::Kv1p1_RI2021_SC`
- `braincell/channel/potassium.py::Kv3p4_RI2021_SC`
- `braincell/channel/potassium.py::Kv4p3_RI2021_SC`
- `braincell/channel/potassium_calcium.py::Kca2p2_RI2021_SC`
- `braincell/channel/potassium_calcium.py::Kca1p1_RI2021_SC`
- `braincell/channel/sodium.py::Nav1p6_RI2021_SC`
- `braincell/channel/sodium.py::Nav1p1_RI2021_SC`
- `braincell/ion/calcium.py::CdpStC_RI2021_SC`

Mod-file year code: `RI21`. Cell type: `SC` (stellate cell). All 15
symbols map 1:1 onto a `<mechanism>_RI21_SC.mod` file.

### Provenance evidence

```
=== SC/channel/Cav2p1_RI21_SC.mod
TITLE P-type calcium channel
COMMENT
Reference: Swensen AM and Bean BP (2005) Robustness of burst firing in dissociated purkinje neurons with acute or long-term reductions in sodium conductance. J Neurosci 25:3509-20
Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Sungho Hong, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2009.
ENDCOMMENT

=== SC/channel/Cav3p2_RI21_SC.mod
TITLE Low threshold calcium current
:   Model of Huguenard & McCormick, J Neurophysiol 68: 1373-1383, 1992.
:   Written by Alain Destexhe, Salk Institute, Sept 18, 1992
:    - see Vitko et al., J. Neurosci 25(19) :4844-4855, 2005

=== SC/channel/Cav3p3_RI21_SC.mod
TITLE CaV 3.3 CA3 hippocampal neuron
COMMENT
    Xu J, Clancy CE (2008) Ionic mechanisms of endogenous bursting in CA3 hippocampal pyramidal neurons:
        a model study. PLoS ONE 3:e2056- [PubMed]
ENDCOMMENT

=== SC/channel/HCN1_RI21_SC.mod
TITLE I-h HCN1 channel from Kamilla Angelo, Michael London,Soren R. Christensen, and Michael Hausser 2007 J. of Neurosci.
COMMENT
We call it HCN1 as PC express only HCN1 Santoro et al. 2000
ENDCOMMENT

=== SC/channel/Kca1p1_RI21_SC.mod
TITLE Large conductance Ca2+ activated K+ channel mslo
COMMENT
Parameters from Cox et al. (1987) J Gen Physiol 110:257-81 (patch 1).
Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Sungho Hong, Okinawa Institute of Science and Technology, March 2009.
ENDCOMMENT

=== SC/channel/Kca2p2_RI21_SC.mod
TITLE SK2 multi-state model Cerebellum Golgi Cell Model
COMMENT
Author:Sergio Solinas, Lia Forti, Egidio DAngelo
Based on data from: Hirschberg, Maylie, Adelman, Marrion J Gen Physiol 1998
Last revised: May 2007
             Jonathan Mapelli, Erik De Schutter and Egidio D`Angelo (2008)
ENDCOMMENT

=== SC/channel/Kir2p3_RI21_SC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Reference: Theta-Frequency Bursting and Resonance in Cerebellar Granule Cells:Experimental
ENDCOMMENT

=== SC/channel/KM_RI21_SC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Author: A. Fontana
	CoAuthor: T.Nieus Last revised: 20.11.99
ENDCOMMENT

=== SC/channel/Kv1p1_RI21_SC.mod
TITLE Voltage-gated low threshold potassium current from Kv1 subunits
COMMENT
Human Kv1.1 expressed in xenopus oocytes: Zerr et al., J Neurosci 18, 2842, 2848, 1998
Reference: Akemann et al., Biophys. J. (2009) 96: 3959-3976

=== SC/channel/Kv3p4_RI21_SC.mod
(no TITLE/COMMENT/Author/Ref/revis/year lines in first 25 lines)

=== SC/channel/Kv4p3_RI21_SC.mod
TITLE Cerebellum Granule Cell Model
COMMENT
	Author: E.D'Angelo, T.Nieus, A. Fontana
	Last revised: Egidio 3.12.2003
ENDCOMMENT

=== SC/channel/Nav1p1_RI21_SC.mod
TITLE Non-resurgent sodium channel in Purkinje cells
COMMENT
This channel was derived from the Narsg channel of Khaliq et al., J. Neurosci. 23(2003)4899
Reference: Akemann et al. Biophys. J. (2009) 96: 3959-3976
Date of Implementation: April 2007
ENDCOMMENT

=== SC/channel/Nav1p6_RI21_SC.mod
TITLE resurgent sodium channel
COMMENT
Based om updated kinetic parameters from Raman and Bean, Biophys.J. 80 (2001) 729
Modified from Khaliq et al., J.Neurosci. 23(2003)4899
Reference: Akemann and Knoepfel, J.Neurosci. 26 (2006) 4602
Date of Implementation: May 2005
ENDCOMMENT

=== SC/ion/CdpStC_RI21_SC.mod
COMMENT
1) Extended using parameters from Schmidt et al. 2003.
2) Pump rate was tuned according to data from Maeda et al. 1999
Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*
PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513
Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.
ENDCOMMENT
```

**Inconsistent-author cases in this bucket:** `Kca2p2_RI21_SC.mod`
("Author:Sergio Solinas, Lia Forti, Egidio DAngelo") and `KM_RI21_SC.mod`
/ `Kv4p3_RI21_SC.mod` ("Author: A. Fontana" / "CoAuthor: T.Nieus" and
"Author: E.D'Angelo, T.Nieus, A. Fontana" respectively) — same pattern as
the other cerebellar buckets.

### Verified record

Confirmed 2026-08-15 against PubMed (`efetch`), the Crossref REST API,
the PMC article HTML (PMC7886897, 279 KB retrieved), and the ModelDB
REST API. The Task 1 hypothesis is confirmed as to authorship but
**not** as to journal: this is *Scientific Reports*, not
*Communications Biology*.

.. [1] Rizza, M. F., Locatelli, F., Masoli, S., Sanchez-Ponce, D.,
       Munoz, A., Prestori, F., & D'Angelo, E. (2021). Stellate cell
       computational modeling predicts signal filtering in the
       molecular layer circuit of cerebellum. Scientific Reports,
       11(1), 3873.
       doi:10.1038/s41598-021-83209-w

PMID 33594118, PMCID PMC7886897, fully open access. `3873` is the
article number and occupies the page slot; `11(1)` is nominal.

**Name-form note.** In *this* paper the fourth and fifth authors are
published **with** accents -- Sánchez-Ponce and Muñoz -- in both
PubMed and Crossref. The ASCII forms are used above for consistency
with the rest of this file; if the docstring pipeline is UTF-8-safe,
prefer the accented forms here.

**The cross-record picture, corrected.** An earlier revision of this
note told later tasks that "the same two appear unaccented in the
`MA2024` and `MA2025` records" and to prefer the plain forms there.
That is wrong for `MA2025`. All three were re-checked against Crossref
and PubMed on 2026-08-15 and they genuinely differ paper by paper:

| Record | Fourth/second author | Muñoz |
|---|---|---|
| `RI2021` (this record) | **Sánchez-Ponce** (accented) | **Muñoz** (accented) |
| `MA2025` | **Sánchez-Ponce** (accented) | Munoz (plain) |
| `MA2024` | Sanchez-Ponce (plain) | Munoz (plain) |

So only `MA2024` prints both plain. Every `.. [N]` entry in this file
uses the ASCII form throughout, so no entry text changes; what changes
is the instruction to a UTF-8-safe pipeline. **Sánchez-Ponce takes the
accent in `RI2021` *and* `MA2025`, and only Muñoz varies between
`RI2021` and the other two.** The inter-paper inconsistency is real
and each record reproduces its own publisher's form -- see also the
`MA2024` name-form note, which is correct as written.

**ModelDB provenance quirk, recorded rather than smoothed over.**
Accession **2018019**. The 2021 paper itself gives no accession -- it
says only that the models "will also be uploaded on ModelDB". The
accession is established from two other places: the ModelDB API
record, and the `MA2025` paper's Data Availability statement, which
states "SC model is available on ModelDB
(https://modeldb.science/2018019)". ModelDB shows it created
2024-11-18, i.e. deposited three years after publication. **It is
therefore not established that this repository's `RI21_SC` `.mod`
files came from accession 2018019 specifically**; an earlier copy
could have circulated via the HBP Brain Simulation Platform. This does
not affect the citation, only any claim about the download source.

### Attribution

**Attribution check: PASSED for all 15 symbols**, with the `Kv3p4` and
`HCN1` caveats carried over from the `MA2020` and `MA2025` blocks.

Method as documented in the `MA2020` `### Attribution` block. All 15
map 1:1 onto an `SC/channel/*_RI21_SC.mod` or `SC/ion/*_RI21_SC.mod`
file and every rate-function constant matches. `Cav2p1_RI2021_SC` and
`Cav3p2_RI2021_SC` and `Cav3p3_RI2021_SC` are the *base* classes that
`MA2024` and `MA2025` inherit from, so their literal overlap with the
`.mod` files is direct and high (14/23, 21/25 and 20/27 respectively;
the remainder are GHK and Kelvin constants). The zero-overlap classes
(`Cav2p1_RI2021_SC_Frozen`, `Kca1p1_RI2021_SC`, `Kca2p2_RI2021_SC`,
`Nav1p1_RI2021_SC`, `Nav1p6_RI2021_SC`, `CdpStC_RI2021_SC`) are
subclasses inheriting their constants; `CdpStC_RI2021_SC` derives from
`CdpStC_NoCAM_MA2020_GoC`, whose literals were re-checked directly
against `SC/ion/CdpStC_RI21_SC.mod` (21 of 26).

**Mapping table.** `.. [1]` is the origin record; `.. [2]` is the
`RI2021` entry above.

| Symbols | Origin `.. [1]` |
|---|---|
| `Cav2p1_RI2021_SC`, `Cav2p1_RI2021_SC_Frozen` | O-SW2005 (recordings), O-AN2012 (model) |
| `Cav3p2_RI2021_SC` | Huguenard & McCormick (1992) -- see the `HM1992` Verified record -- plus O-VI2005 and O-CO1989 (Q10) |
| `Cav3p3_RI2021_SC` | O-XC2008 |
| `HCN1_RI2021_SC` | O-AG2007 (kinetics), O-SA2000 (subunit identity) |
| `KM_RI2021_SC`, `Kir2p3_RI2021_SC`, `Kv4p3_RI2021_SC` | O-DA2001 |
| `Kv1p1_RI2021_SC` | O-ZE1998 (data), O-AK2009 (model) |
| `Kv3p4_RI2021_SC` | O-KH2003 |
| `Kca1p1_RI2021_SC` | O-CX1997 (parameters), O-AN2012 (model) |
| `Kca2p2_RI2021_SC` | O-HI1998 (data), O-SO2007a (model) |
| `Nav1p1_RI2021_SC` | O-KH2003 (`Narsg` derivation), O-AK2009 |
| `Nav1p6_RI2021_SC` | O-RB2001 (kinetics), O-KH2003, O-AK2006 |
| `CdpStC_RI2021_SC` | O-AN2012 (model), O-SC2003 (buffers), O-MD1999 (pump tuning) |

**Caveat specific to this bucket.** `SC/ion/CdpStC_RI21_SC.mod` reads
`cao` but never uses it in its equations, per the README's "Ion_dyn
inherited variants" table; BrainCell drops the unused read. Also note
that `SC` has no `Kca3p1` mechanism, unlike `BC`, `GoC` and `PC`.

### Import deviations

Transcribed from `examples/neuron_compare/Cerebellum_mod/README.md`.

**These deviations are already applied to the `.mod` files in this
repository.** The README's status column reads `已连续化` ("now
continuous") with the former range under `原` ("formerly"), so opening
a shipped `.mod` file shows `cnexp` and no `TABLE`. That is the
deviation having been made, not evidence against it. Do not read the
shipped file as refuting a row below.

**`TABLE` removed, replaced by per-call evaluation.**

| Symbol | Former `TABLE` range | Tabulated |
|---|---|---|
| `Kv4p3_RI2021_SC` | `[-100, 30]` (clamped) | `a_inf`, `tau_a`, `b_inf`, `tau_b` |
| `KM_RI2021_SC` | `[-100, 30]` (clamped) | `n_inf`, `tau_n` |
| `Kir2p3_RI2021_SC` | `[-100, 100]` | `d_inf`, `tau_d` |

**`derivimplicit` -> `cnexp`.** `KM_RI2021_SC`, `Kir2p3_RI2021_SC`,
`Kv4p3_RI2021_SC`.

**Rate-refresh relocation.** None for `SC`.

**NMODL default-precision rewrites.** `Kv4p3_RI2021_SC` only, same
four parameters and values as in the `MA2020` table.

---

## HM1992  (7 symbols)

### Symbols

- `braincell/channel/calcium.py::CaT_HM1992`
- `braincell/channel/calcium.py::CaHT_HM1992`
- `braincell/channel/hyperpolarization_activated.py::HCN_HM1992`
- `braincell/channel/potassium.py::KA1_HM1992`
- `braincell/channel/potassium.py::KA2_HM1992`
- `braincell/channel/potassium.py::KK2A_HM1992`
- `braincell/channel/potassium.py::KK2B_HM1992`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries an
`HM1992`/`HM19` filename fragment. This is a classical/thalamic-literature
key (per the project plan, verified in Task 2, not the cerebellar NEURON
port harvested in Step 2). No repository-local provenance text exists for
this key.

### Verified record

Confirmed 2026-08-15 against PubMed (PMID 1279135, full structured
abstract retrieved via NCBI E-utilities) and the American Physiological
Society publisher record for the DOI.

.. [1] Huguenard, J. R., & McCormick, D. A. (1992). Simulation of the
       currents involved in rhythmic oscillations in thalamic relay
       neurons. Journal of Neurophysiology, 68(4), 1373-1383.
       doi:10.1152/jn.1992.68.4.1373

**Title correction.** The existing docstring at
``braincell/channel/hyperpolarization_activated.py:78-80`` ends the
title "... in thalamic relay neuron" (singular). The published title is
"... in thalamic relay neurons" (plural), per PubMed and the publisher
record. It also carries no DOI. The module task must fix both. (Do not
confuse this paper with its companion, McCormick & Huguenard, 1992,
J Neurophysiol 68(4), 1384-1400, doi:10.1152/jn.1992.68.4.1384.)

### Attribution

**Symbols:** ``CaT_HM1992``, ``CaHT_HM1992``, ``HCN_HM1992``,
``KA1_HM1992``, ``KA2_HM1992``, ``KK2A_HM1992``, ``KK2B_HM1992``.

**Attribution check: PASSED for 6 of 7 symbols; FAILED for**
``CaHT_HM1992`` **(see "Unresolved attributions" item 7).**

The published abstract (retrieved via E-utilities) enumerates exactly
four currents: "the transient, low-voltage-activated Ca2+ current (IT),
the rapidly inactivating transient K+ current (IA), the slowly
inactivating K+ current (IK2), and the hyperpolarization-activated,
mixed cationic current (Ih)". It further states that IA "was modeled by
assuming two components with different time constants of inactivation"
and that IK2 "was also modeled by assuming two components". That maps
onto the BrainCell symbols as:

- IT  -> ``CaT_HM1992`` (``braincell/channel/calcium.py:146``)
- IA  -> ``KA1_HM1992`` / ``KA2_HM1992`` (``potassium.py:197``, ``:250``)
- IK2 -> ``KK2A_HM1992`` / ``KK2B_HM1992`` (``potassium.py:303``,
  ``:352``)
- Ih  -> ``HCN_HM1992`` (``hyperpolarization_activated.py:45``)

Constants were cross-checked against ``ITGHK.mod`` (ModelDB 279),
Destexhe's NEURON implementation of this paper, headed "Model of
Huguenard & McCormick, J Neurophysiol 68: 1373-1383, 1992".
**Every equation quoted in this paragraph comes from ``ITGHK.mod``
alone** (re-read from the ModelDB GitHub mirror, 2026-08-15). It uses
``m_inf = 1/(1 + exp(-(v+shift+actshift+57)/6.2))``,
``h_inf = 1/(1 + exp((v+shift+81)/4))``,
``tau_m = (0.612 + 1/(exp(-(v+shift+actshift+132)/16.7)
+ exp((v+shift+actshift+16.8)/18.2)))/phi_m``, and a piecewise
``tau_h`` branching on ``(v+shift) < -80``:
``exp((v+shift+467)/66.6)/phi_h`` below the branch and
``(28 + exp(-(v+shift+22)/10.5))/phi_h`` above it. ``shift = 2 mV``
(screening charge at 2 mM external Ca); ``actshift`` defaults to 0.

**``IT.mod`` (ModelDB 3817) is a different model and is NOT a source
for any constant above.** An earlier revision of this block wrongly
presented the two mod files as interchangeable. They are not:

- Its header reads "Model based **on the data of** Huguenard &
  McCormick, J Neurophysiol 68: 1373-1383, 1992 **and Huguenard &
  Prince, J Neurosci. 12: 3804-3817, 1992**" -- not "Model of".
- It shares only ``m_inf`` and ``h_inf`` with ``ITGHK.mod``.
- It has **no ``tau_m`` at all**. Activation is taken at steady state
  ("activation considered at steady-state"); ``tau_m`` appears in the
  ``ASSIGNED`` block annotated "dummy variable for compatibility" and
  is never assigned a value.
- Its piecewise ``tau_h`` is **commented out**, replaced by the
  bi-exponential fit ``30.8 + (211.4 + exp((Vm+113.2)/5))/(1 +
  exp((Vm+84)/3.2))``.
- It carries a single ``q10 = 3``, applied to inactivation only.

**How the 2 mV shift is actually applied -- read this before writing
the ``CaT_HM1992`` docstring.** ``CaT_HM1992``
(``braincell/channel/calcium.py:177-195``) folds the shift into the
Boltzmann midpoints **only**: 57 -> 59 in ``f_p_inf``, 81 -> 83 in
``f_q_inf``. The time constants do **not** fold it in. ``f_p_tau``
carries bare 132 and 16.8, and ``f_q_tau`` carries bare 467 and 22
with the branch at ``V >= -80`` -- i.e. exactly ``ITGHK.mod``'s
numbers read at ``shift = 0``, not at its shipped ``shift = 2 mV``.

So the correct claim is: ``CaT_HM1992`` reproduces ``ITGHK.mod``'s
``tau_m``, its piecewise ``tau_h`` and its ``p^2 q`` gating exactly
**against a shift = 0 reading of the mod file**, while its
steady-state midpoints match the mod file **with the 2 mV shift folded
in**. Against the shipped ``shift = 2 mV`` the two taus sit 2 mV away
from the mod file. A docstring must **not** describe the kinetics as
those of the mod file "with the 2 mV screening-charge shift folded
in": that is true of the midpoints and false of the taus. ``HCN_HM1992``'s
``p_inf = 1/(1 + exp((V+75)/5.5))`` and
``tau_p = 1/(exp(-0.086 V - 14.59) + exp(0.0701 V - 1.87))`` are the
standard published Ih parameterisation and are already documented in
the class docstring.

**Caveats for the module task:**

- ``CaT_HM1992`` defaults ``q10_p = 3.55`` at 24 degC. Destexhe's
  reference implementations use ``qm = 5`` for activation (citing
  Coulter et al., J Physiol 414: 587, 1989) and ``qh = 3`` for
  inactivation. The value 3.55 could **not** be traced to the paper or
  to any reference implementation; treat it as a BrainCell/BrainPy
  default and do not attribute it to Huguenard & McCormick.
- ``CaT_HM1992`` applies a further ``V_sh = -3 mV`` on top of the
  folded 2 mV shift, so shipped defaults sit 3 mV from the mod-file
  defaults. This is a documented free parameter, not a citation error.

---

## ZH2019  (5 symbols)

### Symbols

- `braincell/channel/calcium.py::Ca_ZH2019_IO`
- `braincell/channel/calcium.py::Ca_ZH2019_IO_Frozen`
- `braincell/channel/hyperpolarization_activated.py::HCN_ZH2019_IO`
- `braincell/channel/potassium.py::Kdr_ZH2019_IO`
- `braincell/channel/sodium.py::Na_ZH2019_IO`

Mod-file year code: `ZH19`. Cell type: `IO` (inferior olive). All 5
symbols map 1:1 onto a `<mechanism>_ZH19_IO.mod` file. `IO` has no `ion/`
subfolder under `Cerebellum_mod`, so this key has no ion-state symbol.

### Provenance evidence

```
=== IO/channel/Ca_ZH19_IO.mod
COMMENT
Ca channel from Manor (Rinzel, Segev, Yarom) 1997
B. Torben-Nielsen @ HUJI, 7-10-2010
ENDCOMMENT

=== IO/channel/HCN_ZH19_IO.mod
COMMENT
Somatic h channel from Schweighofer et al., 1999
Xu Zhang @ UConn, 6-22-2018
ENDCOMMENT

=== IO/channel/Kdr_ZH19_IO.mod
COMMENT
K_dr channel from Schweighofer et al 1999.
The referred model is an inferior olive neuron
B. Torben-Nielsen @ HUJI, 21-10-2010
ENDCOMMENT

=== IO/channel/Na_ZH19_IO.mod
COMMENT
Na channel from Schweighofer et al 1999.
The referred model is an inferior olive neuron
B. Torben-Nielsen @ HUJI, 21-10-2010
ENDCOMMENT
```

**Inconsistent-author cases in this bucket:** all four files name an
origin unrelated to the `ZH2019`/Zhang-family search target: `Ca_ZH19_IO.mod`
credits "Manor (Rinzel, Segev, Yarom) 1997" and porter "B. Torben-Nielsen
@ HUJI, 7-10-2010"; `HCN_ZH19_IO.mod` credits "Schweighofer et al., 1999"
and porter "Xu Zhang @ UConn, 6-22-2018"; `Kdr_ZH19_IO.mod` and
`Na_ZH19_IO.mod` both credit "Schweighofer et al 1999" and porter
"B. Torben-Nielsen @ HUJI, 21-10-2010". Note `HCN_ZH19_IO.mod`'s "Xu Zhang"
porter credit is at least consistent with the `ZH` initials in the key,
unlike the other three files in this bucket — this is worth flagging to
Task 3 as a possible (not yet verified) partial match.

### Verified record

**Task 1's flag was right: `ZH` is Xu Zhang, the porter named in
`HCN_ZH19_IO.mod`.** The key resolves to **Zhang**, not Zang.
Confirmed 2026-08-15 against the Crossref REST API, PubMed (`efetch`),
the PMC article HTML (PMC6612915), the ModelDB REST API, and the
`github.com/ModelDBRepository/257028` file tree.

.. [1] Zhang, X., & Santaniello, S. (2019). Role of cerebellar
       GABAergic dysfunctions in the origins of essential tremor.
       Proceedings of the National Academy of Sciences of the United
       States of America, 116(27), 13592-13601.
       doi:10.1073/pnas.1817689116

PMID 31209041, PMCID PMC6612915. Published online 17 June 2019, print
issue 2 July 2019. ModelDB accession **257028**, "A
cortico-cerebello-thalamo-cortical loop model under essential tremor
(Zhang & Santaniello 2019)", created 2019-06-04.

**Identification evidence, four independent strands.**

1. The `.mod` headers in ModelDB 257028 are an **exact string match**
   to this repository's: `modfiles/io_h.mod` carries "Somatic h
   channel from Schweighofer et al., 1999" / "Xu Zhang @ UConn,
   6-22-2018"; `io_na.mod` and `io_kdr.mod` carry the Schweighofer
   credit with "B. Torben-Nielsen @ HUJI, 21-10-2010"; `io_ca.mod`
   carries "Ca channel from Manor (Rinzel, Segev, Yarom) 1997" with
   "B. Torben-Nielsen @ HUJI, 7-10-2010".
2. ModelDB's `public_submitter_name` for 257028 is literally
   "Xu Zhang", and its indexed neuron list includes "Inferior olive
   neuron".
3. Crossref gives the first author's affiliation as UConn Biomedical
   Engineering, matching the "@ UConn" porter credit and the June-2018
   porting date against a 2019 publication.
4. The repository's own `<Mechanism>_<Initials><YY>_<Cell>` naming
   convention decodes `ZH19_IO` as Zhang 2019, inferior olive --
   consistent with `SU15_DCN` decoding to Sudhakar 2015, deep
   cerebellar nuclei, which is independently confirmed.

**Yunliang Zang is ruled out**: no 2019 inferior olive paper, and no
UConn affiliation at any point.

**Two caveats that must be recorded.**

- **A 2021 deposit contains byte-identical `io_*.mod` files.** ModelDB
  **266842** (Schreglmann et al. 2021, a phase-locked tACS model, on
  which Xu Zhang is a middle author) ships the same files with the
  same headers -- `io_h.mod` is byte-identical (md5 `66e12eb6...`) in
  both. The header text alone therefore **cannot** discriminate 257028
  from 266842. Strand 4 above resolves it -- a 2021 provenance would
  have been tagged `SC21`/`SR21`, not `ZH19` -- but that is a
  naming-convention inference, a weaker grade of evidence than the
  exact header match. Recorded as `## Unresolved attributions` item 10.
- **The inferior olive neurons in Zhang & Santaniello (2019) are
  single-compartment.** `CCTC_model/cell_ION.hoc` creates
  `IONcell[8]` with `nseg = 1`, `L = diam = 20`, mechanisms
  `ioNa`/`ioKdr`/`ioCa`/`ioh`/`pas`, coupled as an eight-node
  gap-junction lattice; its own header says "Modified from
  Schweighofer et al., 1999 and Torben-Nielson et al., 2012". The
  multi-compartment part of that paper is the Purkinje population, not
  the olive. A docstring must not describe these as multi-compartment
  inferior olive mechanisms.

`IO` has no `ion/` subdirectory under `Cerebellum_mod`, so this key
has no ion-state symbol.

### Attribution

**Attribution check: PASSED for all 5 symbols.**

Method as documented in the `MA2020` `### Attribution` block. All 5
map 1:1 onto an `IO/channel/*_ZH19_IO.mod` file and every
rate-function constant matches. `Ca_ZH2019_IO_Frozen` reports zero
overlap in that scan because it subclasses `Ca_ZH2019_IO`.

**Mapping table.** `.. [1]` is the origin record; `.. [2]` is the
`ZH2019` entry above. O-TN2012 is the intermediate NEURON port and
belongs in `Notes`, not `References`, unless the docstring discusses
the port itself.

| Symbols | Origin `.. [1]` |
|---|---|
| `Na_ZH2019_IO`, `Kdr_ZH2019_IO`, `HCN_ZH2019_IO` | O-SW1999 (via O-TN2012) |
| `Ca_ZH2019_IO`, `Ca_ZH2019_IO_Frozen` | O-MN1997 (via O-TN2012) |

**Fingerprint checked beyond the literal scan.** `Na_ZH2019_IO`
(`braincell/channel/sodium.py:269`) implements
`alpha_m = 0.1 (V+41)/(1 - exp(-(V+41)/10))`,
`beta_m = 9 exp(-(V+66)/20)`, `alpha_h = 5 exp(-(V+60)/15)`,
`beta_h = 10 (V+50)/(1 - exp(-(V+50)/10))` with
`tau_h = 250/(alpha_h + beta_h)` and an instantaneous `m`
(`tau_m = 0.001 ms`) -- the Schweighofer et al. (1999) inferior olive
somatic Na parameterisation, matching `Na_ZH19_IO.mod` term for term.
`Kdr_ZH19_IO.mod`'s `ek = -75 mV` is supplied by the ion object in
BrainCell and so is absent from the class, as expected.

**Caveat for the module task -- a real numerical deviation, not a
citation issue.** `Na_ZH19_IO.mod` and `Kdr_ZH19_IO.mod` each guard a
removable singularity with an explicit
`if (fabs(v + X) < 1e-6) { ... } else { ... }` branch whose taken side
substitutes a perturbed offset literal. **The three guards are not
identical**, and an earlier revision of this paragraph merged them --
corrected here after Task 17a's review, from the sources line by line:

| File | Function | Guard | Perturbed literal | Line |
|---|---|---|---|---|
| `Na_ZH19_IO.mod` | `a_m` | `fabs(v+41.0) < 1e-6` | `41.000001` | 64 |
| `Na_ZH19_IO.mod` | `b_h` | `fabs(v+50.0) < 1e-6` | `50.000001` | 73 |
| `Kdr_ZH19_IO.mod` | `a_n` | `fabs(v+41.0) < 1e-6` | `41.00001` | 59 |

Two things to carry into a docstring. First, `Kdr`'s guarded function
is `a_n`, not `alpha_m` or `beta_h`, and its perturbation is `41.00001`
-- one fewer zero than `Na`'s `41.000001`. A docstring must use its own
mechanism's literal. Second, `Kdr`'s perturbation (1e-5) is ten times
*larger* than the guard window (1e-6) that selects it, so on the taken
branch the substituted offset always lands outside the interval the
branch was testing for; `Na`'s two guards are matched to their windows.

BrainCell replaces all three with a numerically stable
`x/(1 - exp(-x))` helper (`_x_over_one_minus_exp_neg_stable` in
`braincell/channel/sodium.py`), so the perturbed literals do not
appear in the class. The README explicitly excludes those literals
from its NMODL default-precision rewrite table -- they are ordinary
in-formula constants that "keep their original value" -- so the
stable-helper substitution is a separate, undocumented BrainCell
choice. It is exact away from the singularity and better-behaved at
it, but a docstring should say the guard was replaced rather than
imply the mod file's branch was reproduced.

### Import deviations

Transcribed from `examples/neuron_compare/Cerebellum_mod/README.md`.

**These deviations are already applied to the `.mod` files in this
repository.** The README's status column reads `已连续化` ("now
continuous") with the former range under `原` ("formerly"), so opening
a shipped `.mod` file shows `cnexp` and no `TABLE`. That is the
deviation having been made, not evidence against it. Do not read the
shipped file as refuting a row below.

**`TABLE` removed.** None. No `IO` mechanism used a `TABLE`.

**`derivimplicit` -> `cnexp`.** None. `IO` does not appear in the
README's integration-method table.

**Rate-refresh relocation -- this is the `IO` bucket's characteristic
deviation, and it applies to all four mechanisms.** For
`Ca_ZH2019_IO`, `HCN_ZH2019_IO`, `Kdr_ZH2019_IO` and `Na_ZH2019_IO`,
the `rates(v)` call moved from `BREAKPOINT` into `DERIVATIVE states`,
so `inf`/`tau` are refreshed before the `cnexp` state update rather
than after it. Independently confirmed against ModelDB 257028: the
upstream `io_h.mod` has `rates(v)` in `BREAKPOINT`, and this
repository's `HCN_ZH19_IO.mod` has it in `DERIVATIVE states`. That is
a semantic change, not a cosmetic one -- the only other difference
from upstream is the `SUFFIX ioh` -> `SUFFIX HCN_ZH19_IO` rename, and
the `COMMENT` header is untouched, which is why the attribution string
survived intact.

**NMODL default-precision rewrites.** None. The README states
explicitly that `IO`'s in-formula literal `41.000001` keeps its
original value and is **not** part of the default-precision rewrite.
See the singularity-guard caveat above for what BrainCell did instead.

---

## IS2008  (2 symbols)

### Symbols

- `braincell/channel/calcium.py::CaN_IS2008`
- `braincell/channel/calcium.py::CaL_IS2008`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries an
`IS2008`/`IS20` filename fragment. Classical/thalamic-literature key
(Task 2). No repository-local provenance text exists for this key.

### Verified record

**NOT FILLED -- see "Unresolved attributions" item 6.**

The bibliographic record for the candidate paper resolves cleanly, but
the attribution check could not be completed, so no citation is
published here. Do not copy a reference for ``CaN_IS2008`` or
``CaL_IS2008`` out of this file until item 6 is closed.

### Attribution

**Symbols:** ``CaN_IS2008``, ``CaL_IS2008``.

**Attribution check: NOT CONFIRMED.** Recorded in full under
"Unresolved attributions" item 6.

---

## Ba2002  (2 symbols)

### Symbols

- `braincell/channel/potassium.py::KDR_Ba2002`
- `braincell/channel/sodium.py::Na_Ba2002`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries a
`Ba2002`/`Ba20` filename fragment. Classical/thalamic-literature key
(Task 2). No repository-local provenance text exists for this key.

### Verified record

Confirmed 2026-08-15 against PubMed (PMID 12351744), the Journal of
Neuroscience article page, and the full text as deposited in PubMed
Central (PMC6757797), which was read directly for the attribution check
below.

.. [1] Bazhenov, M., Timofeev, I., Steriade, M., & Sejnowski, T. J.
       (2002). Model of thalamocortical slow-wave sleep oscillations and
       transitions to activated states. The Journal of Neuroscience,
       22(19), 8691-8704.
       doi:10.1523/JNEUROSCI.22-19-08691.2002

### Attribution

**Symbols:** ``KDR_Ba2002``, ``Na_Ba2002``.

**Attribution check: PASSED.** Code read at
``braincell/channel/potassium.py:95`` and
``braincell/channel/sodium.py:59``. Full text read from PMC6757797.

Two independent confirmations:

1. *Currents.* The Methods section ("Intrinsic currents: thalamus")
   states verbatim: "For both RE and TC cells we considered a fast
   sodium current, I_Na, a fast potassium current, I_K (Traub and
   Miles, 1991), a low-threshold Ca2+ current, I_T ...". So the paper
   does contain the two currents these symbols implement, and the
   authors themselves attribute their kinetics to Traub & Miles (1991).
2. *Conductance fingerprint.* The same paragraph gives, for TC cells,
   "g_K = 10 mS/cm^2, g_Na = 90 mS/cm^2". BrainCell's defaults are
   ``KDR_Ba2002 g_max = 10.0 mS/cm^2`` and ``Na_Ba2002 g_max = 90.0
   mS/cm^2`` -- an exact match to the paper's TC-cell values, and a
   much stronger fingerprint than the rate equations alone.

Algebraically, ``KDR_Ba2002`` and ``Na_Ba2002`` are the same
Traub-Miles rate functions as ``K_TM1991`` / ``Na_TM1991`` (see the
``TM1991`` attribution block) written in the mirrored sign convention,
both shipping ``V_sh = -50 mV`` and ``q10 = 3`` instead of 1. The -50
mV value replaces -63 mV for the sodium pair only (``Na_Ba2002`` vs.
``Na_TM1991``); the potassium pair's own ``TM1991`` counterpart,
``K_TM1991``, ships -60 mV, not -63 mV -- see the ``TM1991``
attribution block below for the verified values.

**Caveat for the module task:** the paper does *not* print the rate
equations. It defers them: "The expressions for voltage- and
Ca2+-dependent transition rates for all currents are given in Bazhenov
et al. (1998)" (= Bazhenov, Timofeev, Steriade, & Sejnowski, 1998,
J Neurophysiol 79(5), 2730-2748, doi:10.1152/jn.1998.79.5.2730, PMID
9582241 -- record independently confirmed here). The ``V_sh = -50 mV``
shift could therefore not be traced to a printed equation in the 2002
paper itself. A docstring may state that the current is *the one used
in* Bazhenov et al. (2002); it must not claim the 2002 paper prints
these alpha/beta expressions.

---

## TM1991  (2 symbols)

### Symbols

- `braincell/channel/potassium.py::K_TM1991`
- `braincell/channel/sodium.py::Na_TM1991`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries a
`TM1991`/`TM19` filename fragment. Classical/thalamic-literature key
(Task 2). No repository-local provenance text exists for this key.

### Verified record

Confirmed 2026-08-15 against the Cambridge University Press catalogue
and Cambridge Core chapter landing pages (which expose the book DOI and
its chapter-level suffixes), plus the reference list of Bazhenov et al.
(2002) as retrieved from PMC6757797, which cites the same edition.

This is a book, not a journal article, so the entry takes the book form:

.. [1] Traub, R. D., & Miles, R. (1991). Neuronal networks of the
       hippocampus. Cambridge University Press.
       doi:10.1017/CBO9780511895401

ISBN 9780521364812 (0521364817). Chapter-level DOIs exist as
``10.1017/CBO9780511895401.00N`` but no BrainCell symbol needs one.

### Attribution

**Symbols:** ``K_TM1991``, ``Na_TM1991``.

**Attribution check: PASSED.** Code read at
``braincell/channel/potassium.py:129`` and
``braincell/channel/sodium.py:104``. Compared against ``HH2.mod`` from
ModelDB accession 3670 (Destexhe's own NEURON implementation, mirrored
at github.com/ModelDBRepository/3670), whose header reads: "Equations
modified by Traub, for Hippocampal Pyramidal cells, in: Traub & Miles,
Neuronal Networks of the Hippocampus, Cambridge, 1991".

With ``v2 = v - vtraub`` in the mod file and ``V' = V - V_sh`` in
BrainCell, all six rate functions match term for term:

- alpha_m = 0.32 (13 - V') / (exp((13 - V')/4) - 1)
- beta_m  = 0.28 (V' - 40) / (exp((V' - 40)/5) - 1)
- alpha_h = 0.128 exp((17 - V')/18)
- beta_h  = 4 / (1 + exp(-(V' - 40)/5))
- alpha_n = 0.032 (15 - V') / (exp((15 - V')/5) - 1)
- beta_n  = 0.5 exp((10 - V')/40)

**Shift defaults differ -- do not write "both defaulting to -63 mV".**
An earlier revision of this block said the mod file and BrainCell share
a -63 mV default. They do not. ``HH2.mod``'s own ``PARAMETER`` block
ships ``vtraub = -55 (mV)`` (re-read from the ModelDB 3670 GitHub
mirror, 2026-08-15).

**MODULE-TASK WARNING -- BrainCell's two ``TM1991`` classes do not
even agree with each other.** There is no "matching default in
potassium.py"; an earlier revision of this block wrongly claimed one.
Verified 2026-08-15 by reading both constructors directly:
``Na_TM1991`` ships ``V_sh = -63 mV``
(``braincell/channel/sodium.py:121``); ``K_TM1991`` ships
``V_sh = -60 mV`` (``braincell/channel/potassium.py:143``), not
-63 mV. Both classes derive from the same ``HH2.mod`` mechanism (see
"Attribution check" above), so this 3 mV divergence is a BrainCell
choice, not something inherited from the source. **Any docstring
sentence about "the Traub & Miles -63 mV shift" is wrong for
``K_TM1991``** -- write ``K_TM1991``'s default as -60 mV, and
``Na_TM1991``'s as -63 mV.

-63 mV is nevertheless the correct value to ship for ``Na_TM1991``: it
is what Destexhe's network ``.hoc`` code assigns to ``vtraub`` when it
uses this mechanism, and it is the offset associated with the
Traub-Miles hippocampal pyramidal cell. Only the mod file's *own*
default (-55 mV) differs from BrainCell's sodium value; the six rate
equations above are confirmed to match term for term either way, since
the shift enters only through ``v2``/``V'``.

Gating m^3 h and n^4 also match. The mod file's ``tadj = 3^((celsius -
36)/10)`` corresponds to BrainCell's ``q10 = 1.0`` at ``temp_ref = 36
degC`` defaults (both give unity at 36 degC); this is consistent, not a
discrepancy.

**Caveat:** ``Na_TM1991`` ``g_max = 120 mS/cm^2`` and ``K_TM1991``
``g_max = 10 mS/cm^2`` are BrainCell defaults; ``HH2.mod`` ships
gnabar = 0.1 mho/cm^2 (100 mS/cm^2) and gkbar = 0.01 mho/cm^2
(10 mS/cm^2). Do not attribute the 120 to Traub & Miles.

---

## HH1952  (2 symbols)

### Symbols

- `braincell/channel/potassium.py::K_HH1952`
- `braincell/channel/sodium.py::Na_HH1952`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries an
`HH1952`/`HH19` filename fragment. Classical/thalamic-literature key
(Task 2) — this is expected to resolve to the original Hodgkin & Huxley
(1952) squid giant axon paper, but that resolution is Task 2's job, not
this harvest's. No repository-local provenance text exists for this key.

### Verified record

Confirmed 2026-08-15 against PubMed (PMID 12991237), the Physiological
Society / Wiley publisher record for the DOI, and the PubMed Central
deposit PMC1392413. All fields (authors, title, journal, volume, issue,
pages, year, DOI) agree across the three.

.. [1] Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description
       of membrane current and its application to conduction and
       excitation in nerve. The Journal of Physiology, 117(4), 500-544.
       doi:10.1113/jphysiol.1952.sp004764

Do **not** cite doi:10.1007/BF02459568 -- that is the 1990 Bulletin of
Mathematical Biology reprint (52, 25-71), not the original.

### Attribution

**Symbols:** ``K_HH1952``, ``Na_HH1952``.

**Attribution check: PASSED.** Code read at
``braincell/channel/potassium.py:163`` and
``braincell/channel/sodium.py:149``; every rate constant was expanded by
hand and compared with the classical Hodgkin-Huxley rate equations.

With the default ``V_sh = -45 mV`` (so ``V' = V + 45``, putting rest at
-65 mV in the modern absolute-potential convention), the implementation
expands to exactly the published HH rates:

- ``K_HH1952``: alpha_n = 0.01 (V+55) / (1 - exp(-(V+55)/10)),
  beta_n = 0.125 exp(-(V+65)/80), gating n^4.
- ``Na_HH1952``: alpha_m = 0.1 (V+40) / (1 - exp(-(V+40)/10)),
  beta_m = 4 exp(-(V+65)/18), alpha_h = 0.07 exp(-(V+65)/20),
  beta_h = 1 / (1 + exp(-(V+35)/10)), gating m^3 h.

``exprel(x) = (exp(x) - 1) / x`` is used only to remove the removable
singularity at the Boltzmann midpoint; it does not change the function.

**Caveats for the module task (parameter defaults, not citation errors):**

- ``Na_HH1952`` default ``g_max = 120 mS/cm^2`` matches HH's g_Na, but
  ``K_HH1952`` default ``g_max = 10 mS/cm^2`` does **not** match HH's
  g_K = 36 mS/cm^2. Document it as a BrainCell default, not as HH's
  value.
- ``temp_ref`` defaults to 36 degC while HH's rates were measured at
  6.3 degC. ``q10 = 3`` is HH's own factor-of-3-per-10-degC, but the
  reference temperature as shipped makes the correction a no-op at 36
  degC. Do not claim the defaults reproduce HH at 6.3 degC.

---

## HP1992  (1 symbols)

### Symbols

- `braincell/channel/calcium.py::CaT_HP1992`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries an
`HP1992`/`HP19` filename fragment. Classical/thalamic-literature key
(Task 2). No repository-local provenance text exists for this key.

### Verified record

Confirmed 2026-08-15 against PubMed (PMID 1403085, abstract retrieved
via NCBI E-utilities) and the Journal of Neuroscience article page.
PMCID PMC6575965.

.. [1] Huguenard, J. R., & Prince, D. A. (1992). A novel T-type current
       underlies prolonged Ca2+-dependent burst firing in GABAergic
       neurons of rat thalamic reticular nucleus. The Journal of
       Neuroscience, 12(10), 3804-3817.
       doi:10.1523/JNEUROSCI.12-10-03804.1992

### Attribution

**Symbol:** ``CaT_HP1992`` (``braincell/channel/calcium.py:199``).

**Attribution check: PASSED.** The abstract confirms the paper's
subject is a slowly inactivating transient Ca2+ current (I_Ts) measured
by whole-cell voltage clamp in acutely isolated rat thalamic reticular
(nRt) neurons -- exactly the "T-type calcium current for reticular
nucleus" the symbol claims.

Constants were compared against ``IT2.mod`` from ModelDB accession
3670, whose header reads: "The kinetics is described by standard
equations (NOT GHK) using a m2h format, according to the voltage-clamp
data (whole cell patch clamp) of Huguenard & Prince, J Neurosci. 12:
3804-3817, 1992", with "Q10 changed to 5 and 3". The mod file computes:

- ``m_inf = 1/(1 + exp(-(v + shift + 50)/7.4))``
- ``h_inf = 1/(1 + exp((v + shift + 78)/5.0))``
- ``tau_m = 3 + 1/(exp((v+shift+25)/10) + exp(-(v+shift+100)/15))``
- ``tau_h = 85 + 1/(exp((v+shift+46)/4) + exp(-(v+shift+405)/50))``

with ``shift = 2 mV`` (screening charge for external Ca = 2 mM), and
``phi_m = 5^((celsius-24)/10)``, ``phi_h = 3^((celsius-24)/10)``.

``CaT_HP1992`` carries 52, 80, 27, 102, 48 and 407 -- i.e. each
mod-file constant with the 2 mV shift folded in -- and defaults to
``q10_p = 5.0`` / ``q10_q = 3.0`` at ``temp_ref = 24 degC``, matching
phi_m and phi_h exactly. Gating is ``p^2 q`` in both.

**Caveat:** ``CaT_HP1992`` applies a further ``V_sh = -3 mV`` on top of
the folded 2 mV shift, so shipped defaults sit 3 mV depolarized
relative to ``IT2.mod`` defaults. Documented free parameter, not a
citation error. ``g_max = 1.75 mS/cm^2`` matches ``IT2.mod``'s
``gcabar = .00175 mho/cm2``.

---

## Re1993  (1 symbols)

### Symbols

- `braincell/channel/calcium.py::CaHT_Re1993`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries a
`Re1993`/`Re19` filename fragment. Classical/thalamic-literature key
(Task 2). No repository-local provenance text exists for this key.

### Verified record

Confirmed 2026-08-15 against PubMed (PMID 8229187, abstract retrieved
via NCBI E-utilities) and the Journal of Neuroscience article page.
PMCID PMC6576337.

.. [1] Reuveni, I., Friedman, A., Amitai, Y., & Gutnick, M. J. (1993).
       Stepwise repolarization from Ca2+ plateaus in neocortical
       pyramidal cells: evidence for nonhomogeneous distribution of HVA
       Ca2+ channels in dendrites. The Journal of Neuroscience, 13(11),
       4609-4621.
       doi:10.1523/JNEUROSCI.13-11-04609.1993

### Attribution

**Symbol:** ``CaHT_Re1993`` (``braincell/channel/calcium.py:301``).

**Attribution check: PASSED, verbatim.** The abstract confirms the
paper models "the high-voltage-activated Ca2+ conductance underlying
the spike" in compartmental computer models -- i.e. an HVA
(high-threshold) Ca2+ current, which is what the symbol implements.

Constants were compared against ``ca.mod`` from ModelDB accession 2488,
headed "HVA Ca current / Based on Reuveni, Friedman, Amitai and Gutnick
(1993) / J. Neurosci. 13:4609-4621" (implementation by Zach Mainen,
Salk Institute, 1994). Every rate constant matches:

- alpha_m = 0.055 (-27 - V) / (exp((-27 - V)/3.8) - 1)
- beta_m  = 0.94 exp((-75 - V)/17)
- alpha_h = 0.000457 exp((-13 - V)/50)
- beta_h  = 0.0065 / (exp((-15 - V)/28) + 1)

Gating is ``m^2 h`` in both. The temperature parameters also match
exactly: ``ca.mod`` uses ``temp = 23 degC, q10 = 2.3``; ``CaHT_Re1993``
defaults to ``q10_p = q10_q = 2.3`` at ``temp_ref = 23 degC``.

BrainCell writes the rates through ``temp = (-V + V_sh)``, i.e. in the
negated-voltage form the mod file uses inline; with the default
``V_sh = 0 mV`` the two are identical. No discrepancies found.

---

## PC24  (1 symbols)

### Symbols

- `braincell/channel/calcium.py::Cav3p1Test_PC24`

### Provenance evidence

No `.mod` file matches the `PC24` fragment by filename convention (the
regex captured `PC24` from the class name `Cav3p1Test_PC24`, but `PC` is
also the cell-type suffix used elsewhere in this harvest, not a year code
here — this key does not follow the `<initials><2-digit-year>` pattern the
other buckets do). However, the symbol's own docstring in
`braincell/channel/calcium.py` states directly:

> "Template-based import of ``Cav3_1_test.mod``."

Two matching files exist: `examples/neuron_compare/Cerebellum_mod/PC/channel/Cav3_1_test.mod`
and `.../PC/channel/Cav3_1_test2.mod`. Both were harvested by the Step 2
scan (they appear in the file list) but **produced zero matching header
lines** — neither file contains a `TITLE`, `COMMENT`, `Author`, `Ref`, or
`revis` line anywhere at all (checked beyond the first 25 lines too, not
just the Step 2 window). Their only content matching the harvest's `[0-9]
{4}` filter was the numeric constant `9.6485e4` (Faraday's constant), a
false positive, not provenance text:

```
=== PC/channel/Cav3_1_test.mod
	F = 9.6485e4 (coulombs)
	R = 8.3145 (joule/kelvin)

=== PC/channel/Cav3_1_test2.mod
	F = 9.6485e4 (coulombs)
	R = 8.3145 (joule/kelvin)
```

This is a genuine zero-provenance case: the `.mod` source itself is
anonymous. `Cav3p1Test_PC24` is a defrosted/"test" variant of `Cav3p1`
sharing the same steady-state/tau formulas as the templated `Cav3p1`
family (see the docstring already in the source), but nothing in the
`.mod` file names who wrote it or what paper it implements.

### Verified record

**Task 1's "genuine zero-provenance case" verdict is superseded.** The
`.mod` file is anonymous, but it is not unidentifiable: its kinetics
are `Cav3p1_MA24_PC.mod`'s, character for character, so `PC24` inherits
that mechanism's whole attribution chain. `PC24` is therefore the same
model paper as `MA2024`, as the Task 1 brief hypothesised, and the
hypothesis is now confirmed by code comparison rather than assumed
from the `PC` fragment in the class name.

The model-paper entry to copy is the `MA2024` `### Verified record`
block above (Masoli et al., 2024, Communications Biology 7(1), 5,
doi:10.1038/s42003-023-05689-y); the origin-of-kinetics entry is
O-IF2006 with O-AN2012. **Do not retype either.**

**How the identification was made.** `Cav3_1_test.mod` and
`Cav3p1_MA24_PC.mod` declare the identical named parameter block --
`v0_m_inf = -52 mV`, `v0_h_inf = -72 mV`, `k_m_inf = -5 mV`,
`k_h_inf = 7 mV`, `C_tau_m = 1`, `A_tau_m = 1.0`,
`v0_tau_m1 = -40 mV`, `v0_tau_m2 = -102 mV`, `k_tau_m1 = 9 mV`,
`k_tau_m2 = -18 mV`, `C_tau_h = 15`, `A_tau_h = 1.0`,
`v0_tau_h1 = -32 mV`, `k_tau_h1 = 7 mV`, `pcabar = 2.5e-4`,
`q10 = 3` -- and the identical four rate expressions, including
`qt = q10^((celsius - 37)/10)`:

    minf = 1/(1 + exp((v - v0_m_inf)/k_m_inf))
    hinf = 1/(1 + exp((v - v0_h_inf)/k_h_inf))
    taum = (C_tau_m + A_tau_m/(exp((v - v0_tau_m1)/k_tau_m1)
           + exp((v - v0_tau_m2)/k_tau_m2)))/qt
    tauh = (C_tau_h + A_tau_h/exp((v - v0_tau_h1)/k_tau_h1))/qt

**The single difference is the current law**, and it is exactly what
the existing class docstring already says. `Cav3p1_MA24_PC.mod`
computes `ica = (1e3) * pcabar * m*m*h * g` with `g` from GHK and
`pcabar` in `cm/s` (a permeability). `Cav3_1_test.mod` drops the GHK
drive entirely, redeclares `pcabar = 2.5e-4 (S/cm2)` as a conductance
density, and computes `ica = pcabar * m*m*h`. So `Cav3p1Test_PC24` is
the `MA2024` Cav3.1 mechanism with an ohmic current law substituted
for the GHK one -- a variant of a published mechanism, not an
unattributable orphan.

The sibling `Cav3_1_test2.mod` remains unclaimed by any BrainCell
symbol and was not analysed further.

### Attribution

**Symbol:** `Cav3p1Test_PC24`
(`braincell/channel/calcium.py:881`).

**Attribution check: PASSED, with one mandatory caveat.**

The class's own constructor was read and carries `v0_m_inf = -52 mV`,
`v0_h_inf = -72 mV`, `k_m_inf = -5 mV`, `k_h_inf = 7 mV`,
`C_tau_m = 1.0`, `A_tau_m = 1.0`, `v0_tau_m1 = -40 mV`,
`v0_tau_m2 = -102 mV`, `k_tau_m1 = 9 mV`, `k_tau_m2 = -18 mV`,
`g_max = 2.5e-4 S/cm^2`, `q10 = 3.0`, `temp_ref = 37 degC`,
`temp = 22 degC`, with `Gate("p", power=2)` and `Gate("q")` -- i.e. it
reproduces `Cav3_1_test.mod` exactly, and through it the O-IF2006
Cav3.1 kinetics as fitted in Anwar's `CaT3_1.mod` (ModelDB 138382).

**Two-level citation:** `.. [1]` O-IF2006 (with O-AN2012 as the model
the fit was published in), `.. [2]` the `MA2024` model paper.

**Mandatory caveat.** The current law is *not* the published one. The
mechanism as published is a GHK-driven permeability; this variant is
ohmic, and `g_max = 2.5e-4` carries `S/cm^2` here against `cm/s` in
the published mechanism -- the same number in a different dimension.
A docstring must say that the kinetics are those of the Cav3.1
mechanism of the cited model with the GHK drive replaced by a direct
conductance-density current law, and must not present the ohmic form
as the published one. The class name's "Test" is accurate: nothing in
the `.mod` file or the deposit indicates this variant was used to
produce any published result.

### Import deviations

None recorded. `Cav3_1_test.mod` does not appear in any table of
`examples/neuron_compare/Cerebellum_mod/README.md` -- that file covers
only the shipped `channel/ion` mechanisms, and this is a test variant.
It carries no `TABLE`, no `derivimplicit`, and no rate-refresh
relocation, confirmed by reading the file. For the deviations that
apply to the mechanism it is derived from, see the `MA2024`
`### Import deviations` block (`Cav3p1` is not listed there either --
the `MA2024` deviations affect `Kv4p3`, `Kir2p3` and `Kca3p1` only).

---

## Ya1989  (1 symbols)

### Symbols

- `braincell/channel/potassium.py::KNI_Ya1989`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries a
`Ya1989`/`Ya19` filename fragment. Classical/thalamic-literature key
(Task 2). No repository-local provenance text exists for this key.

### Verified record

Confirmed 2026-08-15 against the CaltechAUTHORS record for the chapter
(authors.library.caltech.edu/records/9dnr2-hez54), the Open Library
edition record for ISBN 0262111330 (which supplies the 1989 first
edition's subtitle, "from synapses to networks", publisher, and place),
and the header of Destexhe's ``IM.mod`` (ModelDB 3817), which cites the
chapter directly.

This is a book chapter, not a journal article, so the entry takes the
chapter-in-edited-volume form:

.. [1] Yamada, W. M., Koch, C., & Adams, P. R. (1989). Multiple channels
       and calcium dynamics. In C. Koch & I. Segev (Eds.), Methods in
       neuronal modeling: From synapses to networks (pp. 97-133). MIT
       Press.

**Known ambiguity, recorded rather than papered over.** Sources
disagree on the final page: the CaltechAUTHORS imprint field gives
"97-133", while Destexhe's ``IM.mod`` header gives "p 97-134". Neither
could be adjudicated against a scan of the first edition. 97-133 is
used above because it is the machine-readable bibliographic record; a
maintainer re-auditing this entry should treat the last page as +/- 1.

Note also that CaltechAUTHORS files this 1989 chapter under the
*second* edition's subtitle ("from ions to networks", 1998, ISBN
9780585375878). The 1989 first edition (ISBN 0262111330, MIT Press /
A Bradford Book, Cambridge MA, 524 pp.) is subtitled "From synapses to
networks" per Open Library; that is the edition cited above and the one
contemporaneous with the key's year.

### Attribution

**Symbol:** ``KNI_Ya1989`` (``braincell/channel/potassium.py:405``).

**Attribution check: PASSED, verbatim.** Compared against ``IM.mod``
from ModelDB accession 3817, whose header reads: "Model taken from
Yamada, W.M., Koch, C. and Adams, P.R. Multiple channels and calcium
dynamics. In: Methods in Neuronal Modeling, edited by C. Koch and I.
Segev, MIT press, 1989, p 97-134."

The mod file computes:

- ``m_inf = 1 / (1 + exp(-(v + 35)/10))``
- ``tau_m = tau_peak / (3.3 exp((v + 35)/20) + exp(-(v + 35)/20))``

and a single, non-inactivating, linear-in-m potassium conductance
(``ik = gkbar * m * (v - ek)``). ``KNI_Ya1989`` implements exactly these
two functions and a single ``Gate("p")`` of power 1. This is the M
current (slow non-inactivating K+) of the bullfrog sympathetic ganglion
B-type cell, which the CaltechAUTHORS abstract confirms is the
chapter's subject.

**Caveats for the module task:** ``IM.mod`` ships ``taumax = 1000 ms``
and ``gkbar = 1e-6 mho/cm^2`` (= 1e-3 mS/cm^2); ``KNI_Ya1989`` defaults
to ``tau_max = 4000 ms`` and ``g_max = 0.004 mS/cm^2``. Both are
BrainCell defaults, not values from the chapter. ``IM.mod`` also
assumes Q10 = 2.3 referenced to 36 degC, whereas ``KNI_Ya1989``
defaults to ``q10 = 1.0`` (no temperature correction).

---

## De1994  (1 symbols)

### Symbols

- `braincell/channel/potassium_calcium.py::AHP_De1994`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries a
`De1994`/`De19` filename fragment. Classical/thalamic-literature key
(Task 2). No repository-local provenance text exists for this key.

### Verified record

Confirmed 2026-08-15 against PubMed (PMID 7527077, abstract retrieved
via NCBI E-utilities) and the American Physiological Society publisher
record for the DOI.

.. [1] Destexhe, A., Contreras, D., Sejnowski, T. J., & Steriade, M.
       (1994). A model of spindle rhythmicity in the isolated thalamic
       reticular nucleus. Journal of Neurophysiology, 72(2), 803-818.
       doi:10.1152/jn.1994.72.2.803

### Attribution

**Symbol:** ``AHP_De1994``
(``braincell/channel/potassium_calcium.py:58``).

**Attribution check: PASSED.** The abstract states that "The intrinsic
bursting properties of RE cells in the model were due to the presence
of a low-threshold Ca2+ current and two Ca(2+)-activated currents" --
the slow Ca2+-activated K+ (AHP) current is one of those two.

Constants were compared against ``IAHP.mod`` from ModelDB accession
3670 (the authors' own NEURON implementation of this paper;
``iahp2.mod`` in accession 3808 is the same file). Its header reads:
"Ref: Destexhe et al., J. Neurophysiology 72: 803-818, 1994", and it
implements exactly the scheme ``AHP_De1994`` implements:

    <closed> + n Ca_i <-> <open>     (alpha, beta)

with ``n = 2`` binding sites and ``ik = gkbar * m * m * (v - ek)``,
i.e. Ca-dependent and *not* voltage dependent. The mod file
parameterises it as ``beta = 0.03 (1/ms)`` and ``cac = 0.025 (mM)``,
from which ``alpha = beta / cac^n = 0.03 / 0.025^2 = 48 mM^-2 ms^-1``.
BrainCell's default ``alpha = 48`` reproduces that derived value
exactly, and the ``p^2`` gating, the ``n = 2`` exponent, and the pure
Ca dependence all match.

**Discrepancy found -- flag for the module task.** BrainCell defaults
to ``beta = 0.09 ms^-1`` (confirmed by reading
``braincell/channel/potassium_calcium.py:72``). The reference value is
``beta = 0.03 ms^-1``, a factor of 3 lower.

**Where the 0.03 comes from -- state this precisely.** An earlier
revision of this block called 0.03 "the paper's own reported value".
That overstates the evidence: the 1994 paper is paywalled and was
**not** read; only the PubMed abstract was retrieved. The 0.03 rests
on two secondary sources that agree with each other:

1. ``IAHP.mod``'s ``PARAMETER`` block, which ships ``beta = 0.03
   (1/ms)``. This is the authors' own NEURON implementation, and the
   value was confirmed identical in two independent ModelDB
   accessions (3670 and 3808).
2. BrainPy's ``IAHP_De1994v2`` docstring, which *quotes* the paper as
   saying "The values n=2, alpha=48 ms^-1 mM^-2 and beta=0.03 ms^-1
   yielded AHPs very similar to those RE cells recorded in vivo and in
   vitro" -- while its own constructor nonetheless defaults
   ``beta = 0.09``, which is where BrainCell inherited 0.09 from.

So write "the value used by the authors' reference implementation, and
quoted from the paper by BrainPy" -- not "the paper's own reported
value". Checking that quotation against the published text is still
open.

Unaffected by this: the ``alpha = 48`` derivation above, which is
computed directly from ``IAHP.mod``'s own ``beta`` and ``cac`` and
which BrainCell's default reproduces exactly. In no case may a
docstring present ``beta = 0.09`` as the published value; state 0.03
as the reference value and 0.09 as the BrainCell default, or raise the
mismatch as a separate issue.

---

## Corrections to pre-existing in-code citations

Two `References` blocks already existed in the source tree before this
documentation project began. Task 2 was asked to check them. **This file
is the only artefact Task 2 touches -- no `.py` file was edited.** The
findings below are for whichever module task rewrites those docstrings.

1. **`braincell/ion/calcium.py:227-229`** (in `CalciumDetailed`) --
   "Destexhe, Alain, Agnessa Babloyantz, and Terrence J. Sejnowski.
   *Ionic mechanisms for intrinsic slow oscillations in thalamic relay
   neuron.* Biophysical journal 65, no. 4 (1993): 1538-1552."

   *Record check: PASSED with one error.* Confirmed 2026-08-15 against
   PubMed (PMID 8274647, via NCBI E-utilities); PMCID PMC1225880.
   Authors, journal, volume, issue, pages and year are all correct. The
   title is wrong: the published title ends "... in thalamic relay
   **neurons**" (plural), not "neuron". The entry also carries no DOI.
   Corrected form:

   ```
   .. [1] Destexhe, A., Babloyantz, A., & Sejnowski, T. J. (1993). Ionic
          mechanisms for intrinsic slow oscillations in thalamic relay
          neurons. Biophysical Journal, 65(4), 1538-1552.
          doi:10.1016/S0006-3495(93)81190-1
   ```

   ~~*Attribution check: not performed here.* `CalciumDetailed` is in
   the `NO_KEY` bucket, which Task 3 owns. The abstract does confirm
   the paper's model "included Ca2+ diffusion", which is consistent
   with the Michaelis-Menten Ca pump the class implements, but the pump
   constants were not compared. Task 3 must finish this.~~

   **CLOSED, and the deferred assumption was wrong.** The Task 3
   follow-up performed the check; see `## NO_KEY` -> `### Attribution`
   -> **`CalciumDetailed` -- the correction**. `CalciumDetailed` does
   **not** implement a Michaelis-Menten Ca pump. Its `derivative` is
   the Bazhenov first-order model of item 2 below and nothing else;
   the pump appears only in the docstring's expository prose. The
   record check above stands unchanged and was re-confirmed
   independently. The corrected entry is correct **for the text it
   supports**, and must not be presented as the source of
   `CalciumDetailed.derivative`.

2. **`braincell/ion/calcium.py:230-233`** (in `CalciumDetailed`) --
   "Bazhenov, Maxim, Igor Timofeev, Mircea Steriade, and Terrence J.
   Sejnowski. *Cellular and network models for intrathalamic augmenting
   responses during 10-Hz stimulation.* Journal of neurophysiology 79,
   no. 5 (1998): 2730-2748."

   *Record check: PASSED, no errors.* Confirmed 2026-08-15 against
   PubMed (PMID 9582241, via NCBI E-utilities). Authors, title, journal,
   volume, issue, pages and year are all correct as written. Only the
   DOI is missing. Corrected form:

   ```
   .. [2] Bazhenov, M., Timofeev, I., Steriade, M., & Sejnowski, T. J.
          (1998). Cellular and network models for intrathalamic
          augmenting responses during 10-Hz stimulation. Journal of
          Neurophysiology, 79(5), 2730-2748.
          doi:10.1152/jn.1998.79.5.2730
   ```

   ~~*Attribution check: not performed here* (same reason as item
   1).~~ **CLOSED: PASSED.** The Task 3 follow-up compared the paper's
   first-order model against `CalciumDetailed.derivative` term by
   term; they agree, and this entry -- not item 1's -- is the source of
   what the class computes. **Its parameter defaults are not
   divergences.** `C_rest = 2.4e-4 mM` and `tau = 5.0 ms` are verbatim
   the values in the paper's equation (A5), read from the paper itself;
   the "0.05 uM" in the *existing docstring prose* at
   `braincell/ion/calcium.py:193` is the error and must be corrected or
   removed. The one real BrainCell addition is the `maximum(drive, 0)`
   rectification, which is in neither paper. A further correction
   applies here too: Bazhenov's (A5) credits the model to Destexhe et
   al. (1994a), so the block now carries that paper as `.. [1]`. All of
   this is set out in the `NO_KEY` `### Attribution` block. Note that
   this same paper is where Bazhenov et al. (2002)
   says its INa/IK rate expressions live -- see the `Ba2002`
   attribution block.

3. **`braincell/channel/hyperpolarization_activated.py:78-80`** (in
   `HCN_HM1992`) -- same singular/plural title error ("thalamic relay
   neuron" should be "thalamic relay neurons") and no DOI. Full detail
   and the corrected entry are in the `HM1992` **Verified record** block
   above. The class summary line also contains the typo "propsoed".

---

## Unresolved attributions

Items below need explicit attention in Task 2/3 beyond the default
literature search, because Step 1/Step 2 of this harvest could not
resolve them cleanly:

1. ~~**`CdpStC_NoCAM_MA2020_GoC`**~~ -- **CLOSED by Task 3
   (2026-08-15).** The absent `CdpStC_NoCAM_MA20_GoC.mod` is not a
   missing file: the class is a BrainCell-factored base, and its
   literal set matches `BC/ion/CdpStC_MA25_BC.mod` and
   `SC/ion/CdpStC_RI21_SC.mod` (21 of 26 literals; the five unmatched
   are unit conversions and NMODL range annotations), which are the GoC
   `CdpStC` mechanism with the CAM subnetwork commented out. This is
   what the README's "Ion_dyn inherited variants" table says. Full
   detail and the resulting citation are in the `MA2020`
   `### Attribution` block, Caveat 3.

2. ~~**`SU2015` bucket (all 16 symbols)**~~ -- **CLOSED by Task 3
   (2026-08-15), and the premise of the item was wrong twice.** The
   count is 17 `.mod` files (9 in `DCN/channel/`, 8 in `DCN/ion/`), not
   11; and they are not attribution-free. **9 of the 11 shipped DCN
   mechanisms** carry "Translated from GENESIS by Johannes Luthman and
   Volker Steuber." inside their `COMMENT` block, phrased so that Task
   1's `Author`/`Ref` keyword pattern never matched it;
   `CaLVA_SU15_DCN.mod` and `CdpLVA_SU15_DCN.mod` are the two without
   it, and their absence does not change the outcome (item 8 below has
   the file-by-file count). That resolves the bucket to Steuber et al.
   (2011) via Luthman et al. (2011). See the `SU2015`
   `### Verified record` and
   `### Attribution` blocks. A residual, narrower gap remains and is
   recorded as item 9 below.

3. **`HM1992`, `IS2008`, `Ba2002`, `TM1991`, `HH1952`, `HP1992`,
   `Re1993`, `Ya1989`, `De1994`, `PC24`** — no `.mod` file evidence at
   all was found for any of these 10 keys (9 classical/thalamic keys are
   simply outside the cerebellar `.mod` tree scanned in Step 2; `PC24`'s
   two candidate `.mod` files are themselves anonymous — see the `PC24`
   section above). Task 2 must resolve the 9 classical/thalamic keys from
   the general neuroscience literature directly (short author-initial +
   year keys strongly suggestive of well-known papers, e.g. `HH1952` ->
   Hodgkin & Huxley 1952 — but that identification is Task 2's job to
   make and verify, not this task's).

4. ~~**Unclaimed `.mod` file**~~ -- **CLOSED by Task 3 (2026-08-15).**
   `.../DCN/ion/ToyStoich3ABtoCKinetic_SU15_DCN.mod` was intentionally
   left unported. Its own `COMMENT` says it "exists to validate
   higher-order stoichiometry handling in the BrainCell `KineticIon`
   path against a minimal NMODL `KINETIC` example" (`3a + b <-> c`). It
   is a BrainCell test fixture, not a model of anything published, and
   the same is true of the five `Toy*` files that *were* ported. See
   the `SU2015` `### Attribution` block.

5. **Systemic author mismatch** — every one of the 18 cerebellar `.mod`
   files that carries an explicit `Author:`/`CoAuthor:` line (across the
   `MA2020`, `MA2024`, `MA2025`, and `RI2021` buckets) names an author
   different from the key's own search target. The pattern is completely
   consistent: `Kca2p2_*` and `Kv4p3_*`/`KM_*`/`HCN1_*`/`HCN2_*`/`CaHVA_*`
   variants all carry the *same* header text (D'Angelo/Nieus/Fontana,
   Forti/Solinas, or Solinas/Forti/D'Angelo) copy-pasted verbatim across
   every cell-type port (`GoC`, `GrC`, `PC`, `BC`, `SC`). This is not a
   handful of exceptions — it is the norm for every mechanism that has
   an author line at all. Treat the key name as a "ported by" label, not
   an "authored by" label, for the entire cerebellar half of this
   bibliography.

   **Task 3 confirms and widens this.** It is not only the 18
   author-line files: it holds for every one of the 104 cerebellar
   symbols. Not a single mechanism in any of the seven keys originates
   with the group the key names. That is why `## Origin-of-kinetics
   records` exists and why 99 of the 104 symbols take a
   two-or-more-level citation. The five exceptions take no citation
   at all, and they are exceptions in the opposite direction:
   BrainCell's own `Toy*` test fixtures, which originate with no
   publication whatsoever.

6. **`IS2008` (both symbols: `CaN_IS2008`, `CaL_IS2008`)** -- *record
   resolves; attribution NOT CONFIRMED. Do not cite yet.*

   **What was established.** The key does resolve to a real paper. The
   in-code prose is more specific than the Task 1 brief knew: both class
   summary lines in `braincell/channel/calcium.py` (lines 109 and 352)
   say "Inoue & Strowbridge 2008", not merely "Strowbridge 2008". A
   Crossref bibliographic query plus a PubMed author/year search
   (E-utilities, `Inoue T[Author] AND Strowbridge[Author] AND 2008[DP]`)
   each return exactly one hit, and they agree on every field:

   ```
   Inoue, T., & Strowbridge, B. W. (2008). Transient activity induces a
   long-lasting increase in the excitability of olfactory bulb
   interneurons. Journal of Neurophysiology, 99(1), 187-199.
   doi:10.1152/jn.00526.2007
   ```

   PMID 17959743, PMCID PMC6086124, epub 24 Oct 2007. That record check
   passes cleanly.

   **CORRECTION: the full text IS readable.** An earlier revision of
   this item claimed the Methods section was unreadable and rested the
   whole "not confirmed" verdict on that. The claim was false, and the
   readable text changes the evidence for both symbols.

   *The access route that works.* Fetch the PMC article **HTML** at
   `https://pmc.ncbi.nlm.nih.gov/articles/PMC6086124/` with an
   ordinary browser user-agent. It returns HTTP 200 and serves the
   complete author manuscript (re-confirmed 2026-08-15).

   *Why the OA APIs mislead -- generalise this.* The deposit is an
   NIHPA author manuscript: made freely **readable** under the NIH
   Public Access Policy, but not open-access **licensed**. The
   machine-readable OA surfaces key off the licence, not the
   readability, so every one of them reports absence: `oa.fcgi`
   returns `idIsNotOpenAccess` (re-confirmed 2026-08-15), Europe PMC's
   `fullTextXML` 404s (re-confirmed), the PMC PDF endpoint sits behind
   a proof-of-work challenge, and `journals.physiology.org` returns
   403. **A negative from `oa.fcgi` or `fullTextXML` is not evidence
   that a paper cannot be read.** Always retry the PMC article HTML
   before recording any PMC-hosted paper as inaccessible.

   *What the Methods actually say.* The "Computer simulations" section
   states verbatim:

   ```
   We employed a set of five purely voltage-dependent currents in our
   granule cell model [fast Na current, delayed rectifier K current,
   transient (A-type) K current, low-threshold (T-type) Ca current,
   and high-threshold (P/N-type) Ca current] and one calcium-dependent
   current (calcium- and voltage-dependent nonselective cation
   current).
   ```

   **`CaN_IS2008` -- attribution now substantially supported.** The
   calcium- and voltage-dependent nonselective cation current (I_CAN)
   is present, named explicitly, and central to the paper's thesis:
   it is the current that generates the calcium-dependent
   afterdepolarization underlying the persistent activity the paper is
   about. This lifts the attribution from "plausible" to substantially
   supported.

   **`CaL_IS2008` -- positive evidence of MISATTRIBUTION.** The paper
   contains **no L-type calcium current**. Its only two calcium
   currents are a low-threshold T-type and a high-threshold
   **P/N-type**. The strings "L-type", "L type" and "ICaL" do not
   occur anywhere in the article text. This is the same evidential
   status as `CaHT_HM1992` in item 7 -- not an absence of evidence,
   but positive evidence that the symbol does not correspond to any
   current the cited paper models. Either `CaL_IS2008` is meant to be
   this paper's high-threshold current, in which case it is a P/N-type
   and the symbol name is wrong, or it is a genuine L-type current and
   its source is a different paper entirely.

   **Both symbols still ship no citation. The conservative outcome is
   unchanged; only the evidence behind it changed.** Reading the
   Methods does not close either attribution, because the Methods do
   not print the gating constants. They say only: "Detailed
   information about these models is described in the supplementary
   materials section." That supplement was **not** deposited with the
   PMC author manuscript, so the eight constants listed below still
   cannot be checked against the source.

   Also still true, and still an obstacle: no ModelDB deposit for this
   paper was found (ModelDB's API returns nothing for "Strowbridge",
   and a targeted web search surfaced no accession), so there is no
   reference `.mod` implementation to compare constants against --
   unlike every other key in this task.

   **Positive evidence that these are ports, not original work.** The
   two BrainCell classes are verbatim ports of BrainPy's
   `ICaN_IS2008` and `ICaL_IS2008`
   (`brainpy/dyn/channels/calcium.py`, lines 269 and 757 at HEAD, read
   2026-08-15). Every constant matches: `p_inf = 1/(1 + exp(-(V+43)
   /5.2))`, `tau_p = 2.7/(exp(-(V+55)/15) + exp((V+55)/15)) + 1.6`, the
   `[Ca]/([Ca] + 0.2 mM)` modulation and `E = 10 mV` for the CAN
   channel; `p_inf = 1/(1 + exp(-(V+10)/4))`, `tau_p = 0.4 + 0.7/
   (exp(-(V+5)/15) + exp((V+5)/15))`, `q_inf = 1/(1 + exp((V+25)/2))`,
   `tau_q = 300 + 100/(exp((V+40)/9.5) + exp(-(V+40)/9.5))` and `p^2 q`
   gating for the L-type channel. BrainPy cites the Inoue & Strowbridge
   record above. So the citation is *inherited*, and inheriting an
   unverified citation is exactly what this project exists to stop.

   **Two specific reasons for suspicion, not just missing evidence.**

   - BrainPy's own `ICaN_IS2008` docstring lists *two* references and
     attributes the CAN dynamics to Destexhe et al. (1994) as `[1]`,
     with Inoue & Strowbridge only as `[2]`. The
     `M([Ca]) = [Ca]/([Ca] + 0.2 mM)` modulation term is the standard
     Destexhe I_CAN form. So `CaN_IS2008` looks like a hybrid: a
     Destexhe-family CAN framework carrying an Inoue-Strowbridge
     voltage dependence. A single-source `IS2008` citation would
     misattribute the framework.
   - `CaL_IS2008` defaults to `q10_p = 3.55` / `q10_q = 3.0` at 24 degC
     -- byte-identical to `CaT_HM1992`'s temperature defaults, and 24
     degC is the Huguenard reference temperature, not an olfactory bulb
     one. These read as inherited `p^2 q` template defaults rather than
     values from an Inoue & Strowbridge table. This now compounds with
     the absence of any L-type current in the paper, above: two
     independent signs pointing the same way.

   **What would close this.** The Methods are readable but defer the
   constants, so the remaining artefact is the **supplementary
   materials** of doi:10.1152/jn.00526.2007, which are not deposited
   with the PMC author manuscript. Obtain them from the publisher
   (institutional access) or the print issue and check the eight
   constants listed above. For `CaL_IS2008`, also settle whether the
   symbol is a mislabelled P/N-type or belongs to a different paper.
   Until then the two docstrings must either omit a `References`
   section or state the attribution as unverified; neither may print a
   confident citation.

7. **`CaHT_HM1992`** (`braincell/channel/calcium.py:248`) --
   *attribution FAILED; the record for the `HM1992` key itself is fine.*

   Huguenard & McCormick (1992) models exactly four currents. Its
   abstract enumerates them: IT (transient, **low**-voltage-activated
   Ca2+), IA, IK2 and Ih. There is **no** high-threshold /
   high-voltage-activated Ca2+ current anywhere in the paper. Six of the
   seven `HM1992` symbols map onto those four currents (see the
   `HM1992` attribution block); `CaHT_HM1992` does not.

   Reading the code makes the situation clear: `CaHT_HM1992` is
   character-for-character the same four gating functions as
   `CaT_HM1992` (same 59, 6.2, 0.612, 132, 16.7, 16.8, 18.2, 83, 4.0,
   467, 66.6, 22, 10.5, 28 and the same -80 mV branch point, same
   `p^2 q`, same `q10_p = 3.55` / `q10_q = 3.0` at 24 degC). The *only*
   difference is `V_sh`: `+25.0 mV` instead of `-3.0 mV`. It is the
   low-threshold T current translated 28 mV depolarized and relabelled
   "high-threshold" -- a derived variant, not a current the cited paper
   reports.

   This is inherited from BrainPy's `ICaHT_HM1992`, which has the same
   shape and the same citation.

   **Lead for whoever traces this further: the companion paper.** A
   high-threshold thalamic Ca2+ current *does* exist in the immediately
   adjacent article -- McCormick, D. A., & Huguenard, J. R. (1992),
   "A model of the electrophysiological properties of thalamocortical
   relay neurons", J Neurophysiol 68(4), 1384-1400,
   doi:10.1152/jn.1992.68.4.1384, PMID 1331356. This is the same paper
   the `HM1992` Verified record block warns not to confuse with the
   cited one (note the reversed author order and the adjacent page
   range). Its abstract, retrieved via E-utilities 2026-08-15,
   enumerates the model's currents as "a fast and transient Na+
   current, INa; a persistent, depolarization-activated Na+ current,
   INap; a low-threshold Ca2+ current, I(T); a **high-threshold Ca2+
   current, IL**; a Ca(2+)-activated K+ current, IC; ...". So the
   obvious hypothesis is that `CaHT_HM1992` was meant to be that IL.

   **But the code refutes even that.** `CaHT_HM1992` is not IL: it is
   `CaT_HM1992`'s gating functions verbatim with `V_sh` moved from
   -3 mV to +25 mV, as established above. A translated T current is
   not the companion paper's IL, which has its own kinetics. Cite the
   companion paper only if a future task actually re-derives the class
   against IL's published parameters; do not swap one unverified
   citation for another.

   **Consequence for the module task.** Do not give `CaHT_HM1992` a bare
   `.. [1] Huguenard & McCormick (1992)` reference implying the paper
   describes a high-threshold current. The honest docstring says the
   gating kinetics are those of the T current of Huguenard & McCormick
   (1992) (citation as in the `HM1992` Verified record block), applied
   with a +25 mV shift to produce a high-threshold variant, and that the
   shift is a BrainCell/BrainPy convention with no source in that paper.
   If a genuine high-threshold thalamic Ca current is wanted, the
   provenance of the +25 mV shift needs to be traced separately.

8. **Task 1's harvest missed provenance in five places, and the cause
   was the keyword pattern, not the line window.** Five real
   attribution leads were recorded in this file as absent. All five
   are now resolved, but the pattern matters for anyone re-auditing:
   **a "no provenance found" note in a `### Provenance evidence`
   block is evidence about the harvest, not about the file.**

   **Corrected diagnosis.** An earlier revision of this item blamed
   Step 2's "25-line harvest window". That explanation does not
   survive its own examples: four of the five leads listed below sit
   at lines 1-2, 3-8, 5-6 and 10-14 of their files -- well inside a
   25-line window, and in two cases on the very first lines. Only the
   `Cav3p2_*` lead (line ~85) was actually out of range.

   The real cause is the **keyword pattern**. Step 2 harvested only
   lines matching `TITLE`/`COMMENT`/`Author`/`Ref`/`revis`/4-digit
   year. Every missed lead is phrased so that none of those tokens
   appears on the line carrying the credit:

   - "Translated from GENESIS by Johannes Luthman and Volker
     Steuber." -- names the translators with no `Author`/`Ref` token.
   - ": HH TEA-sensitive Purkinje potassium current" /
     ": Created 8/5/02 - nwg" -- a two-digit year, not four.
   - ": written by Yiota Poirazi on 11/13/00 poirazi@LNC.usc.edu" --
     "written by", not `Author`; again a two-digit year.

   A `COMMENT` block's *opening* line matches the pattern and so was
   captured, while the attribution text **inside** the block did not
   and was dropped -- which is why several files were recorded as
   "`TITLE`-only" or "empty `COMMENT`" despite carrying a credit two
   lines further down. **Widening the line window alone would not
   have found four of these five.** A re-audit should harvest whole
   `COMMENT` blocks verbatim and match on free text ("written by",
   "translated from", "based on", "adapted from", "from ... et al.",
   two-digit dates), not on a fixed keyword list.

   - 9 of the 11 shipped `DCN` mechanisms: "Translated from GENESIS by
     Johannes Luthman and Volker Steuber." -> O-ST2011 / O-LU2011.
     (Not all 11 -- see the count note below.)
   - `Kca2p2_*.mod` (all five cell types): a full citation to
     "Sergio M. Solinas, Lia Forti, Elisabetta Cesana, Jonathan
     Mapelli, Erik De Schutter and Egidio D`Angelo (2008) /
     Computational reconstruction of pacemaking and intrinsic
     electroresponsiveness in cerebellar golgi cells / Frontiers in
     Cellular Neuroscience 2:2" at lines 10-14 -> O-SO2007a (with the
     header's year and volume both corrected).
   - `Kv3p4_*.mod` (all five cell types), recorded as having no
     header lines at all: it has two, ": HH TEA-sensitive Purkinje
     potassium current" / ": Created 8/5/02 - nwg", which identify
     ModelDB 48332's `kpkj.mod` -> O-KH2003.
   - `Cav2p3_MA20_GoC.mod`, recorded as `TITLE`-only: it also carries
     ": written by Yiota Poirazi on 11/13/00 poirazi@LNC.usc.edu" and
     ": From car to Cav2_3" -> O-PO2003a.
   - `Cav3p2_*.mod`: ": (as in Coulter et al., J Physiol 414: 587,
     1989)" at line ~85 -> O-CO1989.

   **Count note: 9 of the 11, not all 11.** The Luthman/Steuber
   translation line was re-checked file by file on 2026-08-15
   (`grep -rl Luthman DCN/`). Of the 11 real (non-`Toy*`) DCN
   mechanisms it appears in 9: all eight `DCN/channel/` files other
   than `CaLVA_SU15_DCN.mod`, plus `DCN/ion/CdpHVA_SU15_DCN.mod`.
   **`CaLVA_SU15_DCN.mod` and `CdpLVA_SU15_DCN.mod` do not carry it.**
   Both instead open with a `COMMENT` describing the GHK coupling
   between the two of them ("This mechanism and the other calcium
   channel (CaHVA.mod) are the only channel mechanisms of the DCN
   model that use the GHK mechanism...") and name no author. **The
   conclusion is unaffected**: the two files are part of the same
   deposit as the other nine, the pair is internally cross-referenced
   by that `COMMENT`, and the O-ST2011 / O-LU2011 chain established in
   the `SU2015` `### Attribution` block rests on the deposit, not on a
   per-file header. It is recorded only so that a re-audit grepping
   for "Luthman" does not read two misses as a discrepancy.

   `NaFHF_MA20_GrC.mod`'s empty `COMMENT` is a genuine absence, but it
   is not a gap: the file is `Nav_MA20_GrC.mod`'s 13-state scheme with
   the blocked-state ladder enabled, so it inherits that file's
   Magistretti/Raman credit. See the `MA2020` `### Attribution` block.

9. **`SU2015`: the per-mechanism attribution is not closed, only the
   per-model one.** The bucket's 11 real symbols are safe to cite as
   the DCN model of Steuber et al. (2011) used by Sudhakar et al.
   (2015), because the translation credit in the `.mod` files and the
   "based on a previously published model [21]" sentence in
   Sudhakar et al. together establish that chain. What could **not** be
   established:

   - No paper in the chain was shown to print the Boltzmann and tau
     constants these classes carry. O-ST2011's parameter tables were
     **not** read; the code-side check was `.mod` -> BrainCell only.
   - Six of the nine channel names (`CaHVA`, `CaL`, `NaF`, `fKdr`,
     `sKdr`, and the phrase "calcium pool") appear **zero** times in
     the Sudhakar et al. (2015) full text.
   - Sudhakar et al. (2015) does not cite Luthman et al. (2011) at
     all; the NEURON-translation step rests solely on the `.mod`
     header line and on Luthman et al.'s own Methods.

   Consequence for the module task: cite as described in the `SU2015`
   `### Attribution` block, and do **not** write that any of these
   papers reports the specific constants. Closing this would need
   O-ST2011's parameter tables read and compared.

10. **`ZH2019`: the exact upstream deposit is inferred, not proved.**
    The `.mod` headers match ModelDB 257028 exactly, but they match
    ModelDB 266842 (Schreglmann et al. 2021) equally exactly --
    `io_h.mod` is byte-identical in both, md5 `66e12eb6...`. The
    identification of 257028 rests on the repository's own
    `<Mechanism>_<Initials><YY>_<Cell>` naming convention decoding
    `ZH19` as Zhang 2019, which is a weaker grade of evidence than the
    header match itself. **This does not affect the citation** --
    Zhang & Santaniello (2019) is the paper either way, since 266842
    is downstream of it and Xu Zhang is an author of both. It affects
    only any claim about which accession the files were downloaded
    from, which no docstring should make.

11. **Symbol count: 104, not 103; 123 keyed, not 122. RESOLVED --
    recorded here as the reconciliation, not as an open question.**
    The Task 3 brief and the project plan say the seven cerebellar
    keys cover 103 public symbols and that the keyed total is 122.
    Both are one short. The figures were re-derived mechanically, by
    parsing `__all__` out of every module under
    `braincell/channel/` and `braincell/ion/` with `ast` and diffing
    the result against this file's own `### Symbols` bullet lists:

    | Bucket | Symbols |
    |---|---|
    | Task 2's nine keys (`HM1992` 7, `IS2008` 2, `Ba2002` 2, `TM1991` 2, `HH1952` 2, `HP1992` 1, `Re1993` 1, `Ya1989` 1, `De1994` 1) | 19 |
    | Task 3's seven keys (`MA2020` 32, `MA2024` 19, `MA2025` 16, `RI2021` 15, `SU2015` 16, `ZH2019` 5, `PC24` 1) | 104 |
    | **Keyed subtotal** | **123** |
    | `NO_KEY` | 32 |
    | **Total** | **155** |

    The diff is empty in both directions: no symbol is in `__all__`
    but missing from this file, none is listed here but absent from
    `__all__`, and none is listed twice. **155 is therefore the figure
    a coverage check should use**, and 123 the keyed figure.

    **What 155 excludes, stated so the number is unambiguous.** The
    parse above reads the *module* files under `braincell/channel/`
    and `braincell/ion/` and does not read either package's
    `__init__.py`. One public symbol lives only there --
    `braincell/ion/__init__.py::build_placeholder_ions`, present in
    that module's `__all__` and documented with `.. autofunction::` in
    `docs/apis/braincell.ion.rst`. A whole-package count would
    therefore return 156, not 155. That symbol is **out of scope by
    the project's own definition** (see the scope exclusion under
    `## How this file was built`) and its absence from this file is
    not a gap. Do not "fix" the count to 156 without also widening the
    scope and adding a record for it.

    *Is `ghk_flux` counted consistently? Yes, and it is the only case
    that could go either way.* It is a module-level **function**; the
    other 154 entries are classes. It is exported in
    `braincell/channel/_base.py::__all__` exactly like the classes,
    it is listed in the `NO_KEY` bucket above, and it is one of the
    155. Nothing in this file counts symbols by "classes only". The
    likeliest origin of the upstream 103/122 is a key-extraction
    pattern requiring a four-digit year, which `PC24` does not match;
    that would drop `Cav3p1Test_PC24` from the keyed side and give
    exactly 103 and 122.

    Note also that the `*Test` classes visible in the source
    (`Cav1p2_MA2020_GoCTest`, `CaHVA_SU2015_DCNTest`, and others) are
    **not** in any `__all__` and are correctly excluded from the 155.
    There are 24 of them across the six affected modules, one per
    templated mechanism family.

12. ~~**`NO_KEY` (32 symbols) is still open and belongs to no task
    yet.** Its `### Verified record` and `### Attribution` blocks are
    still marked `_TODO (Task 3)`, but the Task 3 brief scopes that
    task to the seven cerebellar keys only and Task 3 did not touch
    `NO_KEY`. The marker is therefore stale, not an omission.~~
    **CLOSED** by the Task 3 follow-up. The stale markers are gone,
    three symbols are verified, 29 are recorded as having no primary
    literature source, and the two `CalciumDetailed` sub-items
    inherited from Task 2 are closed in
    `## Corrections to pre-existing in-code citations` items 1 and 2
    -- item 1 with a correction, because the assumption Task 2
    recorded while deferring it turned out to be false.

13. **The Hodgkin & Katz 1949 attribution rests on a secondary
    source.** The record is verified from publisher-deposited metadata
    (Crossref and the Physiological Society's JATS), but the paper's
    *text* is not retrievable: it is free-to-read rather than open
    access, and its PMC deposit is a scan with no text layer. That
    Hodgkin & Katz derive the constant-field **voltage** equation --
    the fact that establishes which of the two GHK papers `ghk_flux`
    should cite first -- is confirmed from Alvarez & Latorre (2017),
    a peer-reviewed JGP centenary review, and not from the 1949 paper
    itself. The Goldman half *is* primary-verified from the OCR'd PMC
    full text. Anyone able to reach the 1949 text should confirm and
    strike this item.

14. **Three numbers inside Destexhe et al. (1993) were not resolved.**
    Sought while checking `CalciumDetailed` and not closed: (a) an
    apparent factor-of-100 inconsistency between `k = 0.1` and
    `k = 10` in the paper's equation (7); (b) a disagreement between
    the 1 um submembrane shell depth stated in the paper and the
    0.1 um used by the widely circulated `cad.mod` implementation of
    it; (c) the individual values of the pump rate constants `c1`,
    `c2`, `c3`, which the paper gives only through the lumped
    `K_T` and `K_d`. **None of these affects any verdict** -- the
    `CalciumDetailed` finding rests on the implementation containing
    no pump term at all, which is settled from the code. They are
    recorded so nobody re-derives them.

15. **Where the NEURON reference sits is a judgement call.**
    `Factor`, `Species`, `Reaction`, `Source` and `Conserve` are the
    surface that maps onto `COMPARTMENT`, `STATE`, `~ A <-> B (kf,
    kb)` and `CONSERVE`, so an argument exists for citing Hines &
    Carnevale on each of them. The determination taken here is that
    the reference belongs on `KineticIon`, which implements the
    semantics, and that the five dataclasses -- inert frozen records
    with no behaviour -- ship without a `References` section and point
    at `KineticIon` through `See Also`. A later task may revisit this;
    it should do so deliberately, not by accident.

16. **Two no-source determinations are weaker than the other 27.**
    `K_Kv_test` uses the generic `vtrap` alpha/beta idiom found in
    dozens of unrelated NEURON `kv.mod` files, and
    `CalciumFirstOrder` hard-codes `alpha = 0.13`, `beta = 0.075`.
    Neither is traceable to a paper from the code alone, and neither
    was pursued further: `K_Kv_test` is marked a fixture by its name,
    its zero default conductance and its unit `Q10`, and
    `CalciumFirstOrder` states no more than the generic first-order
    form. Both are recorded as no-source. If a later task finds a
    source for either, it is an addition to the allowlist's
    exceptions, not a contradiction of a verified record -- no
    `.. [N]` entry was written for either.

    **UNVERIFIED LEAD for `CalciumFirstOrder`, to be checked before
    shipping.** One candidate was *not* pursued and should be, because
    it is the best-known source of a bare `Ca' = -alpha*I_Ca -
    beta*Ca` with numeric `alpha`/`beta` and no shell-depth term:

    ```
    Pinsky, P. F., & Rinzel, J. (1994). Intrinsic and network
    rhythmogenesis in a reduced Traub model for CA3 neurons. Journal
    of Computational Neuroscience, 1(1-2), 39-60.
    ```

    **This is a lead, not a verified record.** The bibliographic
    fields above were read from Crossref (doi:10.1007/BF00962717) and
    are given only so the paper can be found; **the attribution was
    not checked at all.** Specifically, nobody has confirmed that the
    paper's calcium equation prints `alpha = 0.13` and `beta = 0.075`,
    or that its calcium variable is scaled the way
    `CalciumFirstOrder`'s is. Do **not** cite it, do not copy the
    block above into a `.. [N]` entry, and do not treat its presence
    here as weakening the no-source determination, which stands until
    the check is done.

    *What closing it would take.* Read the paper's model equations,
    compare the two constants and the sign convention against
    `braincell/ion/calcium.py::CalciumFirstOrder.derivative`, and be
    aware that the reduced-Traub calcium variable is conventionally
    dimensionless -- if BrainCell's is in `mM`, the constants will not
    match numerically even if the model is the right one, and that
    mismatch would need explaining rather than waving through. If the
    check passes, this becomes a verified record and the symbol leaves
    the no-source list; if it fails, strike this lead so nobody
    re-derives it.

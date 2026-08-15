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

- `### Verified record` and `### Attribution` blocks are intentionally
  **empty** in this task. They are filled in by Task 2 (classical/thalamic
  keys) and Task 3 (cerebellar-model keys) after literature verification.
- Nothing in this file's `### Provenance evidence` blocks is a citation.
  It is raw, unedited text copied from `.mod` file headers (typos,
  inconsistent spacing, and missing apostrophes preserved verbatim) plus
  structural notes about what was and was not found. Do not treat any of
  it as verified.

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
- Title: sentence case exactly as published, ending in a period. Do not
  "fix" the publisher's capitalisation, and do not silently correct a
  singular/plural mismatch without noting it.
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

### Verified record

_TODO (Task 2 / Task 3): not yet verified._

### Attribution

_TODO (Task 2 / Task 3): not yet filled in._

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

_TODO (Task 3): not yet verified._

### Attribution

_TODO (Task 3): not yet filled in._

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

_TODO (Task 3): not yet verified._

### Attribution

_TODO (Task 3): not yet filled in._

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

_TODO (Task 3): not yet verified._

### Attribution

_TODO (Task 3): not yet filled in._

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

_TODO (Task 3): not yet verified._

### Attribution

_TODO (Task 3): not yet filled in._

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

_TODO (Task 3): not yet verified._

### Attribution

_TODO (Task 3): not yet filled in._

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

Constants were cross-checked against Destexhe's NEURON implementations
of this paper: ``IT.mod`` (ModelDB 3817) and ``ITGHK.mod`` (ModelDB
279), both headed "Model of Huguenard & McCormick, J Neurophysiol 68:
1373-1383, 1992". Those files use
``m_inf = 1/(1 + exp(-(v + shift + 57)/6.2))``,
``h_inf = 1/(1 + exp((v + shift + 81)/4))``,
``tau_m = 0.612 + 1/(exp(-(v+132)/16.7) + exp((v+16.8)/18.2))``, and a
piecewise ``tau_h`` of ``exp((v+467)/66.6)`` below -80 mV and
``28 + exp(-(v+22)/10.5)`` above, with ``shift = 2 mV`` (screening
charge at 2 mM external Ca). BrainCell's ``CaT_HM1992`` carries 59 and
83 as the Boltzmann midpoints -- i.e. the mod-file values with the 2 mV
shift already folded in -- and reproduces ``tau_m``, the piecewise
``tau_h``, and the ``p^2 q`` gating exactly. ``HCN_HM1992``'s
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

_TODO (Task 3): not yet verified._

### Attribution

_TODO (Task 3): not yet filled in._

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
       22(19), 8691-8704. doi:10.1523/JNEUROSCI.22-19-08691.2002

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
with ``V_sh = -50 mV`` instead of -63 mV and ``q10 = 3`` instead of 1.

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
BrainCell (both defaulting to a -63 mV shift), all six rate functions
match term for term:

- alpha_m = 0.32 (13 - V') / (exp((13 - V')/4) - 1)
- beta_m  = 0.28 (V' - 40) / (exp((V' - 40)/5) - 1)
- alpha_h = 0.128 exp((17 - V')/18)
- beta_h  = 4 / (1 + exp(-(V' - 40)/5))
- alpha_n = 0.032 (15 - V') / (exp((15 - V')/5) - 1)
- beta_n  = 0.5 exp((10 - V')/40)

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
       4609-4621. doi:10.1523/JNEUROSCI.13-11-04609.1993

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

_TODO (Task 3): not yet verified._

### Attribution

_TODO (Task 3): not yet filled in._

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
to ``beta = 0.09 ms^-1``; the reference implementation and the paper's
own reported value are ``beta = 0.03 ms^-1`` (a factor of 3). This is
inherited from BrainPy, whose ``IAHP_De1994v2`` docstring itself states
"The values n=2, alpha=48 ms^-1 mM^-2 and beta=0.03 ms^-1 yielded AHPs
very similar to those RE cells recorded in vivo and in vitro" while its
constructor still defaults ``beta = 0.09``. The docstring must not
claim that ``beta = 0.09`` is the published value; either state 0.03 as
the paper's value and 0.09 as the BrainCell default, or raise the
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

   *Attribution check: not performed here.* `CalciumDetailed` is in the
   `NO_KEY` bucket, which Task 3 owns. The abstract does confirm the
   paper's model "included Ca2+ diffusion", which is consistent with the
   Michaelis-Menten Ca pump the class implements, but the pump constants
   were not compared. Task 3 must finish this.

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

   *Attribution check: not performed here* (same reason as item 1). Note
   that this same paper is where Bazhenov et al. (2002) says its INa/IK
   rate expressions live -- see the `Ba2002` attribution block.

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

1. **`CdpStC_NoCAM_MA2020_GoC`** (`braincell/ion/calcium.py`) — no
   `CdpStC_NoCAM_MA20_GoC.mod` file exists anywhere under
   `examples/neuron_compare/Cerebellum_mod`. Only the `CAMOnly` and
   plain `CdpStC` variants have `.mod` counterparts in `GoC/ion/`. The
   `NoCAM` variant's provenance (is it a hand-derived complement of the
   other two, or ported from elsewhere?) must be established directly
   from the BrainCell source/tests, not from a `.mod` header.

2. **`SU2015` bucket (all 16 symbols)** — none of the 11 `.mod` files
   under `DCN/channel/` and `DCN/ion/` carry any `Author:`/`Ref:`/
   `Written by` line. Headers are bare `TITLE ... (DCN) neuron` with an
   empty `COMMENT` block. Task 3 has zero in-repo textual lead for this
   entire cell type and must resolve purely from the `SU2015` key
   (presumably a "Su" or similarly-initialed 2015 DCN modeling paper)
   and cross-checking with published deep cerebellar nuclei models.

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

4. **Unclaimed `.mod` file**: `examples/neuron_compare/Cerebellum_mod/DCN/ion/ToyStoich3ABtoCKinetic_SU15_DCN.mod`
   has no corresponding BrainCell symbol in the `SU2015` bucket (or
   anywhere else in `braincell/ion/` or `braincell/channel/`). Flagged in
   case a future task needs to know it was intentionally left unported,
   not missed.

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

   **Why the attribution still fails.** The paper's Methods could not be
   read, so it was not possible to confirm that it contains the two
   currents these symbols implement:

   - The full text is *not* open access. `oa.fcgi` returns
     `idIsNotOpenAccess` for PMC6086124; the Europe PMC `fullTextXML`
     endpoint 404s; the PMC PDF endpoint is behind a proof-of-work
     challenge; and `journals.physiology.org` returns 403.
   - No ModelDB deposit for this paper was found (ModelDB's API returns
     nothing for "Strowbridge", and a targeted web search surfaced no
     accession), so there is no reference `.mod` implementation to
     compare constants against -- unlike every other key in this task.
   - The published **abstract** is consistent with, but does not
     establish, the attribution. It confirms a computational model of
     olfactory bulb granule cells and says persistent activity "results
     from interactions between calcium-dependent afterdepolarizations
     and low-threshold Ca spikes in granule cells". A Ca-dependent ADP
     is the usual signature of an I_CAN, and "Ca spikes" implies a Ca
     current, so both classes are *plausible*. Plausible is not
     confirmed.

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
     values from an Inoue & Strowbridge table.

   **What would close this.** Read the Methods/Appendix of
   doi:10.1152/jn.00526.2007 (institutional access, or the print issue)
   and check for the eight constants listed above. Until then, the two
   docstrings should either omit a `References` section or state the
   attribution as unverified; they must not print a confident citation.

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

   **Consequence for the module task.** Do not give `CaHT_HM1992` a bare
   `.. [1] Huguenard & McCormick (1992)` reference implying the paper
   describes a high-threshold current. The honest docstring says the
   gating kinetics are those of the T current of Huguenard & McCormick
   (1992) (citation as in the `HM1992` Verified record block), applied
   with a +25 mV shift to produce a high-threshold variant, and that the
   shift is a BrainCell/BrainPy convention with no source in that paper.
   If a genuine high-threshold thalamic Ca current is wanted, the
   provenance of the +25 mV shift needs to be traced separately.

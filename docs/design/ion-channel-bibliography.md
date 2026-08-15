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

_TODO (Task 2): not yet verified._

### Attribution

_TODO (Task 2): not yet filled in._

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

_TODO (Task 2): not yet verified._

### Attribution

_TODO (Task 2): not yet filled in._

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

_TODO (Task 2): not yet verified._

### Attribution

_TODO (Task 2): not yet filled in._

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

_TODO (Task 2): not yet verified._

### Attribution

_TODO (Task 2): not yet filled in._

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

_TODO (Task 2): not yet verified._

### Attribution

_TODO (Task 2): not yet filled in._

---

## HP1992  (1 symbols)

### Symbols

- `braincell/channel/calcium.py::CaT_HP1992`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries an
`HP1992`/`HP19` filename fragment. Classical/thalamic-literature key
(Task 2). No repository-local provenance text exists for this key.

### Verified record

_TODO (Task 2): not yet verified._

### Attribution

_TODO (Task 2): not yet filled in._

---

## Re1993  (1 symbols)

### Symbols

- `braincell/channel/calcium.py::CaHT_Re1993`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries a
`Re1993`/`Re19` filename fragment. Classical/thalamic-literature key
(Task 2). No repository-local provenance text exists for this key.

### Verified record

_TODO (Task 2): not yet verified._

### Attribution

_TODO (Task 2): not yet filled in._

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

_TODO (Task 2): not yet verified._

### Attribution

_TODO (Task 2): not yet filled in._

---

## De1994  (1 symbols)

### Symbols

- `braincell/channel/potassium_calcium.py::AHP_De1994`

### Provenance evidence

No `.mod` file under `examples/neuron_compare/Cerebellum_mod` carries a
`De1994`/`De19` filename fragment. Classical/thalamic-literature key
(Task 2). No repository-local provenance text exists for this key.

### Verified record

_TODO (Task 2): not yet verified._

### Attribution

_TODO (Task 2): not yet filled in._

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

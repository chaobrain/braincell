# ModelDB sources without a license statement

`.mod` files from ModelDB entries whose files carry no license statement,
kept byte for byte for the NEURON comparisons under `validation/neuron/`.
The maintainers decided on 2026-11-13 (MECH-48, D13) that such files may be
committed with their provenance; braincell does not relicense them.

One folder per ModelDB accession number. Each folder's README gives the
ModelDB page, the GitHub mirror (`ModelDBRepository/<accession>`) at the
commit the files were taken from, the retrieval date, every file's SHA-256,
the paper and the comparisons that use the files. `.gitattributes` turns off
line-ending conversion here, so the hashes hold for the checked-out files,
and the comparisons check them before compiling (`validation/neuron/ion/runner_test.py`
checks every file against its README). A channel_no_conc configuration names
its files in `identity.pinned_mod_files`; they are compiled into the `testing`
library.

| Folder | Paper | Files |
|---|---|---|
| [`148253/`](148253/README.md) | Jedlicka et al. 2011 | `cldif.mod`, `gabaA_Cl.mod` |
| [`6763/`](6763/README.md) | Canavier 1999 | `nadifl.mod`, `pump.mod`, `leak.mod` |
| [`127021/`](127021/README.md) | Botta et al. 2010 | `K_conc.mod` |
| [`279/`](279/README.md) | Destexhe et al. 1998 | `ITGHK.mod` |

A new folder follows the same layout: copy the files unmodified, record the
hashes in its README and in the comparison that reads them, and keep any
edited variant elsewhere (for example `data/mechanisms/testing/`) under its
own name.

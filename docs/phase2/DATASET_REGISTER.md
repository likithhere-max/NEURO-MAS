# Dataset Register (Phase 2, B)

Only datasets that the code actually references are listed. **No dataset was downloaded or redistributed by Phase 2**;
licences and terms below must be confirmed by the owner from the primary source before any redistribution or publication.

## Datasets used by the shipped ML demos
| Key | Name / source | URL used by the loader | Records (as documented by the publisher) | Used for | Licence / terms |
|---|---|---|---|---|---|
| `parkinsons` | UCI "Parkinsons" (id 174). Little M. A. et al., *Suitability of dysphonia measurements for telemonitoring of Parkinson's disease*, IEEE Trans Biomed Eng 2009 (data first described 2007) | `https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data` | 195 recordings; 22 voice features + `status`; 147 PD / 48 healthy; ~31–32 people | disease classifier | **To verify** at the UCI page (UCI states CC BY 4.0 for many datasets; confirm for this one) |
| `parkinsons_updrs` | UCI "Parkinsons Telemonitoring" (id 189). Tsanas A. et al., IEEE Trans Biomed Eng 2010 | `https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data` | 5875 recordings; 42 patients; 22 columns | risk classifier | **To verify** at the UCI page |

Citation details above are from the publishers' descriptions as I know them; **verify the exact bibliographic entries before citing**.

## Integrity controls in the code
* `ml/validation/data.py` verifies each file against the publisher's structure (row/column counts, label counts, subject
  parsing) and raises `DatasetVerificationError` (status `failed`) instead of analysing a wrong or altered file. The counts are
  integrity checks, **not** tuning targets.
* `fetch` records SHA-256 in `ml/data/DATASETS.lock.json` on first use (trust-on-first-use); later runs refuse a file whose hash differs.
  **That lock file does not exist yet** because the download failed here; it will be created on the first successful `fetch`.
* Downloads happen only when the user runs `fetch`; tests never download.

## Known limitations of these datasets (affect any conclusion)
* Small number of people (≈ 31 and 42); repeated recordings per person (≈ 6 and ≈ 140) are not independent.
* Disease data: ~75 % PD, one lab, one protocol (sustained vowels); no external cohort.
* Telemonitoring data: home device, patients with early PD, 6 months; labels are derived (ΔUPDRS ≥ 3), not clinician-adjudicated outcomes.
* Neither contains information about the intended deployment population.

## Other data in the repository
| Item | What it is | Provenance caveat |
|---|---|---|
| `backend/data/diseases_curated.json` and `disease_seed_275.csv` (knowledge base for retrieval) | curated structured summaries for 10 fully populated neurological conditions, seeded from Orphadata/ICD references | Source list and licence notes: `backend/data/sources_and_licenses.txt` (Orphadata science files are CC BY 4.0 per Orphadata; WHO materials have their own reuse terms — verify per asset; SNOMED CT must not be redistributed without a licence). Citation URL liveness could **not** be checked (all 23 unique URLs unreachable from the sandbox ⇒ "unverifiable", not "dead"); ICD/Orphanet codes pending clinician verification; review queue: `backend/data/kb_review_queue.json` |
| Preset demo scenarios in the frontend | invented example patients | synthetic; not data |
| Test fixtures | invented | synthetic |

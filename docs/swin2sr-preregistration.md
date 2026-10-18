# Pre-registered selection grid and fresh test (review R88-M1, R88-M4, R88-D2)

Status: **pre-registered 2026-10-18, after a reduced CPU timing and memory smoke of C0–C2 (48 / 16 pairs, 2 epochs; validation PSNR C0 26.943, C1 27.059, C2 27.042 — the 0.10 dB margin was set with these numbers in view) and before the full grid or any fresh-test score.** Wording corrected 2026-10-18 after the peer review (F2); `docs/swin2sr-preregistration-amendment-1.md` adds the fresh-test reading rule and the incomplete-grid rule.
This file is committed on its own, ahead of the code that implements it. A later commit may add results under
"Results". It must not change the grid, the rules, the fixed settings or the fresh-test pin above that heading; any
change to them is a new pre-registration with its own date.

Source: the maintainer's rulings of 2026-10-17 on the row 88 tutorial review (R88-M4: a fresh test of 144 photographs,
24 per species; R88-D2: grid on with a GPU, off on CPU). The review found that the shipped default (one trainable stage
at learning rate 1e-4) was accepted as "the smallest configuration that beat bicubic on the held-out split", that is,
on the test split, and that no other configuration ever completed. This grid replaces that justification with a
validation-only choice, and the fresh test replaces the exposed one as the evaluation.

## What is fixed in advance (not tuned)

| Item | Value |
|---|---|
| Model | `caidas/swin2SR-classical-sr-x2-64` at `cee1c923c6a37361c6e5650b65dcf4be821e5d52` (digest-verified snapshot) |
| Data | The 360-photograph iNaturalist sample (`SAMPLE_DIGEST` `827eb4a3…`), split seed 42, 36 / 8 / 16 per species (216 / 48 / 96) |
| Degradation | 192-px centre crop; bicubic ÷2, then JPEG quality 40 |
| Training | 5 epochs, batch 8, adaptation seed 0, AdamW without weight decay, gradient clipping 1.0, L1 loss, no scheduler, no augmentation |
| Epoch rule | Within one config, the epoch with the highest validation PSNR on the 48 validation pairs (epoch 0 = the frozen model) |
| Exposed development test | The 96-crop seed-42 test split. It was looked at during development (the old default was accepted on it), so it is reported for continuity only, never as the claim |
| Fresh test | 144 photographs pinned in `FRESH_TEST_RECORDS` (`FRESH_TEST_DIGEST` `57a6730d…`); see below |

## The grid

| Config | Trainable stages | Learning rate | Role |
|---|---|---|---|
| C0 | 1 | 1e-4 | The current default |
| C1 | 1 | 2e-4 | A larger step |
| C2 | 2 | 1e-4 | More capacity |

## Selection rules (validation PSNR only; no test crop plays a part)

1. Each config's score is its best-epoch validation PSNR on the 48 validation pairs. SSIM is reported, not used.
2. C1 replaces C0 only if its score is at least **0.10 dB** higher than C0's.
3. C2 is selected only if its score is at least 0.10 dB higher than C0's **and** at least 0.10 dB higher than the better
   one-stage config (max of C0 and C1).
4. Otherwise C0 is kept (it is the current and the smaller config). Ties within tolerance keep C0.
5. The frozen model is kept unless the selected config's score is at least 0.10 dB higher than epoch 0 (the frozen
   model's validation PSNR).
6. 0.10 dB is a practical tolerance (about twice the last-epoch gain of 0.039 dB seen in the recorded C0 run; a few
   hundredths of a decibel is seed-level variation here), not a significance threshold.

The rules are implemented in `select_config` (`src/swin2sr_super_resolution_pipeline/samples.py`) and tested in
`tests/test_tutorial_review_2026_10_17.py`.

## When the grid runs (owner ruling D2)

- With a GPU (`RUN_GRID = 'auto'` and a CUDA device), the grid runs by default in Section 7, then the selected config
  is scored. This is the pre-registered procedure.
- On CPU the grid is off by default (`RUN_GRID = 'auto'`): the run reuses the choice recorded by the first hosted run
  (`RECORDED_SELECTION`; `None` until that run is recorded, and then C0 is used and the notebook says so). **Such a run
  differs from the pre-registration, so it is exploratory and never scores the fresh test.** `RUN_GRID = 'on'` runs
  the grid on CPU too (more than an hour on the build workstation).

## Score-once rule (SCORE_ONCE_RULE.md items 1–8)

- The fresh test is scored only when **every** setting above holds, plus the grid itself (`PREREGISTERED_SETTINGS`,
  compared by `preregistration_differences`), and `model_state_check` confirms that the model in memory is the one the
  procedure selected (stages, learning rate, epochs, batch size, seed, 216 / 48 pairs).
- Anything else is exploratory: a changed form field (`EPOCHS`, `BATCH_SIZE`, `SPLIT_SEED`, `RUN_GRID = 'off'`), BYOD,
  the Section 10 activity, the out-of-memory tip `BATCH_SIZE = 4`, or the CPU default. An exploratory run is compared
  on validation and the exposed development test, writes `*_exploratory_<n>.json` files and never overwrites the
  pre-registered exports.
- Outputs carry `test_look` ("pre-registered evaluation (this runtime)" | "reproduction" | "not scored (exploratory)")
  and `settings_differing`.
- The first hosted run with the pre-registered settings is **the evaluation of record**; it is written into
  `RECORDED_EVALUATION` and `RECORDED_SELECTION` and into `docs/release-verification.md`. Later identical hosted runs
  are reproductions. Nothing is changed in response to the fresh-test number.
- CPU checks by the fixer ran with the fresh test masked (rule 8): no CPU look at the fresh test exists.

## The fresh test (R88-M4)

- **Size:** 144 photographs, 24 per species (the target; reached for every species), 19,138,331 bytes.
- **Pin:** `FRESH_TEST_RECORDS` in `samples.py` (record id, species, iNaturalist photo id, observation id, observer
  login, bytes, SHA-256, extension) and `FRESH_TEST_DIGEST` `57a6730d6a3b9d3c9d250b4afa6cab2f47ee169e245a1b38ff85603cbf3f61b6` (the `validate_dataset` digest of the 144
  records after the same crop and degradation as the sample). The draw record is `docs/fresh-test-draw.json`.
- **Query:** iNaturalist API v1 `/v1/observations`, `quality_grade=research`, `photo_license=cc0`, `photos=true`,
  `order_by=id`, `order=asc`, `per_page=200`, paged by `id_above`, `id_below=400,000,000`, one
  `taxon_name` per species; User-Agent `dimer-fresh-test-draw/1.0` (no account and no e-mail sent). Frames listed
  2026-10-18: song_sparrow 5,375, chipping_sparrow 2,216, white_throated_sparrow 2,563, dark_eyed_junco 5,686, house_finch 5,465, american_goldfinch 3,131 observations (ids, SHA-256 of
  the id list and the largest id are in the draw record; the largest listed ids, 398-400 million, sit just under the `id_below` bound, so the bound acted as a recent-observation cutoff on 2026-10-18. No date cutoff was required).
- **Seeded selection:** each frame (ascending observation id) shuffled with `random.Random(f"20261018:<species>")` and
  walked in that order (`tools/draw_fresh_test.py`).
- **Accepted only if:** the first CC0 photo is served as `medium.<ext>` with both sides >= 192 px; the observer is
  in none of the 229 observers of the 360 sample photos and the 20 other fleet photos and
  has not already been used (one photo per observer across all 144); photo and observation ids are in no fleet pin;
  not same-place-same-day (same date and coordinates within 0.01 degrees, or the same place text) as any pinned or
  accepted photo; not a confirmed near-duplicate. Rejections: {"observer already used or pinned": 704}.
- **Near-duplicate screen:** against the 380 pinned photos (train, validation, exposed development test and the
  other fleet photos) and the photos already accepted. Two 64-bit dHashes: the repository's 192-px centre crop
  (`samples.centre_crop`) and the whole photo. Thresholds were set on synthetic copies of 60 pinned photos before the
  walk, never on candidates: crop Hamming <= 4 (largest distance over 360 same-geometry copies: JPEG 30-90,
  +-10 % brightness), whole-photo Hamming <= 13 (480 copies, adding 0.9x / 1.1x rescales). A hit is
  excluded only when content confirms it (64-px greyscale normalised cross-correlation >= 0.9943 for the crop,
  >= 0.9897 for the whole photo, the smallest values on the synthetic copies). Result: 4 whole-photo hash
  hits, 0 confirmed by content (largest correlation 0.47), so none was excluded.
  On unrelated pinned pairs the crop threshold catches 0 of 72,010 and the whole-photo threshold
  7 (whole-photo hash collisions between different photographs, which is why a content check precedes any exclusion).
- **First scored:** on the first hosted run under the pre-registered settings. Until then no runtime has scored it.

## Training data of the checkpoint (DAT9)

Upstream trained the classical ×2 checkpoint on DIV2K and Flickr2K bicubic pairs (Swin2SR paper; the snapshot README
discloses nothing more). It was not trained on iNaturalist. Flickr2K is scraped from Flickr, so an exact overlap with
an iNaturalist photograph cannot be ruled out, though it is unlikely; no date cutoff was applied to the draw for that
reason, per the owner ruling.

## Expected cost

| Step | T4 | Build-workstation CPU |
|---|---|---|
| C0 | 316.7 s (recorded, Colab T4, 2026-10-09) | 1,314.5 s (recorded, blob `4af52954`) |
| C1 | ≈ 317 s (same compute as C0) | ≈ 1,315 s |
| C2 | ≈ 580 s (C0 × 1.84, the ratio in the reduced CPU smoke) | ≈ 2,420 s |
| Whole hosted session | ≈ 32 min (913.5 s recorded without the grid + C1 + C2 + scoring 144 more crops) | grid off by default |

Reduced CPU smoke (2026-10-18, 48 training / 16 validation pairs, 2 epochs, batch 8, main `136092f`, fresh test not
loaded): C0 139.5 s, C1 138.3 s, C2 256.8 s, all completed. Peak private memory on CPU: C0 8.5 GiB, C2 15.8 GiB. Three
earlier C2 attempts crashed with an access violation while another process was competing for memory; run alone, C2
completed (twice, at 1 and 4 threads). **C2's memory need is roughly 1.9× C0's**; whether it fits a T4's 15 GB at batch
8 is unproven until the hosted run. If C2 cannot complete on the hosted runtime, that run is recorded as such and a new
pre-registration decides what replaces it; the grid is not edited after the fact.

## Results

(none yet: added only after the first hosted pre-registered run)

### Dated note, 2026-10-18: results of the first hosted pre-registered run

Added after the run, under "Results" as the status line above allows; nothing above this heading was changed, and the
placeholder line above is kept as committed. The run is the evaluation of record: Colab CLI on a fresh Tesla T4,
commit `e6a7fa2`, notebook blob `eb13bd63`, evidence in `docs/execution-evidence/2026-10-18-e6a7fa2/` and
`docs/release-verification.md` (section *Hosted T4 evaluation of record (2026-10-18)*).

- **Grid (validation PSNR, 48 pairs, epoch 5 kept by all three):** frozen 25.899; C0 26.784; C1 26.833 (+0.049 over
  C0, < 0.10); C2 26.933 (+0.149 over C0, but < 0.10 over C1 on the unrounded scores, although the three-decimal
  difference prints as 0.100). C2 fails the second clause of the selection rule, so **C0 is selected**; C0 beats the
  frozen model by 0.885, so the adapted model is kept. The unrounded scores stayed on the released VM.
- **Fresh test (144 crops, C0):** PSNR frozen 24.508, bicubic 24.868, selected 25.398 dB; vs frozen +0.89 dB
  (95 % CI 0.8168 to 0.967, 144 wins / 0 losses), SSIM +0.02915; vs bicubic +0.5302 dB (0.4614 to 0.6042, 142 / 2),
  SSIM +0.02013. Both verdicts **improved** under amendment 1. Every species gained 0.70 to 1.02 dB.
- **Expected cost, measured** (the table above was an estimate): C0 369.6 s, C1 372.1 s (estimate ≈ 317 s), C2 439.7 s
  (estimate ≈ 580 s, i.e. 1.19× C0 rather than 1.84×); Section 7 in all 1181.7 s; whole hosted session 1615.7 s
  (26.9 min; estimate ≈ 32 min).
- **C2 on a T4 at batch 8:** it fitted. Peak CUDA memory (`torch.cuda.max_memory_allocated`) C2 12,817,431,040 B
  = 11.94 GiB against C0 6,783,353,856 B = 6.32 GiB (1.89×); no out-of-memory. The "unproven" sentence above is answered
  by this note, not edited.
- The package now carries this run as `RECORDED_SELECTION` / `RECORDED_EVALUATION` (commit `5a2bad7`), so a later run
  with identical settings is labelled a reproduction of the pre-registered evaluation.

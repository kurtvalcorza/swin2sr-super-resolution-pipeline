# Pre-registration amendment 1: how the fresh test is read (2026-10-18)

Status: **amendment dated 2026-10-18, committed on its own after the peer review of PR #9 and before the fresh test has
been scored on any runtime.** No fresh-test number exists when this is written: every CPU run so far kept the fresh
test masked (not loaded), and no hosted run of the branch has happened. This amendment adds to
`docs/swin2sr-preregistration.md`. It changes none of the selection rules, configurations, fixed settings or the
fresh-test pin there.

## 1. Reading rule for the fresh test (review F3)

The fresh test is read with paired, per-image statistics over its 144 crops. "Selected" means the system chosen by
the pre-registered grid: the selected config, or the frozen model when the keep-frozen rule applies.

- **Paired differences.** Per crop: PSNR(selected) − PSNR(frozen), and PSNR(selected) − PSNR(bicubic). A crop with a
  non-finite PSNR in either system is left out, and the number left out is reported.
- **Interval.** A 95 % percentile bootstrap interval for the mean of each paired difference:
  - 10,000 resamples, numpy `default_rng(20261018)`;
  - crops are resampled with replacement **within each species**, keeping the species counts (24 each).
- **Verdict.** The comparison is **"improved"** only if both of these hold:
  - the lower bound of the 95 % interval is above 0;
  - the point estimate (the mean paired difference) is at least **0.10 dB**.

  It is **"worse"** if the upper bound is below 0 and the point estimate is at most −0.10 dB. Every other case is
  **"within tolerance"**. These verdicts replace the zero-threshold `> 0` comparisons in Section 8.
- **SSIM.** Reported alongside: the mean, and the same paired bootstrap interval of the SSIM difference. SSIM is not
  used for the verdict.
- **Also reported.** The count of crops the selected system wins (paired difference > 0) against each reference, and
  the per-species means.
- **Exposed development test.** The same rule is applied to it, for consistency. That split remains continuity only.

## 2. An incomplete grid is not the pre-registered procedure (review F1, F6)

- The grid counts as complete only when all three configs (C0, C1, C2) finish in the same Section 7 pass, after the
  latest Section 4 pass, with the current data, split and fixed settings.
- If any config fails (for example a CUDA out-of-memory error in C2), that failure is recorded in the grid file with
  the config, its status and the peak CUDA memory. The run then stops before any fresh-test scoring.
- An incomplete or stale grid is **exploratory**: it never scores the fresh test.
- This is the "recorded as such" outcome promised in the pre-registration. If C2 cannot complete on the hosted T4 at
  batch 8, a new dated pre-registration decides what replaces it.
- **Whether C2 fits a T4 at batch 8 is unproven.** The 15.8 GiB CPU peak is process memory, not CUDA activation
  memory. The hosted run decides, and it logs `torch.cuda.max_memory_allocated()` for every config.
- Look counting within one runtime: the first pre-registered Section 8 pass is the pre-registered evaluation of that
  runtime. Any later pass, including a re-run of Section 8 alone, is a "reproduction" written to `_reproduction_<n>`.
  It never overwrites the main report.

## 3. Corrections to the provenance wording (review F2, F5, F7, F8)

- **F2.** The original status line said the grid was pre-registered "before any grid run". More precisely, the rules
  were written after a reduced CPU timing and memory smoke of C0–C2, and before the full grid or any fresh-test
  score. The smoke used the main `136092f` package, 48 training / 16 validation pairs, 2 epochs and batch 8. Its
  validation PSNRs were C0 26.943, C1 27.059 and C2 27.042. **The 0.10 dB margin was set with those numbers in
  view.** On that reduced split, C1 led C0 by 0.116 dB and C2 led C0 by 0.099 dB. The smoke is not selection
  evidence: it used a reduced split and a 2-epoch budget, and only the full grid selects.
- **F5.** The 0.10 dB margin is a practical threshold chosen by judgement. It is not measured seed-level variation:
  no seed replicate exists. The "twice the last-epoch gain (0.039 dB)" remark describes convergence, not dispersion.
- **F7.** The pinned records (`FRESH_TEST_RECORDS`), not the query, define the fresh test. Re-running the query later
  will not return the same frame. The six frames as listed on 2026-10-18 are committed as
  `docs/fresh-test-frames.json.gz` so the seeded walk can be replayed offline against `sha256_of_ids`. The
  near-duplicate screen catches near-exact re-uploads (re-encodes, ±10 % brightness, ±10 % scale), not crops, flips or
  edits. The observer-disjoint and place-day rules carry most of the protection.
- **F8.** 10 of the 144 fresh observations have ids below 20 million (iNaturalist's early years), when importing
  photos from Flickr was common. For those, an overlap with Flickr2K is slightly less implausible. With about 2,650
  Flickr2K images it stays negligible. No date cutoff is applied (owner ruling).
- **F4.** A gap between the exposed development test and the fresh test mixes two effects:
  - selection optimism on the exposed split;
  - a shift to new photographers and cameras: the fresh test shares no observer with the sample.

  It is read as such, not as one effect.

## Dated note, 2026-10-18 (after the hosted run; the amendment above is unchanged)

- Section 2, "Whether C2 fits a T4 at batch 8 is unproven": the hosted run of `e6a7fa2` / `eb13bd63` answered it. C2
  completed at batch 8 with a peak of 11.94 GiB of CUDA memory (C0 6.32 GiB); the grid was complete, so the incomplete-
  grid rule was not triggered.
- Section 1, the reading rule, was applied to the fresh test in that run: vs frozen +0.89 dB (95 % CI 0.8168 to 0.967),
  vs bicubic +0.5302 dB (0.4614 to 0.6042); both **improved** (CI lower bound > 0 and mean gain >= 0.10 dB).
- Look counting: from commit `5a2bad7` the package names that run as the evaluation of record, so every later pass under
  the pre-registered settings, the first pass of a new runtime included, is a "reproduction" written to
  `_reproduction_<n>`.

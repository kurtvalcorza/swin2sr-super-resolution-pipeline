"""Draw and pin the fresh test (R88-M4): 24 never-scored iNaturalist photographs per species.

Run once by the maintainer; its record is committed as docs/fresh-test-draw.json and the pinned rows as
FRESH_TEST_RECORDS in samples.py. Not run by CI or the notebook.

Query (iNaturalist API v1, generic User-Agent, no account or e-mail sent):
    GET /v1/observations?taxon_name=<scientific name>&quality_grade=research&photo_license=cc0&photos=true
        &order_by=id&order=asc&per_page=200&id_above=<last id>&id_below=<ID_CEILING>
paged until exhausted. That frame (every observation id, recorded with its digest) is shuffled per species with
random.Random(f"{DRAW_SEED}:{label}") and walked in that order; a candidate is accepted when

* its first CC0 photo is served from the open-data bucket as ``medium.<ext>`` with both sides >= HR_CROP and the
  longest side <= MAX_IMAGE_SIDE;
* its observer is new: not among the observers of the 360 pinned photos, of the fleet's other pinned iNaturalist
  photos, or of a photo already accepted (one photo per observer across the whole fresh test);
* its photo id and observation id appear in no fleet pin;
* it is not same-place-same-day with a pinned or already accepted photo (same observed_on date and coordinates
  within 0.01 degrees, or the same place_guess text);
* it is not a confirmed near-duplicate of any pinned photo (train / validation / exposed development test, plus the
  fleet's other photos) or of an accepted one. Two 64-bit dHashes are compared: the repository's 192-px centre crop
  (`samples.centre_crop`), against re-encoded copies (JPEG 30-90, +-10 % brightness: the crop is unchanged in
  geometry), and the whole photo, against those copies and 0.9x / 1.1x rescaled ones (a rescale moves the centre crop,
  so only the whole-photo hash can catch it). Each Hamming threshold is the largest distance between 60 pinned photos
  and their own synthetic copies; a hash hit is excluded only after a content check (normalised cross-correlation of
  the 64-px greyscale images at or above the smallest value seen on those copies). Thresholds come from the synthetic
  copies only, never from the candidates.

usage: python -I tools/draw_fresh_test.py --exclude <fleet_pins.json> --pinned-cache <dir with the 360 + extras>
                                          --work <scratch dir> --out <docs/fresh-test-draw.json>
"""

# ruff: noqa: E501  -- query and record literals are kept on single lines
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swin2sr_super_resolution_pipeline.samples import (  # noqa: E402
    CORPUS_BASE_URL,
    HR_CROP,
    MAX_IMAGE_SIDE,
    SAMPLE_RECORDS,
    SPECIES,
    centre_crop,
)

API = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "dimer-fresh-test-draw/1.0"
DRAW_SEED = 20261018
PER_SPECIES = 24
ID_CEILING = 400_000_000  # observation ids below this; the frame actually listed is recorded by count, max id and digest
SLEEP_S = 1.1


def _get(url: str, *, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
                return response.read()
        except Exception:  # noqa: BLE001 - retried, then re-raised
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def dhash(image: Image.Image) -> int:
    grey = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = list(grey.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def small_grey(image: Image.Image) -> list[float]:
    px = [float(v) for v in image.convert("L").resize((64, 64), Image.Resampling.BILINEAR).getdata()]
    mean = sum(px) / len(px)
    centred = [v - mean for v in px]
    norm = sum(v * v for v in centred) ** 0.5 or 1.0
    return [v / norm for v in centred]


def ncc(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def synthetic_copies(photo: Image.Image) -> list[tuple[bool, Image.Image]]:
    """Whole photos a re-post of the same photograph could plausibly be, flagged True when the geometry is unchanged
    (so the centre crop is comparable)."""
    out = []
    for quality in (30, 50, 70, 90):
        buf = io.BytesIO()
        photo.convert("RGB").save(buf, format="JPEG", quality=quality)
        out.append((True, Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
    for factor in (0.9, 1.1):
        out.append((True, ImageEnhance.Brightness(photo.convert("RGB")).enhance(factor)))
    for scale in (0.9, 1.1):
        size = (round(photo.width * scale), round(photo.height * scale))
        if min(size) >= HR_CROP:
            out.append((False, photo.convert("RGB").resize(size, Image.Resampling.BICUBIC)))
    return out


def list_frame(sci: str, cache: Path) -> list[dict]:
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows, last = [], 0
    while True:
        query = urllib.parse.urlencode(
            {
                "taxon_name": sci,
                "quality_grade": "research",
                "photo_license": "cc0",
                "photos": "true",
                "order_by": "id",
                "order": "asc",
                "per_page": 200,
                "id_above": last,
                "id_below": ID_CEILING,
            }
        )
        page = json.loads(_get(f"{API}?{query}"))["results"]
        time.sleep(SLEEP_S)
        if not page:
            break
        for obs in page:
            photo = next((p for p in obs.get("photos", []) if p.get("license_code") == "cc0"), None)
            rows.append(
                {
                    "obs": obs["id"],
                    "observer": (obs.get("user") or {}).get("login"),
                    "observed_on": obs.get("observed_on"),
                    "location": obs.get("location"),
                    "place_guess": obs.get("place_guess"),
                    "taxon": (obs.get("taxon") or {}).get("name"),
                    "photo": photo["id"] if photo else None,
                    "photo_url": photo.get("url") if photo else None,
                }
            )
        last = page[-1]["id"]
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def lookup_observations(ids: list[int], cache: Path) -> dict[int, dict]:
    if cache.is_file():
        return {int(k): v for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
    out = {}
    for start in range(0, len(ids), 100):
        chunk = ids[start : start + 100]
        query = urllib.parse.urlencode({"id": ",".join(map(str, chunk)), "per_page": 200})
        for obs in json.loads(_get(f"{API}?{query}"))["results"]:
            out[obs["id"]] = {"observed_on": obs.get("observed_on"), "location": obs.get("location"), "place_guess": obs.get("place_guess")}
        time.sleep(SLEEP_S)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def place_keys(meta: dict) -> set[str]:
    keys = set()
    day = meta.get("observed_on")
    if not day:
        return keys
    if meta.get("location"):
        lat, lng = (float(v) for v in str(meta["location"]).split(","))
        keys.add(f"{day}|{round(lat, 2)}|{round(lng, 2)}")
    if meta.get("place_guess"):
        keys.add(f"{day}|{str(meta['place_guess']).strip().lower()}")
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", required=True, help="JSON list of fleet pin rows [id,label,photo,obs,observer,bytes,sha,ext]")
    ap.add_argument("--pinned-cache", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    work = Path(args.work)
    (work / "photos").mkdir(parents=True, exist_ok=True)
    fleet = [tuple(r) for r in json.loads(Path(args.exclude).read_text(encoding="utf-8"))]
    own_sha = {r[6] for r in SAMPLE_RECORDS}
    pinned = list(SAMPLE_RECORDS) + [r for r in fleet if r[6] not in own_sha]
    excl_observers = {r[4] for r in pinned}
    excl_photos = {r[2] for r in pinned} | {r[2] for r in fleet}
    excl_obs = {r[3] for r in pinned} | {r[3] for r in fleet}

    # 1. pinned photos: crop hashes, synthetic-copy calibration, place/day keys
    pinned_hash, pinned_grey, missing = {}, {}, []
    calib_dist, calib_ncc, calib_full, calib_full_ncc = [], [], [], []
    calib_rng = random.Random(DRAW_SEED)
    calib_ids = {(r[0], r[2]) for r in calib_rng.sample(list(SAMPLE_RECORDS), 60)}
    for rid0, _label, pid, _obs, _user, size, digest, _ext in pinned:
        rid = f"{rid0}#{pid}"  # fleet pins reuse record ids for different photos, so key by id and photo
        f = Path(args.pinned_cache) / f"{pid}.jpg"
        data = f.read_bytes() if f.is_file() else b""
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            missing.append(rid)
            continue
        photo = Image.open(io.BytesIO(data))
        photo.load()
        crop = centre_crop(photo, HR_CROP)
        pinned_hash[rid] = (dhash(crop), dhash(photo))
        pinned_grey[rid] = (small_grey(crop), small_grey(photo))
        if (rid0, pid) in calib_ids:
            for same_geometry, copy in synthetic_copies(photo):
                if same_geometry:
                    copy_crop = centre_crop(copy, HR_CROP)
                    calib_dist.append(hamming(pinned_hash[rid][0], dhash(copy_crop)))
                    calib_ncc.append(ncc(pinned_grey[rid][0], small_grey(copy_crop)))
                calib_full.append(hamming(pinned_hash[rid][1], dhash(copy)))
                calib_full_ncc.append(ncc(pinned_grey[rid][1], small_grey(copy)))
    if missing:
        print("pinned photos missing or unverified:", missing, file=sys.stderr)
        return 2
    threshold, ncc_floor = max(calib_dist), min(calib_ncc)
    threshold_full, ncc_floor_full = max(calib_full), min(calib_full_ncc)
    hashes = list(pinned_hash.values())
    unrelated = [hamming(a[0], b[0]) for i, a in enumerate(hashes) for b in hashes[:i]]
    unrelated_full = [hamming(a[1], b[1]) for i, a in enumerate(hashes) for b in hashes[:i]]
    pinned_meta = lookup_observations(sorted({r[3] for r in pinned}), work / "pinned_meta.json")
    taken_places = set()
    for meta in pinned_meta.values():
        taken_places |= place_keys(meta)

    # 2. frames
    frames, frame_digest = {}, {}
    for label, (sci, _common) in SPECIES.items():
        frames[label] = list_frame(sci, work / f"frame_{label}.json")
        ids = ",".join(str(r["obs"]) for r in frames[label])
        frame_digest[label] = {"observations": len(frames[label]), "sha256_of_ids": hashlib.sha256(ids.encode()).hexdigest(), "max_id": max(r["obs"] for r in frames[label])}

    # 3. seeded walk
    accepted, rejected = [], {}
    used_observers, accepted_hash, accepted_grey = set(excl_observers), {}, {}
    hash_hits = []

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for label in SPECIES:
        order = list(frames[label])
        random.Random(f"{DRAW_SEED}:{label}").shuffle(order)
        got = 0
        for row in order:
            if got == PER_SPECIES:
                break
            if not row["photo"] or not row["photo_url"]:
                reject("no cc0 photo")
                continue
            if row["observer"] in used_observers:
                reject("observer already used or pinned")
                continue
            if row["photo"] in excl_photos or row["obs"] in excl_obs:
                reject("photo or observation in a fleet pin")
                continue
            keys = place_keys(row)
            if keys & taken_places:
                reject("same place and day as a pinned or accepted photo")
                continue
            ext = row["photo_url"].rsplit(".", 1)[-1].split("?")[0]
            if ext.lower() not in ("jpg", "jpeg", "png"):
                reject("unsupported extension")
                continue
            local = work / "photos" / f"{row['photo']}.{ext}"
            if not local.is_file():
                try:
                    local.write_bytes(_get(f"{CORPUS_BASE_URL}{row['photo']}/medium.{ext}"))
                except Exception:  # noqa: BLE001
                    reject("not served from the open-data bucket")
                    continue
                time.sleep(0.3)
            data = local.read_bytes()
            try:
                photo = Image.open(io.BytesIO(data))
                photo.load()
            except Exception:  # noqa: BLE001
                reject("undecodable")
                continue
            if min(photo.size) < HR_CROP or max(photo.size) > MAX_IMAGE_SIDE:
                reject("size outside 192 px .. MAX_IMAGE_SIDE")
                continue
            crop = centre_crop(photo, HR_CROP)
            h, g = (dhash(crop), dhash(photo)), (small_grey(crop), small_grey(photo))
            confirmed = False
            for pool_hash, pool_grey in ((pinned_hash, pinned_grey), (accepted_hash, accepted_grey)):
                for other, oh in pool_hash.items():
                    dc, df = hamming(h[0], oh[0]), hamming(h[1], oh[1])
                    if dc <= threshold or df <= threshold_full:
                        cc, cf = ncc(g[0], pool_grey[other][0]), ncc(g[1], pool_grey[other][1])
                        hit = {
                            "candidate_photo": row["photo"], "against": other, "crop_hamming": dc, "full_hamming": df,
                            "crop_ncc": round(cc, 4), "full_ncc": round(cf, 4),
                            "confirmed": (dc <= threshold and cc >= ncc_floor) or (df <= threshold_full and cf >= ncc_floor_full),
                        }
                        hash_hits.append(hit)
                        confirmed = confirmed or hit["confirmed"]
            if confirmed:
                reject("confirmed near-duplicate")
                continue
            rid = f"{label}-fresh-{got:02d}"
            accepted.append([rid, label, row["photo"], row["obs"], row["observer"], len(data), hashlib.sha256(data).hexdigest(), ext])
            accepted_hash[rid], accepted_grey[rid] = h, g
            used_observers.add(row["observer"])
            taken_places |= keys
            got += 1
        print(label, got, flush=True)

    counts = {label: sum(1 for r in accepted if r[1] == label) for label in SPECIES}
    record = {
        "purpose": "fresh test for R88-M4: never scored by any fleet model before the first hosted pre-registered run",
        "drawn": time.strftime("%Y-%m-%d"),
        "api": API,
        "query": {"quality_grade": "research", "photo_license": "cc0", "photos": "true", "order_by": "id", "order": "asc", "per_page": 200, "id_below": ID_CEILING, "taxon_name": {k: v[0] for k, v in SPECIES.items()}},
        "user_agent": USER_AGENT,
        "seed": DRAW_SEED,
        "shuffle": "random.Random(f'{seed}:{label}').shuffle(frame rows in ascending observation id)",
        "per_species_target": PER_SPECIES,
        "per_species_drawn": counts,
        "frame": frame_digest,
        "exclusions": {"pinned_photos_screened": len(pinned), "excluded_observers": len(excl_observers), "excluded_photo_ids": len(excl_photos), "excluded_observation_ids": len(excl_obs), "pinned_place_day_keys": len(taken_places)},
        "near_duplicate_screen": {
            "hash": "64-bit dHash of the 192-px centre crop (samples.centre_crop) and of the whole photo",
            "calibration": f"{len(calib_ids)} pinned photos x their synthetic copies (crop: {len(calib_dist)} same-geometry copies; whole photo: {len(calib_full)} copies incl. 0.9x/1.1x rescales)",
            "hamming_threshold": threshold,
            "synthetic_copy_hamming": {"max": threshold, "median": sorted(calib_dist)[len(calib_dist) // 2]},
            "hamming_threshold_full": threshold_full,
            "synthetic_copy_hamming_full": {"max": threshold_full, "median": sorted(calib_full)[len(calib_full) // 2]},
            "content_check": "normalised cross-correlation of the 64-px greyscale crop / whole photo",
            "ncc_floor": round(ncc_floor, 4),
            "ncc_floor_full": round(ncc_floor_full, 4),
            "unrelated_pinned_pairs_within_threshold": sum(d <= threshold for d in unrelated),
            "unrelated_pinned_pairs_within_threshold_full": sum(d <= threshold_full for d in unrelated_full),
            "unrelated_pinned_pairs": len(unrelated),
            "hash_hits": hash_hits,
        },
        "rejected": rejected,
        "records": accepted,
    }
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: record[k] for k in ("per_species_drawn", "rejected")}), json.dumps({k: v for k, v in record["near_duplicate_screen"].items() if k != "hash_hits"}), "hits", len(hash_hits))
    return 0 if all(v == PER_SPECIES for v in counts.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())

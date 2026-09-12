# Weight provenance and DIMER hosting

- Upstream: `caidas/swin2SR-classical-sr-x2-64`
- Immutable revision: `cee1c923c6a37361c6e5650b65dcf4be821e5d52`
- Weight format: SafeTensors (`model.safetensors`, 48,460,660 bytes)
- Manifest: `weights/swin2sr-x2-64/dimer-base-manifest.json` (4 files, 48,462,226 bytes total, per-file SHA-256)
- Upstream weight license: Apache-2.0
- DIMER hosting: Apache-2.0 permits use, modification, distribution, and commercial use subject to preservation of the license and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision, into the snapshot directory; `verify_snapshot()` then checks every file before any load.
- Loader trust boundary: Transformers `Swin2SRForImageSuperResolution` / `Swin2SRImageProcessor` with `trust_remote_code=False`, `local_files_only=True` from the verified directory.

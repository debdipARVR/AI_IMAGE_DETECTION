# Project: Latent Resonance Image Forensics & ScribeMark Editorial Dashboard

## Architecture
A publication-grade, editorial-styled Streamlit web application for AI Image Detection and Provenance Forensics reproducing the warm editorial parchment aesthetic of ScribeMark, powered by Zero-Shot VAE Latent Resonance and 2D-FFT Azimuthal Spectral Forensics.

- **Frontend**: Streamlit application (`app.py`) governed by a 3-state finite state machine (`landing` -> `processing` -> `results`) with custom CSS injections applying the ScribeMark broadsheet parchment palette and typography.
- **Forensic Engine**: `src/vae_resonance.py` implementing deterministic autoencoder inversion $\hat{x} = \mathcal{D}(\mu(\mathcal{E}(x)))$, spatial residual extraction ($\Delta x = x - \hat{x}$), 2D-FFT periodic lattice peak detection, and azimuthal radial integration $R(r)$.
- **Classification & Calibration**: `src/forensic_classifier.py` providing calibrated 3-tier traffic-light verdicts (`AUTHENTIC OPTICAL PHOTO`, `AI GENERATED DIFFUSION`, `MANIPULATED / RESAMPLED`) and metric thresholds.
- **Evidence Preservation**: `src/pdf_certificate.py` compiling ISO/IEC 27037 and FRE 902(13)/(14) compliant tamper-evident PDF certificates in-memory using ReportLab 5.0.1 with 6-stage SHA-256 chain-of-custody hashes and visual plates.
- **Fast Caching Layer**: In-memory and disk caching (`results/clean_results_cache.pkl` and pre-computed perturbation presets) enabling zero-latency (<0.001s) loading and analysis for all benchmark samples.
- **Verification Harness**: Comprehensive 4-tier automated test suite (`tests/`) executable via `pytest` and headless `streamlit.testing.v1.AppTest`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| 1 | Warm Editorial Parchment CSS | Full ScribeMark palette (`#f0e6c8`, `#e8d9b0`, `#c9b88a`, `#2c1f0e`, `#7a6040`, `#6b4c11`, `#8b2000`, `#4a6b3a`) and typography (`Cinzel`, `Inter`, `JetBrains Mono`, `Newsreader`) | M2 | R1 / UX Survey |
| 2 | Chrome Elimination & Layout | Header/footer suppression, max-width 1100px centered container, marquee dateline banner | M2 | R1 / UX Survey |
| 3 | 3-State App Lifecycle | Atomic state router (`landing` -> `processing` -> `results`) via `st.session_state["app_state"]` | M2 | R1 / UX Survey |
| 4 | Landing View & Presets | Navigation masthead, modal guides, file dropzone (`.png`, `.jpg`, `.jpeg`, `.webp`), 3 one-click preset cards (Authentic, AI, Perturbed), "Begin Scan" button | M2 | R1 / UX Survey |
| 5 | Multi-Pass Terminal Console | 4-pass diagnostic console with progress bar and intermediate metrics (Pass 1: VAE Latent, Pass 2: Spatial, Pass 3: 2D-FFT, Pass 4: ISO 27037) | M2 | R1 / UX Survey |
| 6 | Primary Verdict Card | Broadsheet verdict panel with traffic-light badges (`AUTHENTIC OPTICAL PHOTO`, `AI GENERATED DIFFUSION`, `MANIPULATED / RESAMPLED`) | M2 | R1 / UX Survey |
| 7 | Metric Cards Grid | 4-card grid for PSNR, MSE, Harmonic Spike Ratio, and High-Frequency Energy / Manifold Distance | M2 | R1 / UX Survey |
| 8 | Calibrated Decision Logic | Threshold calibration separating real camera PRNU ($<34.5$ dB) from generative diffusion ($\ge 35.0$ dB) and compression | M1 | R1 / Engine Survey |
| 9 | Preset Cache & Zero Latency | Seamless caching of clean samples and perturbations for instantaneous (<0.001s) demo execution | M1 | R4 / Engine Survey |
| 10 | Offline Standalone Engine | Offline execution of `VAEResonanceEngine` without external APIs or cloud dependencies | M1 | R4 / Engine Survey |
| 11 | Side-by-Side Image Comparison | Visual comparison between input image $x$ and deterministic VAE reconstruction $\hat{x}$ | M3 | R2 / UX Survey |
| 12 | Spatial Residual Heatmap | Full-width $|\Delta x|$ error heatmap with colormap selector (`inferno`, `viridis`, `magma`, `bone`, `turbo`) and gain slider ($1\times - 20\times$) | M3 | R2 / UX Survey |
| 13 | 2D-FFT Spectrum with Harmonics | Centered 2D-FFT residual power spectrum with annotated 8x8 transposed convolution harmonic lattice points | M3 | R2 / UX Survey |
| 14 | Plotly Azimuthal Radial Plot | Responsive Plotly $R(r)$ radial power curve with parchment styling, $1/f^\alpha$ natural optical baseline, and 8x8 lattice markers | M3 | R2 / UX Survey |
| 15 | ISO/IEC 27037 ReportLab Generator | Publication-grade in-memory PDF certificate generation using ReportLab 5.0.1 and `ForensicNumberedCanvas` | M4 | R3 / Spec Miner |
| 16 | 6-Stage SHA-256 Custody Chain | Cryptographic digests of raw file, preprocessed tensor, latent $z$, reconstruction, residual $\Delta x$, and FFT spectrum | M4 | R3 / Spec Miner |
| 17 | Visual Plates PDF Embedding | High-resolution embedded plates (Original, Reconstruction, Residual Heatmap, FFT Spectrum) inside PDF | M4 | R3 / Spec Miner |
| 18 | Digital Verification Seal | HMAC-SHA256 signature seal over canonical JSON manifest complying with FRE 902(13)/(14) | M4 | R3 / Spec Miner |
| 19 | Tier 1 Feature Test Matrix | ≥5 automated tests per feature covering engine, ingestion, inversion, residuals, FFT, azimuthal, PDF, UI | M5 | R4 / Spec Miner |
| 20 | Tier 2 Boundary Test Matrix | ≥5 automated tests per boundary category (small images, extreme aspect ratios, grayscale/alpha, high-res, corrupt files, zero variance, noise) | M5 | R4 / Spec Miner |
| 21 | Tier 3 Pairwise Test Matrix | Orthogonal cross-feature interaction testing (presets, formats, colormaps, gains, exports) | M5 | R4 / Spec Miner |
| 22 | Tier 4 E2E Application Workflows | 5 complete end-to-end user workflows verified via headless Streamlit AppTest and CLI runners | M5 | R4 / Spec Miner |
| 23 | 100% E2E Verification & Gate | Complete test execution pass, review approval, adversarial challenge verification, and forensic audit clean verdict | M6 | Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Engine Calibration & Preset Cache | Implement calibrated classifier, unify preset cache for real/AI/perturbed samples, verify offline zero-latency loading | None | DONE |
| M2 | Editorial Parchment UI & App Flow | Inject ScribeMark CSS tokens, eliminate default chrome, implement 3-state navigation (Landing, Multi-Pass Console, Results) | M1 | IN_PROGRESS (Conv: 51e1986f-6b76-4484-906c-d74374d4f9ea) |
| M3 | Interactive Visual Diagnostic Suite | Side-by-side reconstruction, spatial heatmap with colormap/gain controls, 2D-FFT spectrum with 8x8 harmonics, Plotly azimuthal curve | M2 | IN_PROGRESS (Conv: 51e1986f-6b76-4484-906c-d74374d4f9ea) |
| M4 | ISO/IEC 27037 PDF Certificate Export | In-memory ReportLab PDF generator (`src/pdf_certificate.py`), 6-stage SHA-256 hashes, visual plate embeds, signature seal, download button | M1, M2, M3 | DONE |
| M5 | E2E Testing Suite (Tiers 1-4) | Comprehensive test suite in `tests/` across Tiers 1–4, dual-mode runner (`--fast` / `--integration`), generate `TEST_READY.md` | M1, M2, M3, M4 | DONE |
| M6 | Final Verification & Audit Gate | Run full test suite to 100% pass, execute Reviewers, Challengers, and Forensic Auditor verification | M5 | PLANNED |

## Interface Contracts
### `src/forensic_classifier.py` ↔ `app.py`
- `classify_forensics(spatial_metrics: dict, spectral_metrics: dict) -> dict`:
  - Input: `spatial_metrics` (`mse`, `mae`, `psnr`, `ncc`), `spectral_metrics` (`high_freq_ratio`, `max_harmonic_spike`, `total_spectral_energy`).
  - Output: `{"category": str, "badge_label": str, "badge_color": str, "ai_probability": float, "confidence_str": str, "rationale": str}`.
  - Categories: `"AUTHENTIC OPTICAL PHOTO"`, `"AI GENERATED DIFFUSION"`, `"MANIPULATED / RESAMPLED"`.

### `src/pdf_certificate.py` ↔ `app.py`
- `generate_forensic_certificate(evidence_data: dict, visual_images: dict) -> bytes`:
  - Input: `evidence_data` (case ID, UUID, filename, timestamps, SHA-256 digests, metrics, verdict, seal), `visual_images` (`orig_pil`, `recon_pil`, `residual_pil`, `fft_pil`).
  - Output: `bytes` of valid compiled PDF document.
  - Error Handling: Fallback to standard Helvetica fonts on missing system fonts; handles long text wrapping without overflow.

### Preset Cache ↔ `app.py`
- `load_preset_sample(preset_key: str) -> tuple[PIL.Image, dict]`:
  - Supported keys: `"authentic_camera"`, `"ai_diffusion"`, `"compressed_perturbed"`.
  - Returns raw PIL image and pre-computed analysis dictionary instantly (<0.001s).

## Code Layout
```
c:\books\07_latent_resonance_image_forensics\
├── app.py                         # Production Streamlit UI (ScribeMark Parchment Theme)
├── requirements.txt               # Dependencies
├── PROJECT.md                     # Project master specification and milestones
├── TEST_INFRA.md                  # Test suite architecture and coverage metrics
├── TEST_READY.md                  # Published upon E2E test completion
├── src/
│   ├── vae_resonance.py           # Core VAE Latent Inversion & Spectral Engine
│   ├── forensic_classifier.py     # Calibrated decision logic & traffic-light badges
│   ├── pdf_certificate.py         # ISO/IEC 27037 ReportLab PDF Generator
│   ├── preset_cache.py            # Zero-latency sample preset cache manager
│   └── visualizer.py              # Diagnostic plotting and array normalization
└── tests/
    ├── conftest.py                # Dual-mode fixtures and mock engine
    ├── test_tier1_features/       # Tier 1 Feature Coverage tests
    ├── test_tier2_boundaries/     # Tier 2 Boundary and edge cases
    ├── test_tier3_combinations/   # Tier 3 Pairwise cross-feature tests
    └── test_tier4_e2e/            # Tier 4 Real-world user workflows
```

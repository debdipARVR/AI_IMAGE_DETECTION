# E2E Test Infra: Latent Resonance Image Forensics

## Test Philosophy
- Opaque-box, requirement-driven testing derived from `ORIGINAL_REQUEST.md` and ISO/IEC 27037 standards.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Testing.
- Dual-mode execution: `--fast` for rapid CI verification (<5s) with mocked synthetic telemetry and `--integration` for full model verification.

## Feature Inventory & Test Mapping
| # | Feature | Source | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|:---:|:---:|:---:|
| 1 | Engine Initialization & Offline Cache | R4 / Engine | 5 | N/A | ✓ |
| 2 | Image Ingestion, Resizing & Presets | R1 / Ingestion | 5 | 10 | ✓ |
| 3 | Deterministic VAE Latent Inversion | R1, R2 / VAE | 5 | 5 | ✓ |
| 4 | Spatial Residual Extraction (PSNR/MSE/MAE) | R1, R2 / Spatial | 5 | 5 | ✓ |
| 5 | 2D-FFT Spectral Peak & Energy Analysis | R1, R2 / Spectral | 5 | 5 | ✓ |
| 6 | Azimuthal Radial Power Spectrum R(r) | R2 / Radial | 5 | 5 | ✓ |
| 7 | ISO/IEC 27037 PDF Certificate Generation | R3 / PDF Report | 5 | 5 | ✓ |
| 8 | UI Navigation & 3-State App Lifecycle | R1 / UI State | 5 | 5 | ✓ |

## Test Architecture
- **Test Runner**: `pytest` 9.1.1 executing from project root.
  - Command: `pytest tests/ -v`
  - Fast execution: `pytest tests/ -v --fast`
- **Headless UI Runner**: `streamlit.testing.v1.AppTest` for programmatic state, session, and widget interaction without browser dependencies.
- **Pass/Fail Semantics**: 100% test pass rate (exit code 0), zero unhandled exceptions, all assertions strictly evaluated.
- **Directory Layout**:
  ```
  tests/
  ├── conftest.py
  ├── test_tier1_features/
  │   ├── test_engine.py
  │   ├── test_ingestion.py
  │   ├── test_inversion.py
  │   ├── test_residuals.py
  │   ├── test_fft_spectral.py
  │   ├── test_azimuthal.py
  │   ├── test_pdf_certificate.py
  │   └── test_ui_states.py
  ├── test_tier2_boundaries/
  │   ├── test_small_images.py
  │   ├── test_aspect_ratios.py
  │   ├── test_color_modes.py
  │   ├── test_rgba_alpha.py
  │   ├── test_high_resolution.py
  │   ├── test_corrupted_files.py
  │   ├── test_zero_variance.py
  │   └── test_pure_noise.py
  ├── test_tier3_combinations/
  │   └── test_pairwise_matrix.py
  └── test_tier4_e2e/
      ├── test_e2e_authentic_camera.py
      ├── test_e2e_ai_diffusion.py
      ├── test_e2e_perturbations.py
      ├── test_e2e_session_transitions.py
      └── test_e2e_airgapped_offline.py
  ```

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|---|---|---|
| 1 | Authentic Camera Photo Forensics & Evidence Export | F2, F3, F4, F6, F7, F8 | High |
| 2 | AI Diffusion Synthetic Detection & Harmonic Peak Audit | F2, F3, F4, F5, F7, F8 | High |
| 3 | Lossy JPEG Compression & Resampling Robustness | F2, F4, F5, F8 | Medium |
| 4 | Successive Triage & Session State Isolation | F2, F7, F8 | High |
| 5 | Air-Gapped Standalone Offline Execution Verification | F1, F2, F7, F8 | Medium |

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: ≥40 test cases (≥5 tests across each of the 8 features).
- **Tier 2 (Boundary & Corner Cases)**: ≥40 test cases (≥5 tests across 8 edge condition categories: 1x1 to 128x128 images, extreme aspect ratios, grayscale/palette, RGBA transparency, multi-megapixel high-res, corrupted files, zero variance flat fields, pure noise).
- **Tier 3 (Cross-Feature Combinations)**: ≥10 pairwise orthogonal interaction test cases.
- **Tier 4 (Real-World Application Scenarios)**: 5 end-to-end user workflows.
- **Total Minimum Test Count**: ≥95 test cases.

# TMLR Submission Package: Latent Resonance

**Title:** Latent Resonance: Zero-Shot Autoencoder Inversion and Azimuthal Spectral Forensics for Diffusion Image Attribution  
**Venue:** Transactions on Machine Learning Research (TMLR)  
**Format:** Official TMLR Style (	mlr.sty) — Double-Blind Peer Review Format

---

## Files Included

1. paper.tex: Complete LaTeX source file in TMLR format with double-blind anonymous authors (\usepackage{tmlr}).
2. 	mlr.sty: Official TMLR LaTeX style package.
3. 
eferences.bib: Complete BibTeX citations for Latent Resonance and related literature.
4. igures/: High-resolution figures, including the 6-panel SOTA diagnostic suite (publication_sota_graphic.png).
5. paper.pdf: Standalone compiled PDF rendered with exact TMLR headers, geometry, and styling.
6. generate_tmlr_pdf.py: Python script using ReportLab to compile paper.pdf.

---

## Overleaf Compilation

1. Create a **New Project** on [Overleaf](https://www.overleaf.com/).
2. Select **Upload Project** and upload OVERLEAF_TMLR_LATENT_RESONANCE_PAPER.zip.
3. Set the compiler to **pdfLaTeX** and Main Document to **paper.tex**.
4. Click **Recompile**.

---

## Anonymity & Double-Blind Compliance
- Default: \usepackage{tmlr} sets author to *Anonymous Authors* and header to *Under review as a submission to TMLR*.
- For accepted/camera-ready version: change to \usepackage[accepted]{tmlr}.
- For non-anonymous preprint: change to \usepackage[preprint]{tmlr}.

## 158/258 Assignment 2 – Wine Quality Classification

This repository packages everything needed for Assignment 2:

- Reusable Python modules under `src/goob_ai` that download the UCI Wine Quality data set, engineer labels, train multiple course-aligned models, and emit evaluation artifacts.
- An executed Jupyter notebook at `notebooks/assignment2.ipynb` plus the exported `reports/workbook.html` required for submission.
- Automated tests (`uv run pytest`) to keep the pipeline reliable.

### Environment

```bash
uv sync
source .venv/bin/activate  # or `uv run ...` for ad-hoc commands
```

### Reproducing the analysis

```bash
# 1. Run the experiment headlessly and persist CSV summaries
uv run python -m goob_ai.pipeline

# 2. Execute + export the notebook (produces reports/workbook.html)
uv run jupyter nbconvert \
  --to html \
  --execute notebooks/assignment2.ipynb \
  --output workbook.html \
  --output-dir reports
```

### Testing

```bash
uv run pytest
```

### Submission checklist

- [x] `reports/workbook.html`
- [ ] Record a ~20 minute narrated walkthrough (slides + code). Export to `reports/presentation.mp4`, upload to Drive or YouTube, and replace the placeholder inside `video_url.txt` with the public link before submitting to Gradescope.
- [x] `video_url.txt` placeholder (update with your final URL).

See `presentation_outline.md` for a slide-by-slide script you can adapt while recording the video.


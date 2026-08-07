"""Shared defaults for rasterized PDF / panel output across the project."""

# Uniform publication DPI for matplotlib PNG + PDF panels (inference + QA).
# Matplotlib ``fig.savefig`` from in-memory RGB — never PNG→PDF wrap / upscale.
PDF_EXPORT_DPI = 350

# Alias kept for callers; must stay equal to ``PDF_EXPORT_DPI``.
INFERENCE_PANEL_PDF_DPI = PDF_EXPORT_DPI

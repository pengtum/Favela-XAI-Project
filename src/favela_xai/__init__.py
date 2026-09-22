"""favela_xai: shared code for "Building-scale morphological signatures of informal settlements".

Modules
    config          load config.yaml, resolve repository-relative paths and outputs
    io              read footprints, favela polygons, site windows and shared products
    features        pre-processing, tessellation, the 25 indicators and their registry
    labels          favela / formal-settlement labels and the 2013-2019 overlay status
    model           main per-site XGBoost (Methods 4.4) and validation-pipeline model
    attribution     Tree SHAP, site-level importance, pooled attribution PCA
    boundary        signed distance to the 2013 boundary, distance bins, boundary profiles
    spatial_cv      spatial cross-validation designs and scoring (SI S12)
    stats           Wilson intervals, site bootstrap, Benjamini-Hochberg
    plotting_style  shared matplotlib style and colours
"""
__version__ = "1.0.0"

"""GaugeAlign -- calibration-driven inference-time alignment for multi-camera 3D perception.

First place, AI City Challenge 2026 Track 1 (3D HOTA 56.5447).

The package aligns three interfaces between a pretrained multi-camera detector and
a target site, using only calibration:

    scene_center      the canonical frame functional C-bar = mean camera position
    build_pkl         applies it: t_i' = t_i + R_i * C-bar, and decodes frames
    run_infer         frozen detector + exact gauge inverse + class remap +
                      instance-bank temporal prior
    floor_roi         geometric floor-ROI filter
    postproc          optional causal post-processing (none used in the submission)
    make_submission   validate and package track1.zip
    assoc             CVC-Assoc, a learned per-instance survival gate
                      (exploratory; not part of the submitted system)

The modules are command-line tools first -- see docs/REPRODUCE.md. The one piece
worth importing is the frame functional itself:

    from gaugealign.scene_center import scene_center, camera_centers

Nothing is re-exported at package level on purpose: importing this package stays
free of the heavy dependencies (torch, cv2) that only the inference path needs.
"""

__version__ = "1.0.0"

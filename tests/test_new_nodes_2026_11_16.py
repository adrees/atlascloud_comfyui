"""Metadata-only tests for newly added nodes (2026-11-16).

One new node: `bytedance/seedance-2.5-flash/reference-to-video`, the flash tier
of the existing Seedance 2.5 reference-to-video endpoint already covered by
`bytedance_seedance_2_5_r2v.py`. Its published schema matches the base tier's
(same prompt/duration/resolution/ratio/generate_audio/watermark/return_last_frame/
output_format), except lower reference caps (10 images, 5 videos, 5 audios vs.
30/10/unbounded) — the tests below mainly pin that the model id actually points
at the flash tier and that the flash node did not inherit the base tier's model
id from the file it was derived from.

These tests MUST NOT require ATLASCLOUD_API_KEY.
"""

import inspect

import pytest

from src.atlascloud_comfyui.nodes.video.bytedance_seedance_2_5_flash_r2v import (
    AtlasSeedance25FlashReferenceToVideo,
)

MODEL_ID = "bytedance/seedance-2.5-flash/reference-to-video"
DISPLAY = "AtlasCloud Seedance 2.5 Flash Reference-to-Video"

_DURATIONS = [-1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
_RESOLUTIONS = ["480p", "720p"]
_RATIOS = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"]
_OUTPUT_FORMATS = ["mp4", "mov"]


def test_node_shape():
    inputs = AtlasSeedance25FlashReferenceToVideo.INPUT_TYPES()
    assert "atlas_client" in inputs["required"]
    assert "reference_images" in inputs["required"]
    assert "reference_videos" in inputs["required"]
    assert "prompt" in inputs["optional"]
    assert "reference_audios" in inputs["optional"]
    assert "poll_interval_sec" in inputs["optional"]
    assert "timeout_sec" in inputs["optional"]
    assert AtlasSeedance25FlashReferenceToVideo.RETURN_TYPES == ("STRING", "STRING")
    assert AtlasSeedance25FlashReferenceToVideo.RETURN_NAMES == ("video_url", "prediction_id")
    assert AtlasSeedance25FlashReferenceToVideo.CATEGORY == "AtlasCloud/Video"
    assert AtlasSeedance25FlashReferenceToVideo.FUNCTION == "run"


def test_model_id_is_the_flash_tier():
    source = inspect.getsource(AtlasSeedance25FlashReferenceToVideo.run)
    assert f'"model": "{MODEL_ID}"' in source
    # derived from the base tier node — make sure the base tier id did not survive the copy
    assert '"model": "bytedance/seedance-2.5/reference-to-video"' not in source


def test_enums_match_published_schema():
    optional = AtlasSeedance25FlashReferenceToVideo.INPUT_TYPES()["optional"]
    assert optional["duration"][0] == _DURATIONS
    assert optional["duration"][1]["default"] == 5
    assert optional["resolution"][0] == _RESOLUTIONS
    assert optional["resolution"][1]["default"] == "720p"
    assert optional["ratio"][0] == _RATIOS
    assert optional["ratio"][1]["default"] == "adaptive"
    assert optional["output_format"][0] == _OUTPUT_FORMATS
    assert optional["output_format"][1]["default"] == "mp4"
    assert optional["generate_audio"][1]["default"] is True
    assert optional["watermark"][1]["default"] is False
    assert optional["return_last_frame"][1]["default"] is False


def test_reference_caps_are_the_flash_tier_limits():
    # flash tier caps are lower than the base tier's (10 images/5 videos vs 30/10)
    inputs = AtlasSeedance25FlashReferenceToVideo.INPUT_TYPES()
    assert "up to 10" in inputs["required"]["reference_images"][1]["tooltip"]
    assert "up to 5" in inputs["required"]["reference_videos"][1]["tooltip"]


def test_requires_at_least_one_reference():
    with pytest.raises(RuntimeError, match="Provide at least one reference image or reference video"):
        AtlasSeedance25FlashReferenceToVideo().run(None, "", "")


def test_registered():
    from src.atlascloud_comfyui.registry import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )

    # registry.py imports under the `atlascloud_comfyui.*` path while the tests
    # import under `src.atlascloud_comfyui.*`, so compare by name.
    assert NODE_CLASS_MAPPINGS[DISPLAY].__name__ == AtlasSeedance25FlashReferenceToVideo.__name__
    assert NODE_DISPLAY_NAME_MAPPINGS[DISPLAY] == DISPLAY


def test_readme_lists_new_model():
    import pathlib

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    assert f"| {DISPLAY} | {MODEL_ID} |" in readme.read_text(encoding="utf-8")

import unittest

from pre_experiments.camera_hidden_state_attribution.adaptive_alpha import (
    ALPHA_CHOICES,
    FEATURE_FIELDS,
    build_oracle_labels,
    build_scene_features,
    fit_frozen_selector,
    load_frozen_selector,
    predict_alpha,
)


class AdaptiveAlphaTest(unittest.TestCase):
    def test_scene_features_use_only_predeclared_prediction_fields(self):
        rows = []
        for scene, offset in (("a", 0.0), ("b", 1.0)):
            for frame in range(3):
                rows.append(
                    {
                        "scene": scene,
                        "frame_id": frame,
                        "global_local_token_cosine": offset + frame,
                        "global_local_pose_translation": offset + 2 * frame,
                        "local_local_pose_translation": (
                            None if frame == 0 else offset + 3 * frame
                        ),
                        "global_translation_error_aligned": 1000 + frame,
                    }
                )

        features = build_scene_features(rows)

        self.assertEqual(FEATURE_FIELDS, (
            "global_local_token_cosine_median",
            "global_local_pose_translation_median",
            "local_local_pose_translation_median",
        ))
        self.assertEqual([row["scene"] for row in features], ["a", "b"])
        self.assertEqual(features[0]["global_local_token_cosine_median"], 1.0)
        self.assertEqual(
            features[0]["global_local_pose_translation_median"],
            2.0,
        )
        self.assertEqual(
            features[0]["local_local_pose_translation_median"],
            4.5,
        )
        self.assertFalse(
            any("error" in field for field in features[0])
        )

    def test_oracle_chooses_lowest_delta_and_smaller_alpha_on_tie(self):
        rows = []
        for scene, deltas in (
            ("a", {0.01: -0.1, 0.02: -0.2, 0.05: -0.2}),
            ("b", {0.01: 0.1, 0.02: 0.0, 0.05: -0.1}),
        ):
            for alpha, delta in deltas.items():
                rows.append(
                    {
                        "scene": scene,
                        "condition_family": "selected",
                        "alpha": alpha,
                        "aligned_translation_error_delta": delta,
                    }
                )

        labels = build_oracle_labels(rows)

        self.assertEqual(ALPHA_CHOICES, (0.01, 0.02, 0.05))
        self.assertEqual(labels, {"a": 0.02, "b": 0.05})

    def test_frozen_ridge_selector_is_deterministic_and_authenticated(self):
        features = [
            {
                "scene": "a",
                FEATURE_FIELDS[0]: 0.0,
                FEATURE_FIELDS[1]: 0.0,
                FEATURE_FIELDS[2]: 1.0,
            },
            {
                "scene": "b",
                FEATURE_FIELDS[0]: 1.0,
                FEATURE_FIELDS[1]: 1.0,
                FEATURE_FIELDS[2]: 0.0,
            },
            {
                "scene": "c",
                FEATURE_FIELDS[0]: 2.0,
                FEATURE_FIELDS[1]: 2.0,
                FEATURE_FIELDS[2]: 0.0,
            },
        ]
        labels = {"a": 0.01, "b": 0.02, "c": 0.05}

        frozen = fit_frozen_selector(
            features,
            labels,
            split_digest="split",
            score_run_id="scores",
            replacement_run_id="replacement",
        )
        loaded = load_frozen_selector(
            frozen,
            expected_split_digest="split",
        )

        self.assertEqual(frozen, loaded)
        self.assertNotIn("oracle_alpha_by_scene", frozen)
        self.assertEqual(
            len(frozen["calibration_feature_digest"]),
            64,
        )
        self.assertEqual(
            len(frozen["calibration_label_digest"]),
            64,
        )
        self.assertEqual(
            predict_alpha(loaded, features[0]),
            0.01,
        )
        self.assertIn(
            predict_alpha(loaded, features[2]),
            ALPHA_CHOICES,
        )
        tampered = dict(frozen)
        tampered["ridge"] = 2.0
        with self.assertRaisesRegex(ValueError, "digest"):
            load_frozen_selector(
                tampered,
                expected_split_digest="split",
            )


if __name__ == "__main__":
    unittest.main()

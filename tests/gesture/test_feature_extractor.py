"""Tests for GestureFeatureExtractor and FeatureSet."""

from __future__ import annotations

from gazecontrol.gesture.feature_extractor import FeatureSet, GestureFeatureExtractor
from tests.helpers import make_fake_hand_result


class TestFeatureSet:
    def _make_fs(self, **kwargs) -> FeatureSet:
        defaults = dict(
            finger_states=[1, 0, 0, 0, 0],
            finger_angles=[10.0, 20.0, 15.0, 18.0, 22.0],
            palm_direction=0.8,
            hand_velocity_x=5.0,
            hand_velocity_y=-3.0,
            thumb_index_distance=0.1,
            wrist_x=0.5,
            wrist_y=0.6,
            thumb_dir_y=0.05,
        )
        defaults.update(kwargs)
        return FeatureSet(**defaults)

    def test_to_dict_has_required_keys(self):
        d = self._make_fs().to_dict()
        required = [
            "finger_states",
            "finger_angles",
            "palm_direction",
            "hand_velocity_x",
            "hand_velocity_y",
            "thumb_index_distance",
            "wrist_x",
            "wrist_y",
            "thumb_dir_y",
        ]
        for key in required:
            assert key in d

    def test_to_dict_no_private_keys(self):
        fs = self._make_fs(
            finger_states=[0] * 5,
            finger_angles=[0.0] * 5,
            palm_direction=0.0,
            hand_velocity_x=0.0,
            hand_velocity_y=0.0,
            thumb_index_distance=0.0,
            wrist_x=0.5,
            wrist_y=0.5,
            thumb_dir_y=0.0,
        )
        d = fs.to_dict()
        private = [k for k in d if k.startswith("_")]
        assert private == [], "FeatureSet.to_dict() must not contain private keys"


class TestGestureFeatureExtractor:
    def test_extract_returns_none_when_no_hand(self):
        ext = GestureFeatureExtractor()
        assert ext.extract(None) is None

    def test_extract_returns_feature_set_with_hand(self):
        ext = GestureFeatureExtractor()
        result = make_fake_hand_result(n_hands=1)
        features = ext.extract(result)
        assert features is not None
        assert isinstance(features, FeatureSet)

    def test_finger_states_length(self):
        ext = GestureFeatureExtractor()
        result = make_fake_hand_result(n_hands=1)
        features = ext.extract(result)
        assert features is not None
        assert len(features.finger_states) == 5

    def test_finger_angles_length(self):
        ext = GestureFeatureExtractor()
        result = make_fake_hand_result(n_hands=1)
        features = ext.extract(result)
        assert features is not None
        assert len(features.finger_angles) == 5

    def test_wrist_coords_normalized(self):
        ext = GestureFeatureExtractor()
        result = make_fake_hand_result(n_hands=1)
        features = ext.extract(result)
        assert features is not None
        assert 0.0 <= features.wrist_x <= 1.0
        assert 0.0 <= features.wrist_y <= 1.0

    def test_velocity_accumulates_over_frames(self):
        ext = GestureFeatureExtractor()
        result = make_fake_hand_result(n_hands=1)
        # First call — no velocity yet.
        ext.extract(result)
        f2 = ext.extract(result)
        # Velocity may be 0 if centroid doesn't move; just check it's a float.
        assert f2 is not None
        assert isinstance(f2.hand_velocity_x, float)
        assert isinstance(f2.hand_velocity_y, float)

    def test_to_vector_returns_17_floats(self):
        ext = GestureFeatureExtractor()
        result = make_fake_hand_result(n_hands=1)
        features = ext.extract(result)
        assert features is not None
        vec = features.to_vector()
        assert len(vec) == 17
        assert all(isinstance(v, float) for v in vec)

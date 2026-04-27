"""Test per FramePreprocessor."""

import numpy as np

from gazecontrol.capture.frame_preprocessor import FramePreprocessor, FrameQuality


def _random_frame(w=640, h=480):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _sharp_frame(w=640, h=480):
    """Frame con testo nitido su sfondo grigio (alta varianza Laplaciana, buona luminosità)."""
    import cv2

    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.putText(frame, "SHARP TEXT", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
    cv2.putText(frame, "SHARP TEXT", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return frame


def _blurry_frame(w=640, h=480):
    """Frame uniformemente grigio (bassa varianza Laplaciana)."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_process_returns_correct_types():
    p = FramePreprocessor()
    frame = _random_frame()
    enhanced, quality = p.process(frame)
    assert isinstance(enhanced, np.ndarray)
    assert isinstance(quality, FrameQuality)
    assert enhanced.shape == frame.shape
    assert enhanced.dtype == np.uint8


def test_sharp_frame_is_usable():
    p = FramePreprocessor(blur_threshold=30.0)
    frame = _sharp_frame()
    _, quality = p.process(frame)
    assert quality.is_usable, f"Frame nitido non riconosciuto: var={quality.laplacian_var:.1f}"


def test_blurry_frame_is_not_usable():
    p = FramePreprocessor(blur_threshold=30.0)
    frame = _blurry_frame()
    _, quality = p.process(frame)
    assert not quality.is_usable, f"Frame sfocato non rilevato: var={quality.laplacian_var:.1f}"


def test_clahe_increases_local_contrast():
    """CLAHE deve aumentare o mantenere la varianza delle intensità."""
    import cv2

    p = FramePreprocessor(sharpen=False)
    # Frame con basso contrasto
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    # Aggiungi pattern molto debole
    frame[100:200, 100:200] = 110
    enhanced, _ = p.process(frame)
    var_orig = np.var(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float))
    var_enh = np.var(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(float))
    assert var_enh >= var_orig, f"CLAHE ha ridotto il contrasto: {var_orig:.2f} → {var_enh:.2f}"


def test_dark_frame_not_usable():
    p = FramePreprocessor(brightness_min=30.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, quality = p.process(frame)
    assert not quality.is_usable


def test_overexposed_frame_not_usable():
    p = FramePreprocessor(brightness_max=230.0)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    _, quality = p.process(frame)
    assert not quality.is_usable


if __name__ == "__main__":
    test_process_returns_correct_types()
    test_sharp_frame_is_usable()
    test_blurry_frame_is_not_usable()
    test_clahe_increases_local_contrast()
    test_dark_frame_not_usable()
    test_overexposed_frame_not_usable()
    print("Tutti i test FramePreprocessor superati.")

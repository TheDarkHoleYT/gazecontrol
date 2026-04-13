"""
Benchmark accuratezza Eye Tracking — misura angular error e jitter.

Mostra N punti a schermo in sequenza, raccoglie le predizioni gaze,
calcola le metriche di accuratezza.

Metriche calcolate:
  - MAE (px)    : errore medio assoluto in pixel
  - Angular err : errore in gradi angolari (a 60cm di distanza stimata)
  - Precision   : deviazione standard durante la fixation (jitter)
  - Accuracy 1° : % di campioni entro 1° dal target

Output: stampa a console + salva report JSON in profiles/benchmark_<timestamp>.json

Uso:
    python tools/benchmark_gaze.py --profile default
    python tools/benchmark_gaze.py --profile default --points 13 --dwell 2.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

import gazecontrol.config as config
from eyetrax import GazeEstimator
from gazecontrol.capture.frame_grabber import FrameGrabber
from gazecontrol.capture.frame_preprocessor import FramePreprocessor
from gazecontrol.gaze.one_euro_filter import OneEuroFilter
from gazecontrol.gaze.fixation_detector import FixationDetector


# Pixel per grado a 60cm, monitor 24" FHD: ~ 44 px/deg
PX_PER_DEG = 44.0

# Layout punti di benchmark: griglia 5x3 con esclusione angoli estremi
def _make_points(screen_w: int, screen_h: int, n: int = 13) -> list[tuple[int, int]]:
    """Genera punti di test distribuiti uniformemente escludendo i 10% di bordo."""
    margin_x = int(screen_w * 0.10)
    margin_y = int(screen_h * 0.10)
    w = screen_w - 2 * margin_x
    h = screen_h - 2 * margin_y

    if n <= 9:
        xs = np.linspace(margin_x, margin_x + w, 3, dtype=int)
        ys = np.linspace(margin_y, margin_y + h, 3, dtype=int)
    else:
        cols = 4
        rows = (n + cols - 1) // cols
        xs = np.linspace(margin_x, margin_x + w, cols, dtype=int)
        ys = np.linspace(margin_y, margin_y + h, rows, dtype=int)

    pts = [(int(x), int(y)) for y in ys for x in xs]
    np.random.shuffle(pts)
    return pts[:n]


def run_benchmark(profile_name: str, n_points: int = 13, dwell_s: float = 1.5):
    """Esegue il benchmark e ritorna le metriche."""
    import ctypes
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    # Setup moduli
    grabber = FrameGrabber()
    if not grabber.start():
        print("[ERRORE] Impossibile aprire la camera.")
        sys.exit(1)

    preprocessor = FramePreprocessor()
    filter_x = OneEuroFilter(freq=config.CAMERA_FPS,
                              min_cutoff=config.GAZE_1EURO_MIN_CUTOFF,
                              beta=config.GAZE_1EURO_BETA)
    filter_y = OneEuroFilter(freq=config.CAMERA_FPS,
                              min_cutoff=config.GAZE_1EURO_MIN_CUTOFF,
                              beta=config.GAZE_1EURO_BETA)

    estimator = GazeEstimator(
        model_name='tiny_mlp',
        model_kwargs={'hidden_layer_sizes': (256, 128, 64), 'max_iter': 1000},
    )
    profile_path = os.path.join(config.PROFILES_DIR, f'{profile_name}.pkl')
    if not os.path.exists(profile_path):
        print(f"[ERRORE] Profilo '{profile_name}' non trovato.")
        grabber.stop()
        sys.exit(1)
    estimator.load_model(profile_path)

    points = _make_points(screen_w, screen_h, n_points)

    # Crea finestra fullscreen per visualizzare i target
    win_name = "GazeControl Benchmark"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    all_errors = []
    point_results = []

    print(f"\nBenchmark: {n_points} punti, {dwell_s:.1f}s per punto")
    print("Guarda ogni punto verde finché non scompare.\n")

    for i, (tx, ty) in enumerate(points):
        print(f"  Punto {i+1}/{n_points}: ({tx}, {ty})")
        collected_gaze: list[tuple[float, float]] = []

        t_start = time.monotonic()
        t_collect = t_start + dwell_s * 0.4  # prima metà: attesa; seconda: raccolta

        while True:
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            elapsed = time.monotonic() - t_start
            collecting = elapsed >= dwell_s * 0.4

            # Disegna target
            radius = 20 - int(10 * min(elapsed / dwell_s, 1.0))
            color = (0, 255, 0) if not collecting else (0, 200, 255)
            cv2.circle(canvas, (tx, ty), radius, color, -1)
            cv2.circle(canvas, (tx, ty), radius + 4, (255, 255, 255), 1)

            # Info
            msg = f"Punto {i+1}/{n_points} — guarda il punto verde"
            cv2.putText(canvas, msg, (20, screen_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

            ok_bgr, frame_bgr = grabber.read_bgr()
            if ok_bgr:
                enhanced, quality = preprocessor.process(frame_bgr)
                feats, blink = estimator.extract_features(enhanced)
                if feats is not None and not blink and quality.is_usable:
                    px, py = estimator.predict([feats])[0]
                    fx = filter_x.filter(float(px))
                    fy = filter_y.filter(float(py))
                    cv2.circle(canvas, (int(fx), int(fy)), 6, (0, 0, 255), -1)
                    if collecting:
                        collected_gaze.append((fx, fy))

            cv2.imshow(win_name, canvas)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                grabber.stop()
                return None

            if elapsed >= dwell_s:
                break

        if collected_gaze:
            xs = [g[0] for g in collected_gaze]
            ys = [g[1] for g in collected_gaze]
            mean_x, mean_y = np.mean(xs), np.mean(ys)
            error_px = np.hypot(mean_x - tx, mean_y - ty)
            precision_px = np.mean(np.hypot(
                np.array(xs) - mean_x, np.array(ys) - mean_y
            ))
            all_errors.append(error_px)
            point_results.append({
                'target': (tx, ty),
                'mean_gaze': (round(mean_x, 1), round(mean_y, 1)),
                'error_px': round(error_px, 1),
                'precision_px': round(precision_px, 1),
                'n_samples': len(collected_gaze),
            })
            print(f"    Errore: {error_px:.1f}px ({error_px/PX_PER_DEG:.2f}°) | "
                  f"Precision: {precision_px:.1f}px | N={len(collected_gaze)}")
        else:
            print(f"    [WARN] Nessun dato raccolto")

    cv2.destroyAllWindows()
    grabber.stop()

    if not all_errors:
        return None

    # Calcola metriche finali
    errors = np.array(all_errors)
    mae_px = float(np.mean(errors))
    angular_error = mae_px / PX_PER_DEG
    pct_1deg = float(np.mean(errors < PX_PER_DEG * 1.0) * 100)
    pct_2deg = float(np.mean(errors < PX_PER_DEG * 2.0) * 100)

    report = {
        'timestamp': datetime.now().isoformat(),
        'profile': profile_name,
        'n_points': n_points,
        'screen': (screen_w, screen_h),
        'mae_px': round(mae_px, 1),
        'angular_error_deg': round(angular_error, 2),
        'accuracy_1deg_pct': round(pct_1deg, 1),
        'accuracy_2deg_pct': round(pct_2deg, 1),
        'points': point_results,
    }

    print(f"\n{'='*50}")
    print(f"RISULTATI BENCHMARK")
    print(f"{'='*50}")
    print(f"  MAE:           {mae_px:.1f} px")
    print(f"  Angular error: {angular_error:.2f}°")
    print(f"  Accuracy <1°:  {pct_1deg:.0f}%")
    print(f"  Accuracy <2°:  {pct_2deg:.0f}%")
    print(f"{'='*50}\n")

    # Salva report
    os.makedirs(config.PROFILES_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(config.PROFILES_DIR, f'benchmark_{ts}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report salvato: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description='GazeControl Accuracy Benchmark')
    parser.add_argument('--profile', default='default')
    parser.add_argument('--points', type=int, default=13,
                        help='Numero di punti di test (default: 13)')
    parser.add_argument('--dwell', type=float, default=1.5,
                        help='Secondi per punto (default: 1.5)')
    args = parser.parse_args()

    run_benchmark(args.profile, args.points, args.dwell)


if __name__ == '__main__':
    main()

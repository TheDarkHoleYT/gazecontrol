"""Trova quale indice camera funziona su questo sistema."""
import cv2
import subprocess

print("=== Test per indice (0-5) ===")
found = False
for i in range(6):
    for backend, name in [(cv2.CAP_DSHOW, 'DSHOW'), (cv2.CAP_MSMF, 'MSMF'), (cv2.CAP_ANY, 'ANY')]:
        cap = cv2.VideoCapture(i, backend)
        ok = cap.isOpened()
        if ok:
            ret, frame = cap.read()
            print(f"  Camera {i} [{name}]: APERTA, frame={'OK' if (ret and frame is not None) else 'FAIL'}")
            found = True
        cap.release()
        if ok:
            break
    else:
        pass  # silenzioso se non trovata

if not found:
    print("  Nessuna camera trovata per indice.")

print("\n=== Elenco device video da Windows (wmic) ===")
try:
    result = subprocess.run(
        ['wmic', 'path', 'Win32_PnPEntity', 'where', "Caption like '%camera%' or Caption like '%webcam%' or Caption like '%video%'", 'get', 'Caption,Status'],
        capture_output=True, text=True, timeout=10
    )
    print(result.stdout.strip() or "  Nessun risultato")
except Exception as e:
    print(f"  wmic non disponibile: {e}")

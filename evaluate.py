import csv, os, sys
import numpy as np
from glob import glob

LOG_DIR = "logs"

def load_log(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def analyze(rows):
    total         = len(rows)
    if total == 0:
        print("Empty log."); return

    fps_vals      = [float(r['fps']) for r in rows]
    contact_vals  = [int(r['contact_active']) for r in rows]
    flickers      = int(rows[-1]['flickers_so_far'])
    fp_frames     = sum(1 for r in rows
                        if int(r['contact_active']) and int(r['objects']) == 0)

    contact_pct   = 100 * sum(contact_vals) / total
    avg_fps       = np.mean(fps_vals)
    min_fps       = np.min(fps_vals)
    flicker_rate  = flickers / max(total / 30, 1)  # per second assuming 30fps

    # Stability score — lower flicker rate = more stable (0-100)
    stability     = max(0, 100 - int(flicker_rate * 10))

    print("\n═══════════════════════════════════════")
    print("         EVALUATION REPORT             ")
    print("═══════════════════════════════════════")
    print(f"  Total frames:        {total}")
    print(f"  Avg FPS:             {avg_fps:.1f}")
    print(f"  Min FPS:             {min_fps:.1f}")
    print(f"  Contact active:      {contact_pct:.1f}% of session")
    print(f"  Total flickers:      {flickers}")
    print(f"  Flicker rate:        {flicker_rate:.2f}/sec")
    print(f"  False pos frames:    {fp_frames}")
    print(f"  Stability score:     {stability}/100")
    print("───────────────────────────────────────")

    # Health assessment
    if stability >= 80:
        print("  ✅ Detection stability: GOOD")
    elif stability >= 50:
        print("  ⚠️  Detection stability: MODERATE — consider tuning threshold")
    else:
        print("  ❌ Detection stability: POOR — flickering too high")

    if avg_fps >= 25:
        print("  ✅ FPS: Real-time capable")
    elif avg_fps >= 15:
        print("  ⚠️  FPS: Acceptable but borderline")
    else:
        print("  ❌ FPS: Too slow for real-time use")

    if fp_frames < total * 0.05:
        print("  ✅ False positives: Low")
    elif fp_frames < total * 0.15:
        print("  ⚠️  False positives: Moderate")
    else:
        print("  ❌ False positives: High — lower confidence threshold")

    print("═══════════════════════════════════════\n")

# ── Run on latest log or specified file ──────────────────────────────────────
if len(sys.argv) > 1:
    log_path = sys.argv[1]
else:
    logs = sorted(glob(f"{LOG_DIR}/session_*.csv"))
    if not logs:
        print("No logs found in logs/ folder. Run live_contact_a1.py first.")
        sys.exit()
    log_path = logs[-1]
    print(f"Analyzing latest log: {log_path}")

rows = load_log(log_path)
analyze(rows)
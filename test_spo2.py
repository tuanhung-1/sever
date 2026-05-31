#!/usr/bin/env python3
"""
Test script để tính SpO2 từ PPG window payload.
"""

import json
import sys
import time
from model import from_json_samples

# Helper function (from app.py)
def _to_non_negative_number(value):
    try:
        v = float(value)
        return v if v >= 0.0 else None
    except (TypeError, ValueError):
        return None


def normalize_raw_payload_for_api(raw_payload):
    """Extract IR/RED from data array and normalize payload (from app.py logic)."""
    if not isinstance(raw_payload, dict):
        raise ValueError("Payload khong hop le, can doi tuong JSON")

    if isinstance(raw_payload.get("data"), list) and raw_payload.get("data"):
        series = raw_payload.get("data")
        times = []
        ir_vals = []
        red_vals = []
        for item in series:
            if not isinstance(item, dict):
                continue
            t = item.get("t") if "t" in item else item.get("ts", item.get("timestamp"))
            if t is not None:
                try:
                    times.append(int(t))
                except Exception:
                    times.append(None)
            else:
                times.append(None)

            ir_vals.append(int(item.get("ir")) if item.get("ir") is not None else 0)
            red_vals.append(int(item.get("red")) if item.get("red") is not None else 0)

        ts0_int = None
        ts_int = None
        valid_times = [t for t in times if isinstance(t, int)]
        if valid_times:
            ts0_int = valid_times[0]
            ts_int = valid_times[-1]

        fs = raw_payload.get("fs")
        sample_interval_ms = None
        try:
            if fs is not None:
                f = float(fs)
                if f > 0:
                    fs_interval_ms = max(1, int(round(1000.0 / f)))
                else:
                    fs_interval_ms = None
            else:
                fs_interval_ms = None
        except Exception:
            fs_interval_ms = None

        sample_interval_ms_time = None
        try:
            if len(valid_times) > 1:
                import numpy as np
                diffs = np.diff(np.array(valid_times, dtype=np.int64))
                diffs = diffs[diffs > 0]
                if diffs.size > 0:
                    sample_interval_ms_time = int(max(1, int(np.median(diffs))))
        except Exception:
            sample_interval_ms_time = None

        if sample_interval_ms_time is not None:
            sample_interval_ms = sample_interval_ms_time
            if fs_interval_ms is not None and fs_interval_ms > 0:
                diff_frac = abs(sample_interval_ms_time - fs_interval_ms) / float(fs_interval_ms)
                if diff_frac > 0.2:
                    print(f"⚠️  fs ({fs_interval_ms}ms) and timestamps differ by {diff_frac*100:.0f}% - using timestamps")
        else:
            sample_interval_ms = fs_interval_ms

        return {
            "ts": ts_int if ts_int is not None else int(time.time() * 1000),
            "timestamp": ts_int if ts_int is not None else int(time.time() * 1000),
            "ts0": ts0_int,
            "temp": _to_non_negative_number(raw_payload.get("temp")),
            "sample_interval_ms": sample_interval_ms,
            "heart_rate": raw_payload.get("heart_rate", raw_payload.get("bpm")),
            "spo2": raw_payload.get("spo2"),
            "ir": ir_vals,
            "red": red_vals,
        }

    return raw_payload

# ════════════════════════════════════════════════════════════════════════════════
# PAYLOAD TEST
# ════════════════════════════════════════════════════════════════════════════════

test_payload = {
    "type": "ppg_window",
    "fs": 100,
    "window_size": 200,
    "step_size": 50,
    "temp": 33.65,
    "quality": {
        "status": "GOOD",
        "valid": True,
        "ir_mean": 121919,
        "red_mean": 100861,
        "ir_p2p": 3858,
        "red_p2p": 1466,
        "ir_acdc": 0.03164,
        "red_acdc": 0.01453,
        "max_jerk": 1.83,
        "bad_motion_rate": 1.696
    },
    "data": [
        {"t": 59195, "ir": 122687, "red": 101285},
        {"t": 59235, "ir": 122207, "red": 101137},
        {"t": 59275, "ir": 121450, "red": 100883},
        {"t": 59315, "ir": 121250, "red": 100805},
        {"t": 59355, "ir": 121387, "red": 100856},
        {"t": 59395, "ir": 121423, "red": 100877},
        {"t": 59435, "ir": 121506, "red": 100898},
        {"t": 59475, "ir": 121713, "red": 100954},
        {"t": 59515, "ir": 121859, "red": 100980},
        {"t": 59555, "ir": 121975, "red": 101021},
        {"t": 59595, "ir": 122106, "red": 101043},
        {"t": 59635, "ir": 122239, "red": 101077},
        {"t": 59675, "ir": 122372, "red": 101107},
        {"t": 59715, "ir": 122490, "red": 101151},
        {"t": 59755, "ir": 122591, "red": 101211},
        {"t": 59795, "ir": 122653, "red": 101247},
        {"t": 59835, "ir": 122697, "red": 101256},
        {"t": 59875, "ir": 122769, "red": 101295},
        {"t": 59915, "ir": 122834, "red": 101309},
        {"t": 59955, "ir": 122859, "red": 101305},
        {"t": 59995, "ir": 122864, "red": 101273},
        {"t": 60035, "ir": 122856, "red": 101254},
        {"t": 60075, "ir": 122735, "red": 101180},
        {"t": 60115, "ir": 122459, "red": 101069},
        {"t": 60155, "ir": 122378, "red": 101054},
        {"t": 60195, "ir": 122569, "red": 101116},
        {"t": 60235, "ir": 122683, "red": 101141},
        {"t": 60275, "ir": 122768, "red": 101181},
        {"t": 60315, "ir": 122885, "red": 101211},
        {"t": 60355, "ir": 122955, "red": 101231},
        {"t": 60395, "ir": 123036, "red": 101267},
        {"t": 60435, "ir": 123127, "red": 101293},
        {"t": 60475, "ir": 123242, "red": 101326},
        {"t": 60515, "ir": 123349, "red": 101364},
        {"t": 60555, "ir": 123409, "red": 101413},
        {"t": 60595, "ir": 123394, "red": 101427},
        {"t": 60635, "ir": 123366, "red": 101403},
        {"t": 60675, "ir": 123387, "red": 101402},
        {"t": 60715, "ir": 123351, "red": 101390},
        {"t": 60755, "ir": 123348, "red": 101383},
        {"t": 60795, "ir": 123307, "red": 101365},
        {"t": 60835, "ir": 123064, "red": 101290},
        {"t": 60875, "ir": 122604, "red": 101133},
        {"t": 60915, "ir": 122523, "red": 101108},
        {"t": 60935, "ir": 122684, "red": 101151},
        {"t": 60995, "ir": 122784, "red": 101170},
        {"t": 61035, "ir": 122853, "red": 101200},
        {"t": 61075, "ir": 122921, "red": 101216},
        {"t": 61115, "ir": 122938, "red": 101231},
        {"t": 61155, "ir": 122933, "red": 101230},
        {"t": 61195, "ir": 122980, "red": 101245},
        {"t": 61235, "ir": 123058, "red": 101259},
        {"t": 61275, "ir": 123134, "red": 101275},
        {"t": 61315, "ir": 123192, "red": 101279},
        {"t": 61355, "ir": 123239, "red": 101307},
        {"t": 61395, "ir": 123264, "red": 101288},
        {"t": 61435, "ir": 123289, "red": 101304},
        {"t": 61475, "ir": 123310, "red": 101308},
        {"t": 61515, "ir": 123330, "red": 101324},
        {"t": 61555, "ir": 123349, "red": 101325},
        {"t": 61595, "ir": 123310, "red": 101330},
        {"t": 61635, "ir": 122911, "red": 101207},
        {"t": 61675, "ir": 122267, "red": 100995},
        {"t": 61715, "ir": 122167, "red": 100963},
        {"t": 61755, "ir": 122316, "red": 101029},
        {"t": 61795, "ir": 122446, "red": 101064},
        {"t": 61835, "ir": 122578, "red": 101105},
        {"t": 61875, "ir": 122741, "red": 101153},
        {"t": 61915, "ir": 122807, "red": 101165},
        {"t": 61955, "ir": 122827, "red": 101176},
        {"t": 61995, "ir": 122817, "red": 101178},
        {"t": 62035, "ir": 122733, "red": 101177},
        {"t": 62075, "ir": 122716, "red": 101182},
        {"t": 62115, "ir": 122797, "red": 101196},
        {"t": 62155, "ir": 122882, "red": 101213},
        {"t": 62195, "ir": 122954, "red": 101224},
        {"t": 62235, "ir": 123003, "red": 101243},
        {"t": 62275, "ir": 123044, "red": 101257},
        {"t": 62315, "ir": 123070, "red": 101260},
        {"t": 62355, "ir": 123100, "red": 101280},
        {"t": 62395, "ir": 123154, "red": 101279},
        {"t": 62435, "ir": 123195, "red": 101296},
        {"t": 62475, "ir": 123006, "red": 101227},
        {"t": 62505, "ir": 122146, "red": 100934},
        {"t": 62555, "ir": 121609, "red": 100733},
        {"t": 62595, "ir": 121774, "red": 100790},
        {"t": 62635, "ir": 122051, "red": 100885},
        {"t": 62675, "ir": 122147, "red": 100925},
        {"t": 62715, "ir": 122334, "red": 100988},
        {"t": 62755, "ir": 122409, "red": 101013},
        {"t": 62795, "ir": 122321, "red": 100990},
        {"t": 62835, "ir": 121957, "red": 100878},
        {"t": 62875, "ir": 121689, "red": 100801},
        {"t": 62915, "ir": 121552, "red": 100781},
        {"t": 62935, "ir": 121503, "red": 100775},
        {"t": 62995, "ir": 121513, "red": 100781},
        {"t": 63035, "ir": 121590, "red": 100809},
        {"t": 63075, "ir": 121776, "red": 100850},
        {"t": 63115, "ir": 121978, "red": 100925},
        {"t": 63155, "ir": 122191, "red": 100978},
        {"t": 63195, "ir": 122390, "red": 101044},
        {"t": 63235, "ir": 122508, "red": 101081},
        {"t": 63275, "ir": 122631, "red": 101104},
        {"t": 63315, "ir": 122746, "red": 101112},
        {"t": 63355, "ir": 122343, "red": 100974},
        {"t": 63395, "ir": 121223, "red": 100556},
        {"t": 63435, "ir": 120723, "red": 100364},
        {"t": 63475, "ir": 120785, "red": 100388},
        {"t": 63515, "ir": 120936, "red": 100457},
        {"t": 63555, "ir": 121030, "red": 100497},
        {"t": 63595, "ir": 121248, "red": 100579},
        {"t": 63635, "ir": 121444, "red": 100652},
        {"t": 63675, "ir": 121625, "red": 100718},
        {"t": 63715, "ir": 121767, "red": 100777},
        {"t": 63755, "ir": 121746, "red": 100775},
        {"t": 63795, "ir": 121766, "red": 100802},
        {"t": 63835, "ir": 121911, "red": 100853},
        {"t": 63875, "ir": 122143, "red": 100945},
        {"t": 63915, "ir": 122264, "red": 101011},
        {"t": 63955, "ir": 122324, "red": 101032},
        {"t": 63995, "ir": 122394, "red": 101055},
        {"t": 64035, "ir": 122402, "red": 101068},
        {"t": 64075, "ir": 122409, "red": 101073},
        {"t": 64105, "ir": 122435, "red": 101079},
        {"t": 64145, "ir": 122454, "red": 101073},
        {"t": 64185, "ir": 122455, "red": 101053},
        {"t": 64225, "ir": 122417, "red": 101033},
        {"t": 64265, "ir": 122372, "red": 101022},
        {"t": 64305, "ir": 122174, "red": 100942},
        {"t": 64345, "ir": 120963, "red": 100530},
        {"t": 64385, "ir": 120173, "red": 100260},
        {"t": 64425, "ir": 120063, "red": 100213},
        {"t": 64465, "ir": 120187, "red": 100232},
        {"t": 64505, "ir": 120253, "red": 100235},
        {"t": 64545, "ir": 120410, "red": 100276},
        {"t": 64585, "ir": 120587, "red": 100352},
        {"t": 64625, "ir": 120658, "red": 100375},
        {"t": 64665, "ir": 120595, "red": 100378},
        {"t": 64705, "ir": 120529, "red": 100365},
        {"t": 64745, "ir": 120516, "red": 100352},
        {"t": 64785, "ir": 120574, "red": 100362},
        {"t": 64825, "ir": 120683, "red": 100397},
        {"t": 64865, "ir": 120763, "red": 100414},
        {"t": 64905, "ir": 120902, "red": 100454},
        {"t": 64935, "ir": 121116, "red": 100523},
        {"t": 64985, "ir": 121316, "red": 100598},
        {"t": 65025, "ir": 121309, "red": 100611},
        {"t": 65065, "ir": 121331, "red": 100616},
        {"t": 65105, "ir": 121414, "red": 100647},
        {"t": 65145, "ir": 121507, "red": 100674},
        {"t": 65185, "ir": 121652, "red": 100729},
        {"t": 65225, "ir": 121777, "red": 100780},
        {"t": 65265, "ir": 121856, "red": 100803},
        {"t": 65305, "ir": 121203, "red": 100581},
        {"t": 65345, "ir": 119989, "red": 100130},
        {"t": 65385, "ir": 119557, "red": 99967},
        {"t": 65425, "ir": 119551, "red": 99961},
        {"t": 65465, "ir": 119682, "red": 99992},
        {"t": 65505, "ir": 119817, "red": 100032},
        {"t": 65535, "ir": 120105, "red": 100127},
        {"t": 65585, "ir": 120301, "red": 100191},
        {"t": 65625, "ir": 120388, "red": 100205},
        {"t": 65665, "ir": 120349, "red": 100190},
        {"t": 65705, "ir": 120396, "red": 100212},
        {"t": 65745, "ir": 120575, "red": 100256},
        {"t": 65785, "ir": 120792, "red": 100335},
        {"t": 65825, "ir": 120955, "red": 100387},
        {"t": 65865, "ir": 121086, "red": 100443},
        {"t": 65905, "ir": 121274, "red": 100510},
        {"t": 65945, "ir": 121440, "red": 100553},
        {"t": 65985, "ir": 121548, "red": 100603},
        {"t": 66025, "ir": 121588, "red": 100655},
        {"t": 66065, "ir": 121812, "red": 100846},
        {"t": 66105, "ir": 122081, "red": 100976},
        {"t": 66145, "ir": 122435, "red": 101104},
    ]
}


def main():
    print("=" * 80)
    print("TEST SpO2 COMPUTATION FROM PPG WINDOW")
    print("=" * 80)
    
    print(f"\n📊 Payload Info:")
    print(f"  - Type: {test_payload.get('type')}")
    print(f"  - fs: {test_payload.get('fs')}")
    print(f"  - Samples: {len(test_payload.get('data', []))}")
    print(f"  - Temp: {test_payload.get('temp')}°C")
    print(f"  - Quality: {test_payload.get('quality', {}).get('status')}")
    
    try:
        # Step 1: Normalize payload (extract IR/RED from data array)
        print(f"\n🔄 Normalizing payload (extracting IR/RED from data array)...")
        normalized = normalize_raw_payload_for_api(test_payload)
        
        print(f"✅ Payload normalized:")
        print(f"  - IR values: {len(normalized.get('ir', []))}")
        print(f"  - RED values: {len(normalized.get('red', []))}")
        print(f"  - Sample interval: {normalized.get('sample_interval_ms')}ms")
        
        # Step 2: Convert to JSON and call from_json_samples
        payload_json = json.dumps(normalized)
        
        print(f"\n🔄 Calling from_json_samples()...")
        samples = from_json_samples(payload_json)
        
        if not samples:
            print("❌ No samples returned!")
            return 1
        
        print(f"✅ Got {len(samples)} sample(s)")
        
        # Display last sample (most recent)
        last_sample = samples[-1]
        
        print(f"\n📈 Last Sample (Most Recent):")
        print(f"  - Timestamp: {last_sample.timestamp}")
        print(f"  - BPM: {last_sample.heart_rate}")
        print(f"  - SpO2: {last_sample.spo2}%")
        print(f"  - Temp: {last_sample.temp}°C")
        print(f"  - Status: {last_sample.status}")
        
        # Summary
        print(f"\n{'='*80}")
        if last_sample.spo2 is not None:
            print(f"✅ SUCCESS: SpO2 = {last_sample.spo2:.1f}%")
        else:
            print(f"❌ FAILED: SpO2 is None")
        print(f"{'='*80}")
        
        return 0 if last_sample.spo2 is not None else 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

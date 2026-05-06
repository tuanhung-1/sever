import binascii

from app import _decode_fall_raw_binary, _process_fall_raw_with_model


def main() -> None:
    file_path = "sample_fall_raw.bin"

    with open(file_path, "rb") as f:
        buf = f.read()

    print("=== Raw (before decode) ===")
    print(f"Bytes: {len(buf)}")
    print("Hex preview (first 64 bytes):")
    print(binascii.hexlify(buf[:64]).decode("ascii"))

    decoded = _decode_fall_raw_binary(buf)

    print("\n=== Decoded ===")
    if decoded is None:
        print("Decode failed")
        return

    print(f"trigger_ts: {decoded['trigger_ts']}")
    print(f"num_samples: {decoded['num_samples']}")
    print(f"pre_samples: {decoded['pre_samples']}")
    print(f"reason: {decoded['reason']}")

    data = decoded["data"]
    print(f"data shape: {data.shape}")
    print("first 3 samples:")
    for i in range(min(3, data.shape[0])):  
        print(data[i])

    print("\n=== Model Inference ===")
    alert_data = {"trigger": decoded.get("reason", "unknown")}
    result = _process_fall_raw_with_model(decoded, alert_data)
    if result is None:
        print("Model did not run (insufficient samples or error)")
        return
    print(result)


if __name__ == "__main__":
    main()

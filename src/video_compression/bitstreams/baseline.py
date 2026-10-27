"""Native big-endian storage with leading zero padding, matching upstream utils."""


def pack(bits: str) -> bytes:
    if not bits or set(bits) - {"0", "1"}:
        raise ValueError("Expected a nonempty binary string")
    return int(bits, 2).to_bytes((len(bits) + 7) // 8, "big")


def unpack(payload: bytes) -> str:
    if not payload:
        raise ValueError("Empty bitstream")
    return bin(int.from_bytes(payload, "big"))[2:].zfill(len(payload) * 8)

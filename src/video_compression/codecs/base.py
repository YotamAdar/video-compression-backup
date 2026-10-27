from typing import Any, Protocol


class Codec(Protocol):
    """Images are RGB tensors [1, 3, H, W] in [-1, 1].

    Parameters needed to decode belong in the run metadata. A temporal codec can
    implement this per frame with explicit state, or introduce a sequence API.
    """

    def encode(self, image: Any) -> bytes: ...

    def decode(self, payload: bytes) -> Any: ...

    def metadata(self) -> dict: ...

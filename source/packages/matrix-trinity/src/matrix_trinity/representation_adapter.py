"""TRINITY adapter for SMITH representation requests."""

from matrix_smith import RepresentationRequest


def validate_trinity_representation(request: RepresentationRequest) -> RepresentationRequest:
    """Validate and return the canonical request consumed by TRINITY."""

    if not isinstance(request, RepresentationRequest):
        raise TypeError("TRINITY requires a matrix_smith.RepresentationRequest")
    return request

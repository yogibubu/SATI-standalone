"""ARCHITECT adapter for shared ZAFF/EVB representation requests."""

from matrix_smith import SurfaceRepresentationRequest


def validate_architect_representation(request: SurfaceRepresentationRequest) -> SurfaceRepresentationRequest:
    if not isinstance(request, SurfaceRepresentationRequest):
        raise TypeError("ARCHITECT requires a SurfaceRepresentationRequest")
    return request

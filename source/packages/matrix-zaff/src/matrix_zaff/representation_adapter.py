"""ZAFF adapter for the shared SMITH surface representation contract."""

from matrix_smith import SurfaceRepresentationRequest, evaluate_surface_request


def validate_zaff_representation(request: SurfaceRepresentationRequest) -> SurfaceRepresentationRequest:
    if not isinstance(request, SurfaceRepresentationRequest) or request.provider != "ZAFF":
        raise TypeError("ZAFF requires a ZAFF SurfaceRepresentationRequest")
    return request


def evaluate_zaff_request(request, evaluator, values):
    """Run an existing ZAFF evaluator through the shared surface dispatcher."""

    validated = validate_zaff_representation(request)
    return evaluate_surface_request(validated, evaluator, values)

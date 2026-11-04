"""LINK constructors for global PES and local SONIC representation requests."""

from matrix_smith import RepresentationRequest


def global_pes_representation() -> RepresentationRequest:
    return RepresentationRequest(
        mode="PERIODIC_EMBEDDING",
        purpose="GLOBAL_PES",
        continuous=True,
    )


def local_optimization_representation() -> RepresentationRequest:
    return RepresentationRequest(mode="SCALAR", purpose="LOCAL_OPTIMIZATION")

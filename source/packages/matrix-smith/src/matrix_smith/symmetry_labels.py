"""Point-group irrep labels used by frozen GIC definitions."""

from __future__ import annotations

import re

import numpy as np

from matrix_chem import dihedral_operation_label_from_matrix


_IRREP_ORDER_BY_POINT_GROUP = {
    "C1": ("A",),
    "CS": ("A'", "A''"),
    "CI": ("Ag", "Au"),
    "C2": ("A", "B"),
    "C2V": ("A1", "A2", "B1", "B2"),
    "D2": ("A", "B1", "B2", "B3"),
    "C2H": ("Ag", "Bg", "Au", "Bu"),
    "D2H": ("Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"),
    "C3V": ("A1", "A2", "E"),
    "C4V": ("A1", "A2", "B1", "B2", "E"),
    "C6V": ("A1", "A2", "B1", "B2", "E1", "E2"),
    "CINFV": ("A1", "A2", "B1", "B2", "E1", "E2"),
    "D3H": ("A1'", "A2'", "E'", "A1''", "A2''", "E''"),
    "D4H": (
        "A1g",
        "A2g",
        "B1g",
        "B2g",
        "Eg",
        "A1u",
        "A2u",
        "B1u",
        "B2u",
        "Eu",
    ),
    "DINFH": (
        "A1g",
        "A2g",
        "B1g",
        "B2g",
        "E1g",
        "E2g",
        "A1u",
        "A2u",
        "B1u",
        "B2u",
        "E1u",
        "E2u",
    ),
    "T": ("A", "E", "T"),
    "TD": ("A1", "A2", "E", "T1", "T2"),
    "TH": ("Ag", "Eg", "Tg", "Au", "Eu", "Tu"),
    "O": ("A1", "A2", "E", "T1", "T2"),
    "OH": ("A1g", "A2g", "Eg", "T1g", "T2g", "A1u", "A2u", "Eu", "T1u", "T2u"),
    "I": ("A", "T1", "T2", "G", "H"),
    "IH": ("Ag", "T1g", "T2g", "Gg", "Hg", "Au", "T1u", "T2u", "Gu", "Hu"),
}


def normalized_point_group(point_group: str | None) -> str:
    text = (point_group or "C1").strip()
    return text or "C1"


def irrep_sequence(point_group: str | None) -> tuple[str, ...]:
    """Return a deterministic irrep order with the totally symmetric irrep first."""
    group = normalized_point_group(point_group)
    exact = _IRREP_ORDER_BY_POINT_GROUP.get(group.upper())
    if exact:
        return exact

    improper_match = re.fullmatch(r"S(\d+)", group.upper())
    if improper_match:
        order = int(improper_match.group(1))
        if order >= 4 and order % 2 == 0:
            return tuple(name for name, _wave, _dimension in _s2n_irrep_specs(order))

    match = re.fullmatch(r"([CD])(\d+)([VHD]?)", group.upper())
    if not match:
        return ("A",)
    family, n_text, suffix = match.groups()
    n = int(n_text)
    e_names = [_e_name(idx, n) for idx in range(1, (n + 1) // 2)]
    if family == "C":
        if suffix == "V":
            base = ["A1", "A2"]
            if n % 2 == 0:
                base.extend(["B1", "B2"])
            base.extend(e_names)
            return tuple(base)
        elif suffix == "H":
            if n == 2:
                return _IRREP_ORDER_BY_POINT_GROUP["C2H"]
            if n % 2 == 1:
                base = ["A'", "A''"]
                for name in e_names:
                    base.extend([f"{name}'", f"{name}''"])
                return tuple(base)
            modes = ["A", "B"]
            modes.extend(e_names)
            return tuple(f"{name}g" for name in modes) + tuple(
                f"{name}u" for name in modes
            )
        base = ["A"]
        if n % 2 == 0:
            base.append("B")
        base.extend(e_names)
        return tuple(base)

    base = ["A1", "A2"]
    if n % 2 == 0:
        base.extend(["B1", "B2"])
    base.extend(e_names)
    if suffix == "D":
        if n % 2 == 1:
            return tuple(
                dict.fromkeys([f"{name}g" for name in base] + [f"{name}u" for name in base])
            )
        return tuple(
            ["A1", "A2", "B1", "B2"]
            + [_e_name(idx, 2 * n) for idx in range(1, n)]
        )
    if suffix == "H" and n % 2 == 0:
        return tuple(f"{name}g" for name in base) + tuple(
            f"{name}u" for name in base
        )
    if suffix == "H":
        return tuple(f"{name}'" for name in base) + tuple(
            f"{name}''" for name in base
        )
    return tuple(base)


def total_symmetric_irrep(point_group: str | None) -> str:
    return irrep_sequence(point_group)[0]


def non_total_irrep_sequence(point_group: str | None) -> tuple[str, ...]:
    irreps = irrep_sequence(point_group)
    return tuple(irrep for irrep in irreps if irrep != irreps[0])


def is_total_symmetric_irrep(point_group: str | None, irrep: str | None) -> bool:
    return (irrep or "").strip() == total_symmetric_irrep(point_group)


def irrep_name_prefix(irrep: str | None) -> str:
    """Convert an irrep label to a compact coordinate-name prefix."""
    text = (irrep or "X").strip() or "X"
    return text.replace("'", "p").replace('"', "pp")


def irrep_dimension(irrep: str | None) -> int:
    """Return the dimension of a conventional real molecular-point-group irrep."""

    label = (irrep or "").strip().upper()
    if not label:
        raise ValueError("irrep label must be non-empty")
    dimensions = {"A": 1, "B": 1, "E": 2, "T": 3, "G": 4, "H": 5}
    dimension = dimensions.get(label[0])
    if dimension is None:
        raise ValueError(f"unsupported real molecular-point-group irrep: {irrep}")
    return dimension


def irrep_characters_for_operations(
    operation_labels: tuple[str, ...] | list[str],
    point_group: str | None,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None = None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    labels = tuple(_canonical_operation_label(label) for label in operation_labels)
    if labels == ("E",):
        return (("A", (1.0,)),)
    group = normalized_point_group(point_group)
    group_key = group.upper()
    if group_key == "CS":
        return (
            ("A'", tuple(1.0 for _label in labels)),
            (
                "A''",
                tuple(-1.0 if label.startswith("sigma") else 1.0 for label in labels),
            ),
        )
    if group_key == "CI":
        return (
            ("Ag", tuple(1.0 for _label in labels)),
            ("Au", tuple(-1.0 if label == "i" else 1.0 for label in labels)),
        )
    if group_key == "C2":
        return (
            ("A", tuple(1.0 for _label in labels)),
            ("B", tuple(-1.0 if label.startswith("C2") else 1.0 for label in labels)),
        )
    if group_key == "C2H":
        return _c2h_characters(labels, operation_matrices=operation_matrices)
    if group_key == "C2V":
        sigma_labels = [label for label in labels if label.startswith("sigma")]
        preferred = ("sigma_xz", "sigma_yz", "sigma_xy")
        first_sigma = next((label for label in preferred if label in sigma_labels), None)
        second_sigma = next(
            (label for label in sorted(sigma_labels) if label != first_sigma),
            None,
        )
        if first_sigma and second_sigma:
            table = {
                "E": (1.0, 1.0, 1.0, 1.0),
                "C2x": (1.0, 1.0, -1.0, -1.0),
                "C2y": (1.0, 1.0, -1.0, -1.0),
                "C2z": (1.0, 1.0, -1.0, -1.0),
                "C2_perp": (1.0, 1.0, -1.0, -1.0),
                first_sigma: (1.0, -1.0, 1.0, -1.0),
                second_sigma: (1.0, -1.0, -1.0, 1.0),
            }
            rows = np.array(
                [
                    table.get(
                        "C2_perp" if label.startswith("C2_xy") else label,
                        (1.0, 1.0, 1.0, 1.0),
                    )
                    for label in labels
                ]
            )
            return tuple(
                (name, tuple(float(value) for value in rows[:, idx]))
                for idx, name in enumerate(("A1", "A2", "B1", "B2"))
            )
    if group_key == "D2":
        return _d2_characters(labels)
    if group_key == "D2H":
        return _d2h_characters(labels)
    generic = _generic_family_characters(
        labels,
        group_key,
        operation_matrices=operation_matrices,
    )
    if generic:
        return generic
    polyhedral = _polyhedral_family_characters(
        labels,
        group_key,
        operation_matrices=operation_matrices,
    )
    if polyhedral:
        return polyhedral
    return ()


def _generic_family_characters(
    labels: tuple[str, ...],
    group_key: str,
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None = None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    improper_match = re.fullmatch(r"S(\d+)", group_key)
    if improper_match:
        return _s2n_characters(
            labels,
            int(improper_match.group(1)),
            operation_matrices=operation_matrices,
        )
    match = re.fullmatch(r"([CD])(\d+)([VHD]?)", group_key)
    if not match:
        return ()
    family, n_text, suffix = match.groups()
    n = int(n_text)
    if n < 2:
        return ()
    if family == "C" and suffix == "":
        return _cn_characters(labels, n)
    if family == "C" and suffix == "V":
        return _cnv_characters(labels, n, operation_matrices=operation_matrices)
    if family == "C" and suffix == "H":
        return _cnh_characters(labels, n)
    if family == "D" and suffix == "":
        return _dn_characters(labels, n)
    if family == "D" and suffix == "H":
        return _dnh_characters(labels, n)
    if family == "D" and suffix == "D":
        if operation_matrices is None:
            return _matrix_dnd_label_characters(labels, n)
        return _dnd_characters(labels, n, operation_matrices=operation_matrices)
    return ()


def _s2n_irrep_specs(order: int) -> tuple[tuple[str, int, float], ...]:
    """Return real-irrep names, cyclic wave numbers and dimensions for S_2n."""

    if order < 4 or order % 2:
        return ()
    n = order // 2
    if n % 2 == 0:
        return tuple(
            [("A", 0, 1.0), ("B", n, 1.0)]
            + [(_e_name(wave, order), wave, 2.0) for wave in range(1, n)]
        )
    modes = [("A", 0, 1.0)] + [
        (_e_name(wave, n), wave, 2.0) for wave in range(1, (n + 1) // 2)
    ]
    specs: list[tuple[str, int, float]] = []
    for suffix, parity in (("g", 1), ("u", -1)):
        for name, wave, dimension in modes:
            cyclic_wave = wave if (-1) ** wave == parity else wave + n
            specs.append((name + suffix, cyclic_wave % order, dimension))
    return tuple(specs)


def _s2n_characters(
    labels: tuple[str, ...],
    order: int,
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if operation_matrices is None or len(operation_matrices) != len(labels):
        return ()
    matrices = tuple(np.asarray(matrix, dtype=float) for matrix in operation_matrices)
    powers = _cyclic_improper_operation_powers(matrices, order)
    if powers is None:
        return ()
    rows: list[tuple[str, tuple[float, ...]]] = []
    for name, wave, dimension in _s2n_irrep_specs(order):
        rows.append(
            (
                name,
                tuple(
                    float(dimension * np.cos(2.0 * np.pi * wave * power / order))
                    for power in powers
                ),
            )
        )
    return tuple(rows)


def _cyclic_improper_operation_powers(
    matrices: tuple[np.ndarray, ...],
    order: int,
) -> tuple[int, ...] | None:
    identity = np.eye(3)
    for generator in matrices:
        if float(np.linalg.det(generator)) >= 0.0:
            continue
        generated: list[np.ndarray] = []
        current = identity.copy()
        for _power in range(order):
            generated.append(current.copy())
            current = current @ generator
        if not np.allclose(current, identity, atol=1.0e-8):
            continue
        mapping: list[int] = []
        for matrix in matrices:
            matches = [
                power
                for power, candidate in enumerate(generated)
                if np.allclose(matrix, candidate, atol=1.0e-8)
            ]
            if len(matches) != 1:
                break
            mapping.append(matches[0])
        if len(mapping) == order:
            return tuple(mapping)
    return None


def _cn_characters(
    labels: tuple[str, ...],
    n: int,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    rows: list[tuple[str, tuple[float, ...]]] = []
    rows.append(("A", tuple(_rotation_character(label, n, 0) for label in labels)))
    if n % 2 == 0:
        rows.append(("B", tuple(_rotation_character(label, n, n // 2) for label in labels)))
    for order in range(1, (n + 1) // 2):
        if n % 2 == 0 and order == n // 2:
            continue
        rows.append(
            (
                _e_name(order, n),
                tuple(2.0 * _rotation_character(label, n, order) for label in labels),
            )
        )
    return tuple(rows)


def _cnv_characters(
    labels: tuple[str, ...],
    n: int,
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None = None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    operation_coordinates = _cnv_operation_coordinates(operation_matrices, n)
    if operation_coordinates is not None:
        rows: list[tuple[str, tuple[float, ...]]] = [
            (
                "A1",
                tuple(1.0 for _reflected, _power in operation_coordinates),
            ),
            (
                "A2",
                tuple(-1.0 if reflected else 1.0 for reflected, _power in operation_coordinates),
            ),
        ]
        if n % 2 == 0:
            rows.extend(
                (
                    (
                        "B1",
                        tuple(
                            (-1.0) ** power
                            for _reflected, power in operation_coordinates
                        ),
                    ),
                    (
                        "B2",
                        tuple(
                            -((-1.0) ** power) if reflected else (-1.0) ** power
                            for reflected, power in operation_coordinates
                        ),
                    ),
                )
            )
        for order in range(1, (n + 1) // 2):
            if n % 2 == 0 and order == n // 2:
                continue
            rows.append(
                (
                    _e_name(order, n),
                    tuple(
                        0.0
                        if reflected
                        else float(2.0 * np.cos(2.0 * np.pi * order * power / n))
                        for reflected, power in operation_coordinates
                    ),
                )
            )
        return tuple(rows)

    rows: list[tuple[str, tuple[float, ...]]] = [
        ("A1", tuple(_cnv_one_dim_character(label, n, 0, 1) for label in labels)),
        ("A2", tuple(_cnv_one_dim_character(label, n, 0, -1) for label in labels)),
    ]
    if n % 2 == 0:
        rows.extend(
            (
                (
                    "B1",
                    tuple(_cnv_one_dim_character(label, n, n // 2, 1) for label in labels),
                ),
                (
                    "B2",
                    tuple(_cnv_one_dim_character(label, n, n // 2, -1) for label in labels),
                ),
            )
        )
    for order in range(1, (n + 1) // 2):
        if n % 2 == 0 and order == n // 2:
            continue
        rows.append(
            (
                _e_name(order, n),
                tuple(_cnv_e_character(label, n, order) for label in labels),
            )
        )
    return tuple(rows)


def _cnv_operation_coordinates(
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
    n: int,
) -> tuple[tuple[bool, int], ...] | None:
    if operation_matrices is None or len(operation_matrices) != 2 * n:
        return None
    matrices = tuple(np.asarray(matrix, dtype=float) for matrix in operation_matrices)
    identity = np.eye(3)
    generator = next(
        (
            matrix
            for matrix in matrices
            if float(np.linalg.det(matrix)) > 0.0
            and np.allclose(
                np.linalg.matrix_power(matrix, n),
                identity,
                atol=1.0e-8,
            )
            and not any(
                np.allclose(
                    np.linalg.matrix_power(matrix, power),
                    identity,
                    atol=1.0e-8,
                )
                for power in range(1, n)
            )
        ),
        None,
    )
    reflection = next(
        (matrix for matrix in matrices if float(np.linalg.det(matrix)) < 0.0),
        None,
    )
    if generator is None or reflection is None:
        return None
    rotations = tuple(np.linalg.matrix_power(generator, power) for power in range(n))
    reflected = tuple(reflection @ rotation for rotation in rotations)
    coordinates: list[tuple[bool, int]] = []
    for matrix in matrices:
        matches = [
            (False, power)
            for power, candidate in enumerate(rotations)
            if np.allclose(matrix, candidate, atol=1.0e-8)
        ] + [
            (True, power)
            for power, candidate in enumerate(reflected)
            if np.allclose(matrix, candidate, atol=1.0e-8)
        ]
        if len(matches) != 1:
            return None
        coordinates.append(matches[0])
    return tuple(coordinates)


def _cnh_characters(
    labels: tuple[str, ...],
    n: int,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    rows: list[tuple[str, tuple[float, ...]]] = []
    base: list[tuple[str, int, float]] = [("A", 0, 1.0)]
    if n % 2 == 0:
        base.append(("B", n // 2, 1.0))
    for order in range(1, (n + 1) // 2):
        if n % 2 == 0 and order == n // 2:
            continue
        base.append((_e_name(order, n), order, 2.0))
    if n % 2 == 1:
        for name, order, dimension in base:
            rows.append(
                (
                    name + "'",
                    tuple(
                        dimension * _cnh_character(label, n, order, 1)
                        for label in labels
                    ),
                )
            )
            rows.append(
                (
                    name + "''",
                    tuple(
                        dimension * _cnh_character(label, n, order, -1)
                        for label in labels
                    ),
                )
            )
        return tuple(rows)
    for suffix, inversion_parity in (("g", 1.0), ("u", -1.0)):
        for name, order, dimension in base:
            horizontal_sign = inversion_parity * (-1.0 if order % 2 else 1.0)
            rows.append(
                (
                    name + suffix,
                    tuple(
                        dimension
                        * _cnh_character(label, n, order, int(horizontal_sign))
                        for label in labels
                    ),
                )
            )
    return tuple(rows)


def _dn_characters(
    labels: tuple[str, ...],
    n: int,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if n == 2:
        return _d2_characters(labels)
    rows: list[tuple[str, tuple[float, ...]]] = [
        ("A1", tuple(_dn_one_dim_character(label, n, 0, 1) for label in labels)),
        ("A2", tuple(_dn_one_dim_character(label, n, 0, -1) for label in labels)),
    ]
    if n % 2 == 0:
        rows.extend(
            (
                (
                    "B1",
                    tuple(_dn_one_dim_character(label, n, n // 2, 1) for label in labels),
                ),
                (
                    "B2",
                    tuple(_dn_one_dim_character(label, n, n // 2, -1) for label in labels),
                ),
            )
        )
    for order in range(1, (n + 1) // 2):
        if n % 2 == 0 and order == n // 2:
            continue
        rows.append(
            (
                _e_name(order, n),
                tuple(_dn_e_character(label, n, order) for label in labels),
            )
        )
    return tuple(rows)


def _rotation_character(label: str, n: int, order: int) -> float:
    power = _principal_rotation_power(label, n)
    if power is None:
        return 0.0
    return float(np.cos(2.0 * np.pi * order * power / n))


def _cnv_one_dim_character(label: str, n: int, order: int, reflection_sign: int) -> float:
    if _is_vertical_reflection(label):
        mirror_index = _vertical_reflection_index(label, n)
        phase = 1.0 if order == 0 else (-1.0 if (mirror_index % 2) else 1.0)
        return float(reflection_sign) * phase
    return _rotation_character(label, n, order)


def _cnv_e_character(label: str, n: int, order: int) -> float:
    if _is_vertical_reflection(label):
        return 0.0
    return 2.0 * _rotation_character(label, n, order)


def _cnh_character(label: str, n: int, order: int, reflection_sign: int) -> float:
    reflected = _is_horizontal_reflected(label, n)
    power = _cnh_rotation_power(label, n)
    if power is None:
        return 0.0
    value = float(np.cos(2.0 * np.pi * order * power / n))
    return value * (float(reflection_sign) if reflected else 1.0)


def _dn_one_dim_character(label: str, n: int, order: int, c2_sign: int) -> float:
    if _is_c2_prime(label):
        mirror_index = _c2_prime_index(label, n)
        phase = 1.0 if order == 0 else (-1.0 if (mirror_index % 2) else 1.0)
        return float(c2_sign) * phase
    return _rotation_character(label, n, order)


def _dn_e_character(label: str, n: int, order: int) -> float:
    if _is_c2_prime(label):
        return 0.0
    return 2.0 * _rotation_character(label, n, order)


def _dnh_characters(
    labels: tuple[str, ...],
    n: int,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if n == 2:
        return _d2h_characters(labels)
    base_names = ["A1", "A2"]
    if n % 2 == 0:
        base_names.extend(["B1", "B2"])
    for order in range(1, (n + 1) // 2):
        if n % 2 == 0 and order == n // 2:
            continue
        base_names.append(_e_name(order, n))

    if n % 2 == 0:
        return tuple(
            (name + suffix, tuple(_dnh_value(label, n, name, parity) for label in labels))
            for suffix, parity in (("g", 1), ("u", -1))
            for name in base_names
        )
    return tuple(
        (name + suffix, tuple(_dnh_value(label, n, name, parity) for label in labels))
        for suffix, parity in (("'", 1), ("''", -1))
        for name in base_names
    )


def _dnd_characters(
    labels: tuple[str, ...],
    n: int,
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if n % 2 == 0:
        effective_labels = _dnd_even_effective_labels(
            labels,
            n,
            operation_matrices=operation_matrices,
        )
        if effective_labels is None:
            return ()
        return _dn_characters(effective_labels, 2 * n)

    underlying = _dnd_odd_underlying_labels(
        labels,
        n,
        operation_matrices=operation_matrices,
    )
    if underlying is None:
        return ()
    base_names = ["A1", "A2"]
    for order in range(1, (n + 1) // 2):
        base_names.append(_e_name(order, n))
    return tuple(
        (
            name + suffix,
            tuple(
                _dn_irrep_value(underlying_label, n, name) * (float(parity) if reflected else 1.0)
                for underlying_label, reflected in underlying
            ),
        )
        for suffix, parity in (("g", 1), ("u", -1))
        for name in base_names
    )


def _dnd_even_effective_labels(
    labels: tuple[str, ...],
    n: int,
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[str, ...] | None:
    order = 2 * n
    matrix_tuple = (
        tuple(np.asarray(matrix, dtype=float) for matrix in operation_matrices)
        if operation_matrices is not None
        else None
    )
    if matrix_tuple is not None and len(matrix_tuple) != len(labels):
        return None
    out: list[str] = []
    for idx, label in enumerate(labels):
        # In D_nd the serialized physical-axis index is not the axis index of
        # the effective D_2n representation.  In particular, for D2d the two
        # sigma_d planes and the two perpendicular C2 axes used to collapse
        # onto the same effective labels, making the A2 and B1 character rows
        # identical.  The operation matrix carries the unambiguous effective
        # orientation, so prefer it whenever it is available.
        mapped = (
            _dnd_even_effective_label_from_matrix(matrix_tuple[idx], n)
            if matrix_tuple is not None
            else None
        )
        if mapped is None:
            mapped = _dnd_even_effective_label(label, n)
        if mapped is None:
            return None
        out.append(_canonical_dnd_effective_label(mapped, order))
    return tuple(out)


def _dnd_even_effective_label(label: str, n: int) -> str | None:
    order = 2 * n
    if label == "E":
        return "E"
    if label == "i":
        return f"C{order}z^{n}"
    match = re.fullmatch(r"Dnd_C(\d+)z\^(\d+)", label)
    if match:
        op_order, power = match.groups()
        relative = order * int(power) / int(op_order)
        return f"C{order}z^{int(round(relative)) % order}"
    match = re.fullmatch(r"Dnd_C2_xy_(\d+)_(\d+)", label)
    if match:
        _op_order, index = match.groups()
        return f"C2_xy_{order}_{int(index) % order}"
    match = re.fullmatch(r"sigma_v_(\d+)_(\d+)", label)
    if match:
        op_order, index = match.groups()
        relative = order * int(index) / int(op_order)
        nearest = int(round(relative))
        if abs(relative - nearest) <= 1.0e-8:
            return f"C2_xy_{order}_{nearest % order}"
    match = re.fullmatch(r"sigma_h\*C(\d+)z\^(\d+)", label)
    if match:
        op_order, power = (int(item) for item in match.groups())
        relative = order * power / op_order
        nearest = int(round(relative))
        if abs(relative - nearest) <= 1.0e-8 and nearest % 2 == 1:
            return f"C{order}z^{nearest % order}"
    match = re.fullmatch(r"C(\d+)z\^(\d+)", label)
    if match:
        op_order, power = match.groups()
        relative = n * int(power) / int(op_order)
        nearest = int(round(relative))
        if abs(relative - nearest) <= 1.0e-8:
            return f"C{order}z^{(2 * nearest) % order}"
    match = re.fullmatch(r"C2_xy_(\d+)_(\d+)", label)
    if match:
        op_order, index = match.groups()
        relative = order * int(index) / int(op_order)
        nearest = int(round(relative))
        if abs(relative - nearest) <= 1.0e-8:
            return f"C2_xy_{order}_{nearest % order}"
    if label in {"C2x", "C2y"}:
        return f"C2_xy_{order}_{0 if label == 'C2x' else n}"
    if label == "C2z":
        return f"C{order}z^{n}"
    return None


def _dnd_even_effective_label_from_matrix(matrix: np.ndarray, n: int) -> str | None:
    order = 2 * n
    det = float(np.linalg.det(matrix))
    if det > 0.0:
        dn_label = _dn_label_from_matrix(matrix, n)
        if dn_label is None:
            return None
        return _dnd_even_effective_label(dn_label, n)
    sd = _diagonal_reflection_matrix()
    dn_matrix = sd @ matrix
    dn_label = _dn_label_from_matrix(dn_matrix, n)
    if dn_label is None:
        return None
    if dn_label == "E":
        return f"C2_xy_{order}_1"
    match = re.fullmatch(r"C(\d+)z\^(\d+)", dn_label)
    if match:
        _op_order, power = match.groups()
        return f"C2_xy_{order}_{(2 * int(power) + 1) % order}"
    match = re.fullmatch(r"C2_xy_(\d+)_(\d+)", dn_label)
    if match:
        _op_order, index = match.groups()
        return f"C{order}z^{(2 * int(index) + 1) % order}"
    if dn_label == "C2z":
        return f"C2_xy_{order}_{(n + 1) % order}"
    return None


def _canonical_dnd_effective_label(label: str, order: int) -> str:
    match = re.fullmatch(r"C(\d+)z\^(\d+)", label)
    if match:
        _op_order, power = match.groups()
        power_int = int(power) % order
        if power_int == 0:
            return "E"
        if power_int == order // 2:
            return "C2z"
        return f"C{order}z^{power_int}"
    return label


def _dnd_odd_underlying_labels(
    labels: tuple[str, ...],
    n: int,
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[tuple[str, bool], ...] | None:
    matrix_tuple = (
        tuple(np.asarray(matrix, dtype=float) for matrix in operation_matrices)
        if operation_matrices is not None
        else None
    )
    if matrix_tuple is not None and len(matrix_tuple) != len(labels):
        return None
    out: list[tuple[str, bool]] = []
    for idx, label in enumerate(labels):
        if matrix_tuple is not None:
            matrix = matrix_tuple[idx]
            reflected = float(np.linalg.det(matrix)) < 0.0
            proper = -matrix if reflected else matrix
            underlying = _dn_label_from_matrix(proper, n)
            if underlying is None:
                underlying, reflected = _dnd_odd_underlying_label(label, n)
        else:
            underlying, reflected = _dnd_odd_underlying_label(label, n)
        if underlying is None:
            return None
        out.append((underlying, reflected))
    return tuple(out)


def _dnd_odd_underlying_label(label: str, n: int) -> tuple[str | None, bool]:
    if label == "i":
        return "E", True
    if label in {"sigma_xz", "sigma_yz"} or label.startswith("sigma_v"):
        return "C2_xy", True
    match = re.fullmatch(r"sigma_h\*C(\d+)z\^(\d+)", label)
    if match:
        operation_order, power = (int(item) for item in match.groups())
        if operation_order != 2 * n or power % 2 != 1:
            return None, True
        return f"C{n}z^{((n + power) // 2) % n}", True
    match = re.fullmatch(r"C2_xy_(\d+)_(\d+)", label)
    if match:
        operation_order, index = (int(item) for item in match.groups())
        if operation_order == 2 * n and index % 2 == 1:
            return "C2_xy", False
    if label.startswith("Dnd_") or label.startswith("sigma"):
        return None, True
    if label == "E" or label.startswith("C"):
        return label, False
    return None, False


def _matrix_dnd_label_characters(
    labels: tuple[str, ...],
    n: int,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    canonical = tuple(_matrix_canonical_operation_label(label) for label in labels)
    base = _matrix_dn_label_characters(canonical, n)
    if "i" in canonical:
        return _matrix_gerade_ungerade(base, canonical)
    return _matrix_prime_doubleprime(base, canonical)


def _matrix_dn_label_characters(
    labels: tuple[str, ...],
    n: int,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    rows: list[tuple[str, tuple[float, ...]]] = [
        ("A1", tuple(1.0 for _label in labels)),
        (
            "A2",
            tuple(-1.0 if _matrix_is_c2_perpendicular(label) else 1.0 for label in labels),
        ),
    ]
    if n % 2 == 0:
        rows.extend(
            (
                (
                    "B1",
                    tuple((-1.0) ** _matrix_rotation_power(label) for label in labels),
                ),
                (
                    "B2",
                    tuple(
                        -((-1.0) ** _matrix_rotation_power(label))
                        if _matrix_is_c2_perpendicular(label)
                        else (-1.0) ** _matrix_rotation_power(label)
                        for label in labels
                    ),
                ),
            )
        )
    max_order = (n - 1) // 2 if n % 2 else (n // 2 - 1)
    for order in range(1, max_order + 1):
        values = []
        for label in labels:
            if _matrix_is_c2_perpendicular(label):
                values.append(0.0)
            else:
                values.append(
                    2.0 * np.cos(2.0 * np.pi * order * _matrix_rotation_power(label) / float(n))
                )
        rows.append((f"E{order}", tuple(float(2.0 * value) for value in values)))
    return tuple(rows)


def _matrix_gerade_ungerade(
    base: tuple[tuple[str, tuple[float, ...]], ...],
    labels: tuple[str, ...],
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    parity = tuple(-1.0 if label == "i" or label.startswith("sigma") else 1.0 for label in labels)
    rows: list[tuple[str, tuple[float, ...]]] = []
    for name, chars in base:
        rows.append((f"{name}g", chars))
        rows.append((f"{name}u", tuple(char * sign for char, sign in zip(chars, parity))))
    return tuple(rows)


def _matrix_prime_doubleprime(
    base: tuple[tuple[str, tuple[float, ...]], ...],
    labels: tuple[str, ...],
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    reflection = tuple(
        -1.0 if label.startswith("sigma") or label.startswith("S") else 1.0 for label in labels
    )
    rows: list[tuple[str, tuple[float, ...]]] = []
    for name, chars in base:
        rows.append((f"{name}'", chars))
        rows.append((f"{name}''", tuple(char * sign for char, sign in zip(chars, reflection))))
    return tuple(rows)


def _matrix_rotation_power(label: str) -> int:
    if label == "E":
        return 0
    match = re.match(r"C(\d+)[xyz]\^(\d+)", label)
    if match:
        return int(match.group(2))
    if label.startswith("C2"):
        return 1
    return 0


def _matrix_is_c2_perpendicular(label: str) -> bool:
    return label == "C2_perp" or label.startswith("C2x") or label.startswith("C2y")


def _matrix_canonical_operation_label(label: str) -> str:
    text = str(label)
    if text == "E" or text == "i" or text.startswith("sigma"):
        return text
    match = re.match(r"C(\d+)([xyz])\^(\d+)", text)
    if match:
        order, axis, power = match.groups()
        if int(order) == 2:
            return f"C2{axis}"
        return f"C{order}{axis}^{power}"
    if text.startswith("C2_xy"):
        return "C2_perp"
    if text.startswith("S"):
        return text
    return text


def _dn_label_from_matrix(matrix: np.ndarray, n: int) -> str | None:
    return dihedral_operation_label_from_matrix(matrix, n)


def _diagonal_reflection_matrix() -> np.ndarray:
    return np.array(
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )


def _dnh_value(label: str, n: int, irrep: str, reflection_sign: int) -> float:
    if n % 2 == 0:
        underlying, reflected = _dnh_even_underlying_operation(label, n)
    else:
        underlying, reflected = _dnh_odd_underlying_operation(label, n)
    value = _dn_irrep_value(underlying, n, irrep)
    return value * (float(reflection_sign) if reflected else 1.0)


def _dnh_even_underlying_operation(label: str, n: int) -> tuple[str, bool]:
    if label == "i":
        return "E", True
    if label == "sigma_xy":
        return "C2z", True
    if label == "sigma_xz":
        return "C2y", True
    if label == "sigma_yz":
        return "C2x", True
    if _is_vertical_reflection(label):
        return label.replace("sigma_v", "C2_xy"), True
    match = re.fullmatch(r"sigma_h\*C(\d+)z\^(\d+)", label)
    if match:
        order, power = match.groups()
        relative = n * int(power) / int(order)
        nearest = int(round(relative))
        if abs(relative - nearest) > 1.0e-8:
            return label, False
        return f"C{n}z^{(nearest + n // 2) % n}", True
    return label, False


def _dnh_odd_underlying_operation(label: str, n: int) -> tuple[str, bool]:
    if label == "sigma_xy":
        return "E", True
    if label == "sigma_xz":
        return "C2x", True
    if label == "sigma_yz":
        return "C2y", True
    if _is_vertical_reflection(label):
        return label.replace("sigma_v", "C2_xy"), True
    match = re.fullmatch(r"sigma_h\*C(\d+)z\^(\d+)", label)
    if match:
        order, power = match.groups()
        return f"C{order}z^{power}", True
    return label, False


def _dn_irrep_value(label: str, n: int, irrep: str) -> float:
    if irrep == "A1":
        return _dn_one_dim_character(label, n, 0, 1)
    if irrep == "A2":
        return _dn_one_dim_character(label, n, 0, -1)
    if irrep == "B1" and n % 2 == 0:
        return _dn_one_dim_character(label, n, n // 2, 1)
    if irrep == "B2" and n % 2 == 0:
        return _dn_one_dim_character(label, n, n // 2, -1)
    match = re.fullmatch(r"E(\d*)", irrep)
    if match:
        order = int(match.group(1) or "1")
        return _dn_e_character(label, n, order)
    return 0.0


def _d2_characters(labels: tuple[str, ...]) -> tuple[tuple[str, tuple[float, ...]], ...]:
    values = {
        "A": {"C2x": 1.0, "C2y": 1.0, "C2z": 1.0},
        "B1": {"C2x": -1.0, "C2y": -1.0, "C2z": 1.0},
        "B2": {"C2x": -1.0, "C2y": 1.0, "C2z": -1.0},
        "B3": {"C2x": 1.0, "C2y": -1.0, "C2z": -1.0},
    }
    return tuple(
        (name, tuple(chars.get(label, 1.0) for label in labels)) for name, chars in values.items()
    )


def _c2h_characters(
    labels: tuple[str, ...],
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None = None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if operation_matrices is not None and len(operation_matrices) == len(labels):
        rotations = tuple(np.asarray(matrix, dtype=float) for matrix in operation_matrices)
        c2_sign = tuple(
            1.0
            if np.allclose(rotation, np.eye(3), atol=1.0e-8)
            or np.allclose(rotation, -np.eye(3), atol=1.0e-8)
            else -1.0
            for rotation in rotations
        )
        inversion_sign = tuple(
            -1.0 if float(np.linalg.det(rotation)) < 0.0 else 1.0
            for rotation in rotations
        )
        return tuple(
            (
                name,
                tuple(
                    (a_sign if a_or_b == "A" else a_sign * rotation_sign)
                    * (1.0 if parity == "g" else inversion)
                    for rotation_sign, inversion in zip(
                        c2_sign,
                        inversion_sign,
                        strict=True,
                    )
                ),
            )
            for name, a_or_b, parity, a_sign in (
                ("Ag", "A", "g", 1.0),
                ("Bg", "B", "g", 1.0),
                ("Au", "A", "u", 1.0),
                ("Bu", "B", "u", 1.0),
            )
        )
    values = {
        "Ag": {"E": 1.0, "C2z": 1.0, "i": 1.0, "sigma_xy": 1.0},
        "Bg": {"E": 1.0, "C2z": -1.0, "i": 1.0, "sigma_xy": -1.0},
        "Au": {"E": 1.0, "C2z": 1.0, "i": -1.0, "sigma_xy": -1.0},
        "Bu": {"E": 1.0, "C2z": -1.0, "i": -1.0, "sigma_xy": 1.0},
    }
    return tuple(
        (name, tuple(chars.get(label, 0.0) for label in labels)) for name, chars in values.items()
    )


def _d2h_characters(labels: tuple[str, ...]) -> tuple[tuple[str, tuple[float, ...]], ...]:
    rows: list[tuple[str, tuple[float, ...]]] = []
    for name in ("A", "B1", "B2", "B3"):
        rows.append((name + "g", tuple(_d2h_value(name, label, 1) for label in labels)))
    for name in ("A", "B1", "B2", "B3"):
        rows.append((name + "u", tuple(_d2h_value(name, label, -1) for label in labels)))
    return tuple(rows)


def _d2h_value(name: str, label: str, inversion_sign: int) -> float:
    d2_values = {
        "A": {"E": 1.0, "C2z": 1.0, "C2y": 1.0, "C2x": 1.0},
        "B1": {"E": 1.0, "C2z": 1.0, "C2y": -1.0, "C2x": -1.0},
        "B2": {"E": 1.0, "C2z": -1.0, "C2y": 1.0, "C2x": -1.0},
        "B3": {"E": 1.0, "C2z": -1.0, "C2y": -1.0, "C2x": 1.0},
    }[name]
    if label in d2_values:
        return d2_values[label]
    if label == "i":
        return float(inversion_sign)
    sigma_to_rotation = {
        "sigma_xy": "C2z",
        "sigma_xz": "C2y",
        "sigma_yz": "C2x",
    }
    if label in sigma_to_rotation:
        return float(inversion_sign) * d2_values[sigma_to_rotation[label]]
    return 0.0


def _polyhedral_family_characters(
    labels: tuple[str, ...],
    group_key: str,
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if operation_matrices is None:
        matrix_rows = _matrix_polyhedral_label_characters(labels, group_key)
        if matrix_rows:
            return matrix_rows
    if group_key == "T":
        return _t_characters(labels, operation_matrices=operation_matrices)
    if group_key == "TD":
        return _td_like_characters(labels, operation_matrices=operation_matrices)
    if group_key == "TH":
        return _th_characters(labels, operation_matrices=operation_matrices)
    if group_key == "O":
        return _o_characters(labels, operation_matrices=operation_matrices)
    if group_key == "OH":
        return _oh_characters(labels, operation_matrices=operation_matrices)
    if group_key == "I":
        return _icosahedral_characters(
            labels,
            operation_matrices=operation_matrices,
            centrosymmetric=False,
        )
    if group_key == "IH":
        return _icosahedral_characters(
            labels,
            operation_matrices=operation_matrices,
            centrosymmetric=True,
        )
    return ()


def _t_characters(
    labels: tuple[str, ...],
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    table = {
        "A": {"E": 1.0, "C3": 1.0, "C2": 1.0},
        "E": {"E": 2.0, "C3": -1.0, "C2": 2.0},
        "T": {"E": 3.0, "C3": 0.0, "C2": -1.0},
    }
    classes = _polyhedral_operation_classes(labels, operation_matrices, family="TD")
    if classes is None or any(operation_class not in {"E", "C3", "C2"} for operation_class in classes):
        return ()
    return tuple(
        (irrep, tuple(chars[operation_class] for operation_class in classes))
        for irrep, chars in table.items()
    )


def _th_characters(
    labels: tuple[str, ...],
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if operation_matrices is None or len(operation_matrices) != len(labels):
        return ()
    base = {
        "A": {"E": 1.0, "C3": 1.0, "C2": 1.0},
        "E": {"E": 2.0, "C3": -1.0, "C2": 2.0},
        "T": {"E": 3.0, "C3": 0.0, "C2": -1.0},
    }
    classified: list[tuple[str, bool]] = []
    for label, matrix in zip(labels, operation_matrices):
        arr = np.asarray(matrix, dtype=float)
        reflected = float(np.linalg.det(arr)) < 0.0
        proper = -arr if reflected else arr
        operation_class = _td_matrix_class(label, proper)
        if operation_class not in {"E", "C3", "C2"}:
            return ()
        classified.append((str(operation_class), reflected))
    rows: list[tuple[str, tuple[float, ...]]] = []
    for suffix, parity in (("g", 1.0), ("u", -1.0)):
        for name, chars in base.items():
            rows.append(
                (
                    name + suffix,
                    tuple(
                        chars[operation_class] * (parity if reflected else 1.0)
                        for operation_class, reflected in classified
                    ),
                )
            )
    return tuple(rows)


def _o_characters(
    labels: tuple[str, ...],
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    table = {
        "A1": {"E": 1.0, "C3": 1.0, "C2_axis": 1.0, "C4": 1.0, "C2_edge": 1.0},
        "A2": {"E": 1.0, "C3": 1.0, "C2_axis": 1.0, "C4": -1.0, "C2_edge": -1.0},
        "E": {"E": 2.0, "C3": -1.0, "C2_axis": 2.0, "C4": 0.0, "C2_edge": 0.0},
        "T1": {"E": 3.0, "C3": 0.0, "C2_axis": -1.0, "C4": 1.0, "C2_edge": -1.0},
        "T2": {"E": 3.0, "C3": 0.0, "C2_axis": -1.0, "C4": -1.0, "C2_edge": 1.0},
    }
    classes = _polyhedral_operation_classes(labels, operation_matrices, family="OH")
    if classes is None or any(operation_class.startswith("i*") for operation_class in classes):
        return ()
    return tuple(
        (irrep, tuple(chars[operation_class] for operation_class in classes))
        for irrep, chars in table.items()
    )


def _td_like_characters(
    labels: tuple[str, ...],
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    table = {
        "A1": {"E": 1.0, "C3": 1.0, "C2": 1.0, "S4": 1.0, "sigma_d": 1.0},
        "A2": {"E": 1.0, "C3": 1.0, "C2": 1.0, "S4": -1.0, "sigma_d": -1.0},
        "E": {"E": 2.0, "C3": -1.0, "C2": 2.0, "S4": 0.0, "sigma_d": 0.0},
        "T1": {"E": 3.0, "C3": 0.0, "C2": -1.0, "S4": 1.0, "sigma_d": -1.0},
        "T2": {"E": 3.0, "C3": 0.0, "C2": -1.0, "S4": -1.0, "sigma_d": 1.0},
    }
    classes = _polyhedral_operation_classes(labels, operation_matrices, family="TD")
    if classes is None:
        return ()
    return tuple(
        (irrep, tuple(chars.get(operation_class, 0.0) for operation_class in classes))
        for irrep, chars in table.items()
    )


def _oh_characters(
    labels: tuple[str, ...],
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    proper_table = {
        "A1": {"E": 1.0, "C3": 1.0, "C2_axis": 1.0, "C4": 1.0, "C2_edge": 1.0},
        "A2": {"E": 1.0, "C3": 1.0, "C2_axis": 1.0, "C4": -1.0, "C2_edge": -1.0},
        "E": {"E": 2.0, "C3": -1.0, "C2_axis": 2.0, "C4": 0.0, "C2_edge": 0.0},
        "T1": {"E": 3.0, "C3": 0.0, "C2_axis": -1.0, "C4": 1.0, "C2_edge": -1.0},
        "T2": {"E": 3.0, "C3": 0.0, "C2_axis": -1.0, "C4": -1.0, "C2_edge": 1.0},
    }
    classes = _polyhedral_operation_classes(labels, operation_matrices, family="OH")
    if classes is None:
        return ()

    rows: list[tuple[str, tuple[float, ...]]] = []
    for suffix, parity in (("g", 1.0), ("u", -1.0)):
        for base_name, chars in proper_table.items():
            values: list[float] = []
            for operation_class in classes:
                reflected = operation_class.startswith("i*")
                proper_class = operation_class[2:] if reflected else operation_class
                value = chars.get(proper_class, 0.0)
                values.append(value * (parity if reflected else 1.0))
            rows.append((base_name + suffix, tuple(values)))
    return tuple(rows)


def _icosahedral_characters(
    labels: tuple[str, ...],
    *,
    operation_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
    centrosymmetric: bool,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    phi_bar = (1.0 - np.sqrt(5.0)) / 2.0
    table = {
        "A": {"E": 1.0, "C3": 1.0, "C2": 1.0, "C5": 1.0, "C5_2": 1.0},
        "T1": {"E": 3.0, "C3": 0.0, "C2": -1.0, "C5": phi, "C5_2": phi_bar},
        "T2": {"E": 3.0, "C3": 0.0, "C2": -1.0, "C5": phi_bar, "C5_2": phi},
        "G": {"E": 4.0, "C3": 1.0, "C2": 0.0, "C5": -1.0, "C5_2": -1.0},
        "H": {"E": 5.0, "C3": -1.0, "C2": 1.0, "C5": 0.0, "C5_2": 0.0},
    }
    classes = _icosahedral_operation_classes(labels, operation_matrices)
    if classes is None:
        return ()
    if not centrosymmetric:
        return tuple(
            (irrep, tuple(chars.get(operation_class, 0.0) for operation_class in classes))
            for irrep, chars in table.items()
        )

    rows: list[tuple[str, tuple[float, ...]]] = []
    for suffix, parity in (("g", 1.0), ("u", -1.0)):
        for base_name, chars in table.items():
            values: list[float] = []
            for operation_class in classes:
                reflected = operation_class.startswith("i*")
                proper_class = operation_class[2:] if reflected else operation_class
                value = chars.get(proper_class, 0.0)
                values.append(value * (parity if reflected else 1.0))
            rows.append((base_name + suffix, tuple(values)))
    return tuple(rows)


def _matrix_polyhedral_label_characters(
    labels: tuple[str, ...],
    group_key: str,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if group_key == "TH":
        base = _matrix_polyhedral_label_characters(labels, "T")
        parity = tuple(
            -1.0
            if label == "i"
            or label.startswith("i")
            or label.startswith("sigma")
            or label.startswith("S")
            else 1.0
            for label in labels
        )
        return tuple((f"{name}g", chars) for name, chars in base) + tuple(
            (f"{name}u", tuple(char * sign for char, sign in zip(chars, parity)))
            for name, chars in base
        )
    if group_key in {"TD", "O"}:
        table = {
            "A1": (1.0, 1.0, 1.0, 1.0, 1.0),
            "A2": (1.0, 1.0, 1.0, -1.0, -1.0),
            "E": (2.0, -1.0, 2.0, 0.0, 0.0),
            "T1": (3.0, 0.0, -1.0, 1.0, -1.0),
            "T2": (3.0, 0.0, -1.0, -1.0, 1.0),
        }
        return tuple(
            (
                name,
                tuple(_matrix_poly_char(label, values) for label in labels),
            )
            for name, values in table.items()
        )
    root = group_key.rstrip("HD")
    if root == "T":
        table = {
            "A": (1.0, 1.0, 1.0),
            "E": (2.0, -1.0, 2.0),
            "T": (3.0, 0.0, -1.0),
        }
        return tuple(
            (
                name,
                tuple(_matrix_poly_char(label, values) for label in labels),
            )
            for name, values in table.items()
        )
    if group_key == "OH":
        base = _matrix_polyhedral_label_characters(labels, "O")
        parity = tuple(
            -1.0 if label == "i" or label.startswith("sigma") or label.startswith("S") else 1.0
            for label in labels
        )
        rows: list[tuple[str, tuple[float, ...]]] = []
        for name, chars in base:
            rows.append((f"{name}g", chars))
            rows.append((f"{name}u", tuple(char * sign for char, sign in zip(chars, parity))))
        return tuple(rows)
    if group_key == "IH":
        base = _matrix_polyhedral_label_characters(labels, "I")
        parity = tuple(
            -1.0 if label == "i" or label.startswith("sigma") or label.startswith("S") else 1.0
            for label in labels
        )
        return tuple((f"{name}g", chars) for name, chars in base) + tuple(
            (f"{name}u", tuple(char * sign for char, sign in zip(chars, parity)))
            for name, chars in base
        )
    if root == "I":
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        phi_bar = (1.0 - np.sqrt(5.0)) / 2.0
        table = {
            "A": (1.0, 1.0, 1.0, 1.0, 1.0),
            "T1": (3.0, 0.0, -1.0, phi, phi_bar),
            "T2": (3.0, 0.0, -1.0, phi_bar, phi),
            "G": (4.0, 1.0, 0.0, -1.0, -1.0),
            "H": (5.0, -1.0, 1.0, 0.0, 0.0),
        }
        return tuple(
            (
                name,
                tuple(_matrix_poly_char(label, values) for label in labels),
            )
            for name, values in table.items()
        )
    return ()


def _matrix_poly_char(label: str, values: tuple[float, ...]) -> float:
    if label == "E":
        return float(values[0])
    if "C3" in label:
        return float(values[1])
    if "C2" in label:
        return float(values[2])
    if "C4" in label or "S4" in label:
        return float(values[3] if len(values) > 3 else 0.0)
    if "C5" in label:
        if "2" in label or "3" in label:
            return float(values[4] if len(values) > 4 else 0.0)
        return float(values[3] if len(values) > 3 else 0.0)
    if label.startswith("sigma"):
        return float(values[4] if len(values) > 4 else values[0])
    return float(values[0])


def _icosahedral_operation_classes(
    labels: tuple[str, ...],
    matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
) -> tuple[str, ...] | None:
    if matrices is not None:
        matrix_tuple = tuple(np.asarray(matrix, dtype=float) for matrix in matrices)
        if len(matrix_tuple) != len(labels):
            return None
        classes: list[str] = []
        for label, matrix in zip(labels, matrix_tuple):
            operation_class = _icosahedral_matrix_class(label, matrix)
            if operation_class is None:
                return None
            classes.append(operation_class)
        return tuple(classes)

    classes = []
    for label in labels:
        operation_class = _icosahedral_label_class(label)
        if operation_class is None:
            return None
        classes.append(operation_class)
    return tuple(classes)


def _polyhedral_operation_classes(
    labels: tuple[str, ...],
    matrices: tuple[tuple[tuple[float, ...], ...], ...]
    | list[tuple[tuple[float, ...], ...]]
    | None,
    *,
    family: str,
) -> tuple[str, ...] | None:
    if matrices is not None:
        matrix_tuple = tuple(np.asarray(matrix, dtype=float) for matrix in matrices)
        if len(matrix_tuple) != len(labels):
            return None
        c4_squares: tuple[np.ndarray, ...] | None = None
        if family == "OH":
            proper_matrices = tuple(
                matrix if float(np.linalg.det(matrix)) > 0.0 else -matrix
                for matrix in matrix_tuple
            )
            c4_squares = tuple(
                matrix @ matrix
                for matrix in proper_matrices
                if _close(float(np.trace(matrix)), 1.0)
            )
        classes: list[str] = []
        for label, matrix in zip(labels, matrix_tuple):
            operation_class = (
                _td_matrix_class(label, matrix)
                if family == "TD"
                else _oh_matrix_class(label, matrix, c4_squares=c4_squares)
            )
            if operation_class is None:
                return None
            classes.append(operation_class)
        return tuple(classes)

    classes = []
    for label in labels:
        operation_class = _td_label_class(label) if family == "TD" else _oh_label_class(label)
        if operation_class is None:
            return None
        classes.append(operation_class)
    return tuple(classes)


def _td_matrix_class(label: str, matrix: np.ndarray) -> str | None:
    if _is_matrix_close(matrix, np.eye(3)) or label == "E":
        return "E"
    det = float(np.linalg.det(matrix))
    trace = float(np.trace(matrix))
    if det > 0.0:
        if _close(trace, 0.0):
            return "C3"
        if _close(trace, -1.0):
            return "C2"
    if det < 0.0:
        if _close(trace, -1.0):
            return "S4"
        if _close(trace, 1.0):
            return "sigma_d"
    return _td_label_class(label)


def _oh_matrix_class(
    label: str,
    matrix: np.ndarray,
    *,
    c4_squares: tuple[np.ndarray, ...] | None = None,
) -> str | None:
    if _is_matrix_close(matrix, np.eye(3)) or label == "E":
        return "E"
    det = float(np.linalg.det(matrix))
    proper = matrix if det > 0.0 else -matrix
    proper_class = _octahedral_proper_matrix_class(
        label,
        proper,
        c4_squares=c4_squares,
    )
    if proper_class is None:
        return _oh_label_class(label)
    return f"i*{proper_class}" if det < 0.0 else proper_class


def _icosahedral_matrix_class(label: str, matrix: np.ndarray) -> str | None:
    if _is_matrix_close(matrix, np.eye(3)) or label == "E":
        return "E"
    det = float(np.linalg.det(matrix))
    proper = matrix if det > 0.0 else -matrix
    proper_class = _icosahedral_proper_matrix_class(label, proper)
    if proper_class is None:
        return _icosahedral_label_class(label)
    return f"i*{proper_class}" if det < 0.0 else proper_class


def _icosahedral_proper_matrix_class(label: str, matrix: np.ndarray) -> str | None:
    if _is_matrix_close(matrix, np.eye(3)) or label == "E":
        return "E"
    trace = float(np.trace(matrix))
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    phi_bar = (1.0 - np.sqrt(5.0)) / 2.0
    if _close(trace, 0.0):
        return "C3"
    if _close(trace, -1.0):
        return "C2"
    if _close(trace, phi):
        return "C5"
    if _close(trace, phi_bar):
        return "C5_2"
    return None


def _octahedral_proper_matrix_class(
    label: str,
    matrix: np.ndarray,
    *,
    c4_squares: tuple[np.ndarray, ...] | None = None,
) -> str | None:
    if _is_matrix_close(matrix, np.eye(3)) or label == "E":
        return "E"
    trace = float(np.trace(matrix))
    if _close(trace, 0.0):
        return "C3"
    if _close(trace, 1.0):
        return "C4"
    if _close(trace, -1.0):
        if c4_squares is not None:
            return (
                "C2_axis"
                if any(_is_matrix_close(matrix, square) for square in c4_squares)
                else "C2_edge"
            )
        return "C2_axis" if _is_coordinate_axis_c2(matrix) else "C2_edge"
    return None


def _td_label_class(label: str) -> str | None:
    if label == "E":
        return "E"
    if "C3" in label:
        return "C3"
    if "C2" in label:
        return "C2"
    if "S4" in label:
        return "S4"
    if label.startswith("sigma"):
        return "sigma_d"
    return None


def _oh_label_class(label: str) -> str | None:
    if label == "E":
        return "E"
    if label == "i":
        return "i*E"
    if "S6" in label:
        return "i*C3"
    if "S4" in label:
        return "i*C4"
    if label.startswith("sigma"):
        return "i*C2_axis" if label in {"sigma_xy", "sigma_xz", "sigma_yz"} else "i*C2_edge"
    if "C3" in label:
        return "C3"
    if "C4" in label:
        return "C4"
    if "C2" in label:
        return "C2_axis" if label in {"C2x", "C2y", "C2z"} else "C2_edge"
    return None


def _icosahedral_label_class(label: str) -> str | None:
    if label == "E":
        return "E"
    if label == "i":
        return "i*E"
    reflected = label.startswith("ih_i_") or label.startswith("i*")
    text = label[5:] if label.startswith("ih_i_") else label
    text = text[2:] if text.startswith("i*") else text
    if "C3" in text:
        operation_class = "C3"
    elif "C2" in text:
        operation_class = "C2"
    elif "C5_2" in text or "C5^2" in text:
        operation_class = "C5_2"
    elif "C5" in text:
        operation_class = "C5"
    else:
        return None
    return f"i*{operation_class}" if reflected else operation_class


def _is_coordinate_axis_c2(matrix: np.ndarray) -> bool:
    rounded = np.rint(matrix)
    if not np.allclose(matrix, rounded, atol=1.0e-8):
        return False
    diagonal = np.diag(rounded)
    return bool(np.count_nonzero(np.abs(diagonal) > 0.5) == 3)


def _is_matrix_close(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, atol=1.0e-8))


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-8


def _e_name(order: int, n: int) -> str:
    if order == 1 and n in {3, 4}:
        return "E"
    return f"E{order}"


def _principal_rotation_power(label: str, n: int) -> int | None:
    if label == "E":
        return 0
    match = re.fullmatch(r"C(\d+)([xyz])\^(\d+)", label)
    if match:
        order, axis, power = match.groups()
        if axis != "z":
            return None
        order_int = int(order)
        power_int = int(power)
        relative = n * power_int / order_int
        nearest = int(round(relative))
        if abs(relative - nearest) > 1.0e-8:
            return None
        return nearest % n
    match = re.fullmatch(r"C2([xyz])", label)
    if match:
        axis = match.group(1)
        if axis == "z" and n % 2 == 0:
            return n // 2
    return None


def _cnh_rotation_power(label: str, n: int) -> int | None:
    if label == "sigma_xy":
        return 0
    if label == "i" and n % 2 == 0:
        return n // 2
    match = re.fullmatch(r"sigma_h\*C(\d+)z\^(\d+)", label)
    if match:
        order, power = match.groups()
        relative = n * int(power) / int(order)
        nearest = int(round(relative))
        if abs(relative - nearest) > 1.0e-8:
            return None
        return nearest % n
    return _principal_rotation_power(label, n)


def _is_vertical_reflection(label: str) -> bool:
    return label in {"sigma_xz", "sigma_yz"} or label.startswith("sigma_v")


def _vertical_reflection_index(label: str, n: int) -> int:
    match = re.fullmatch(r"sigma_v_(\d+)_(\d+)", label)
    if match:
        order, index = match.groups()
        return int(round(n * int(index) / int(order))) % n
    if label == "sigma_xz":
        return 0
    if label == "sigma_yz":
        return n // 2 if n % 2 == 0 else 1
    return 0


def _is_horizontal_reflected(label: str, n: int) -> bool:
    if label == "sigma_xy" or label.startswith("sigma_h"):
        return True
    return bool(label == "i" and n % 2 == 0)


def _is_c2_prime(label: str) -> bool:
    return label.startswith("C2_xy") or label in {"C2x", "C2y"}


def _c2_prime_index(label: str, n: int) -> int:
    match = re.fullmatch(r"C2_xy_(\d+)_(\d+)", label)
    if match:
        order, index = match.groups()
        return int(round(n * int(index) / int(order))) % n
    if label == "C2x":
        return 0
    if label == "C2y":
        return n // 2 if n % 2 == 0 else 1
    return 0


def _canonical_operation_label(label: str) -> str:
    text = str(label)
    if text in {"E", "i"} or text.startswith("sigma"):
        return text
    match = re.match(r"C(\d+)([xyz])\^(\d+)", text)
    if match:
        order, axis, _power = match.groups()
        return f"C2{axis}" if int(order) == 2 else text
    return text

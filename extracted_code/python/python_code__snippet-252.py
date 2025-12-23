def _get_env(environment: Dict[str, str], name: str) -> str:
    value: Union[str, Undefined] = environment.get(name, _undefined)

    if isinstance(value, Undefined):
        raise UndefinedEnvironmentName(
            f"{name!r} does not exist in evaluation environment."
        )

    return value


def _evaluate_markers(markers: List[Any], environment: Dict[str, str]) -> bool:
    groups: List[List[bool]] = [[]]

    for marker in markers:
        assert isinstance(marker, (list, tuple, str))

        if isinstance(marker, list):
            groups[-1].append(_evaluate_markers(marker, environment))
        elif isinstance(marker, tuple):
            lhs, op, rhs = marker

            if isinstance(lhs, Variable):
                lhs_value = _get_env(environment, lhs.value)
                rhs_value = rhs.value
            else:
                lhs_value = lhs.value
                rhs_value = _get_env(environment, rhs.value)

            groups[-1].append(_eval_op(lhs_value, op, rhs_value))
        else:
            assert marker in ["and", "or"]
            if marker == "or":
                groups.append([])

    return any(all(item) for item in groups)


def format_full_version(info: "sys._version_info") -> str:
    version = "{0.major}.{0.minor}.{0.micro}".format(info)
    kind = info.releaselevel
    if kind != "final":
        version += kind[0] + str(info.serial)
    return version


def default_environment() -> Dict[str, str]:
    iver = format_full_version(sys.implementation.version)
    implementation_name = sys.implementation.name
    return {
        "implementation_name": implementation_name,
        "implementation_version": iver,
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_full_version": platform.python_version(),
        "platform_python_implementation": platform.python_implementation(),
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "sys_platform": sys.platform,
    }


class Marker:
    def __init__(self, marker: str) -> None:
        try:
            self._markers = _coerce_parse_result(MARKER.parseString(marker))
        except ParseException as e:
            raise InvalidMarker(
                f"Invalid marker: {marker!r}, parse error at "
                f"{marker[e.loc : e.loc + 8]!r}"
            )

    def __str__(self) -> str:
        return _format_marker(self._markers)

    def __repr__(self) -> str:
        return f"<Marker('{self}')>"

    def evaluate(self, environment: Optional[Dict[str, str]] = None) -> bool:
        """Evaluate a marker.

        Return the boolean from evaluating the given marker against the
        environment. environment is an optional argument to override all or
        part of the determined environment.

        The environment is determined from the current Python process.
        """
        current_environment = default_environment()
        if environment is not None:
            current_environment.update(environment)

        return _evaluate_markers(self._markers, current_environment)



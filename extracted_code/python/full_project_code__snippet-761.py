if isinstance(header_part, str):
    validator = _HEADER_VALIDATORS_STR[header_validator_index]
elif isinstance(header_part, bytes):
    validator = _HEADER_VALIDATORS_BYTE[header_validator_index]
else:
    raise InvalidHeader(
        f"Header part ({header_part!r}) from {header} "
        f"must be of type str or bytes, not {type(header_part)}"
    )

if not validator.match(header_part):
    header_kind = "name" if header_validator_index == 0 else "value"
    raise InvalidHeader(
        f"Invalid leading whitespace, reserved character(s), or return"
        f"character(s) in header {header_kind}: {header_part!r}"
    )



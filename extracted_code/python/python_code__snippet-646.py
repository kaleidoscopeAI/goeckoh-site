def json_name(field: str) -> str:
    return field.lower().replace("-", "_")


def msg_to_json(msg: Message) -> Dict[str, Any]:
    """Convert a Message object into a JSON-compatible dictionary."""

    def sanitise_header(h: Union[Header, str]) -> str:
        if isinstance(h, Header):
            chunks = []
            for bytes, encoding in decode_header(h):
                if encoding == "unknown-8bit":
                    try:
                        # See if UTF-8 works
                        bytes.decode("utf-8")
                        encoding = "utf-8"
                    except UnicodeDecodeError:
                        # If not, latin1 at least won't fail
                        encoding = "latin1"
                chunks.append((bytes, encoding))
            return str(make_header(chunks))
        return str(h)

    result = {}
    for field, multi in METADATA_FIELDS:
        if field not in msg:
            continue
        key = json_name(field)
        if multi:
            value: Union[str, List[str]] = [
                sanitise_header(v) for v in msg.get_all(field)  # type: ignore
            ]
        else:
            value = sanitise_header(msg.get(field))  # type: ignore
            if key == "keywords":
                # Accept both comma-separated and space-separated
                # forms, for better compatibility with old data.
                if "," in value:
                    value = [v.strip() for v in value.split(",")]
                else:
                    value = value.split()
        result[key] = value

    payload = msg.get_payload()
    if payload:
        result["description"] = payload

    return result



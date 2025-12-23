class SearchCommand(Command, SessionCommandMixin):
    """Search for PyPI packages whose name or summary contains <query>."""

    usage = """
      %prog [options] <query>"""
    ignore_require_venv = True

    def add_options(self) -> None:
        self.cmd_opts.add_option(
            "-i",
            "--index",
            dest="index",
            metavar="URL",
            default=PyPI.pypi_url,
            help="Base URL of Python Package Index (default %default)",
        )

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options: Values, args: List[str]) -> int:
        if not args:
            raise CommandError("Missing required argument (search query).")
        query = args
        pypi_hits = self.search(query, options)
        hits = transform_hits(pypi_hits)

        terminal_width = None
        if sys.stdout.isatty():
            terminal_width = shutil.get_terminal_size()[0]

        print_results(hits, terminal_width=terminal_width)
        if pypi_hits:
            return SUCCESS
        return NO_MATCHES_FOUND

    def search(self, query: List[str], options: Values) -> List[Dict[str, str]]:
        index_url = options.index

        session = self.get_default_session(options)

        transport = PipXmlrpcTransport(index_url, session)
        pypi = xmlrpc.client.ServerProxy(index_url, transport)
        try:
            hits = pypi.search({"name": query, "summary": query}, "or")
        except xmlrpc.client.Fault as fault:
            message = "XMLRPC request failed [code: {code}]\n{string}".format(
                code=fault.faultCode,
                string=fault.faultString,
            )
            raise CommandError(message)
        assert isinstance(hits, list)
        return hits


def transform_hits(hits: List[Dict[str, str]]) -> List["TransformedHit"]:
    """
    The list from pypi is really a list of versions. We want a list of
    packages with the list of versions stored inline. This converts the
    list from pypi into one we can use.
    """
    packages: Dict[str, "TransformedHit"] = OrderedDict()
    for hit in hits:
        name = hit["name"]
        summary = hit["summary"]
        version = hit["version"]

        if name not in packages.keys():
            packages[name] = {
                "name": name,
                "summary": summary,
                "versions": [version],
            }
        else:
            packages[name]["versions"].append(version)

            # if this is the highest version, replace summary and score
            if version == highest_version(packages[name]["versions"]):
                packages[name]["summary"] = summary

    return list(packages.values())


def print_dist_installation_info(name: str, latest: str) -> None:
    env = get_default_environment()
    dist = env.get_distribution(name)
    if dist is not None:
        with indent_log():
            if dist.version == latest:
                write_output("INSTALLED: %s (latest)", dist.version)
            else:
                write_output("INSTALLED: %s", dist.version)
                if parse_version(latest).pre:
                    write_output(
                        "LATEST:    %s (pre-release; install"
                        " with `pip install --pre`)",
                        latest,
                    )
                else:
                    write_output("LATEST:    %s", latest)


def print_results(
    hits: List["TransformedHit"],
    name_column_width: Optional[int] = None,
    terminal_width: Optional[int] = None,

    class RequestModule(sys.modules[__name__].__class__):
        def __call__(self, *args, **kwargs):
            """
            If user tries to call this module directly urllib3 v2.x style raise an error to the user
            suggesting they may need urllib3 v2
            """
            raise TypeError(
                "'module' object is not callable\n"
                "urllib3.request() method is not supported in this release, "
                "upgrade to urllib3 v2 to use it\n"
                "see https://urllib3.readthedocs.io/en/stable/v2-migration-guide.html"
            )

    sys.modules[__name__].__class__ = RequestModule



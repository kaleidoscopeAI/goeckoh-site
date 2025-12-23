    class Container(object):
        """
        A generic container for when multiple values need to be returned
        """

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)



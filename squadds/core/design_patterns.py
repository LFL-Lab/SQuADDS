class SingletonMeta(type):
    """
    Metaclass for implementing the Singleton design pattern.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Call method for the Singleton metaclass.

        This method ensures that only one instance of the class is created and returned.

        Args:
            cls: The class being instantiated.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The instance of the class.

        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

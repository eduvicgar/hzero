import inspect
from functools import wraps
from typing import Callable, Any


def validate_d_nonnegative(func: Callable) -> Callable:
    """
    Decorator that validates the 'd' argument of a function, ensuring it is positive.

    This decorator inspects the arguments passed to the decorated function and checks whether the 'd'
    parameter is present and positive. If 'alpha' is provided and is not within this range, a ValueError
    is raised.

    :param func: The function to be decorated. It must accept an 'd' parameter either directly or via **kwargs.
    :return: The wrapped function with added validation for the 'd' parameter.
    :raises ValueError: If 'd' is provided and is not positive.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound_args = (inspect.signature(func)  # Obtiene firma de la función (sus parámetros)
                      .bind(*args, **kwargs))  # Asocia valores dados con los parámetros definidos en la firma
        bound_args.apply_defaults()  # Completa con parámetros que vienen por defecto
        d = bound_args.arguments.get("d")

        if d is not None and d < 0:
            raise ValueError(
                f"Function '{func.__name__}': d must be between positive, got {d}"
            )

        return func(*args, **kwargs)
    return wrapper

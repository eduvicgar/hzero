import inspect
from functools import wraps
from typing import Callable, Any


def validate_alpha(func: Callable) -> Callable:
    """
    Decorator that validates the 'alpha' argument of a function, ensuring it is a float between 0 and 1 (exclusive).

    This decorator inspects the arguments passed to the decorated function and checks whether the 'alpha'
    parameter is present and within the valid range (0, 1). If 'alpha' is provided and is not within this range,
    a ValueError is raised.

    :param func: The function to be decorated. It must accept an 'alpha' parameter either directly or via **kwargs.
    :return: The wrapped function with added validation for the 'alpha' parameter.
    :raises ValueError: If 'alpha' is provided and is not between 0 and 1.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound_args = (inspect.signature(func) # Obtiene firma de la función (sus parámetros)
                      .bind(*args, **kwargs)) # Asocia valores dados con los parámetros definidos en la firma
        bound_args.apply_defaults() # Completa con parámetros que vienen por defecto
        alpha = bound_args.arguments.get("alpha")

        if alpha is not None and not 0 < alpha < 1:
            raise ValueError(
                f"Function '{func.__name__}': alpha must be between 0 and 1, got {alpha}"
            )
        return func(*args, **kwargs)

    return wrapper

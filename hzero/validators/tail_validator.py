import inspect
from functools import wraps
from typing import Callable, Tuple, Optional, Any


def validate_tail(allowed: Tuple[Optional[str], ...] = ("left", "right", "bilateral", None)) -> Callable:
    """
    Decorator that validates the 'tail' argument of a function, ensuring it matches one of the allowed values.

    This decorator inspects the 'tail' argument passed to the decorated function and checks whether it is
    included in the set of allowed values. If 'tail' is not one of the allowed options, a ValueError is raised.

    :param allowed: A tuple of valid values for the 'tail' argument. Defaults to ("left", "right", "bilateral", None).
    :return: A decorator that adds validation logic to a function for the 'tail' argument.
    :raises ValueError: If 'tail' is provided and is not among the allowed values.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound_args = (inspect.signature(func)  # Obtiene firma de la función (sus parámetros)
                          .bind(*args, **kwargs))  # Asocia valores dados con los parámetros definidos en la firma
            bound_args.apply_defaults()  # Completa con parámetros que vienen por defecto
            tail = bound_args.arguments.get("tail")

            if tail not in allowed:
                raise ValueError(
                    f"Function '{func.__name__}': tail must be one of {allowed}"
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator

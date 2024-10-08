from typing import Any, List, Callable


class Pipeline:
    """In contrast to torchvision.transforms.Compose, this class is a simple pipeline that can be used with any function
    and does NOT unzip the arguemnts before passing them to the next function. This is useful when you want to pass
    an array to the next function, but the function expects the array as a single argument.
    """

    def __init__(self, pipes: List[Callable]):
        self.pipes = pipes

    def __call__(self, *args: Any, **kwargs) -> Any:
        next_arg = self.pipes[0](*args, **kwargs)

        for pipe in self.pipes[1:]:
            next_arg = pipe(next_arg)

        return next_arg

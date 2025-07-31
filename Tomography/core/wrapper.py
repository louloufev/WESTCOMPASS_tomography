import copy
from functools import wraps

def log_args_changes(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save deep copies of original args/kwargs
        args_before = copy.deepcopy(args)
        kwargs_before = copy.deepcopy(kwargs)
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Compare after execution
        print(f"\n--- {func.__name__}() Changes ---")
        print("Args before:", args_before)
        print("Args after :", args)
        print("Kwargs before:", kwargs_before)
        print("Kwargs after :", kwargs)
        print("--------------------------\n")
        
        return result
    return wrapper
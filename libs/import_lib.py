import importlib
import logging

def get_class(lib_file, class_name, base_class):
    logging.info(f"Get Class {class_name} from {lib_file}")
    try:
        my_module = importlib.import_module(lib_file)
        if hasattr(my_module, class_name):
            # Access the class using getattr
            new_class = getattr(my_module, class_name)
            logging.info("New Class created")
            if not issubclass(new_class, base_class):
                raise Exception(f"Error: Class {class_name} has wrong Base Class")
            return new_class()
        else:
            raise Exception(f"Error: Class {class_name} not found in Module {lib_file}")
    except ImportError:
        raise Exception(f"Error: Module '{lib_file}' not found.")
    except AttributeError:
        raise Exception(f"Error: Class '{class_name}' not found in module '{lib_file}'.")
    except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
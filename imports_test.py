import sys

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"Successfully imported {module_name}")
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
    except Exception as e:
        print(f"Error while importing {module_name}: {e}")

def main():
    # Test import of main package
    test_import("squadds")

    # Test import of submodules
    submodules = [
        "squadds.calcs",
        "squadds.core",
        "squadds.database",
        "squadds.interpolations",
        "squadds.core.utils",
        "squadds.core.design_patterns",
        "squadds.core.analysis",
        "squadds.database.utils",
        "squadds.database.design_patterns",
        "squadds.database.analysis",
        "squadds.interpolations.interpolator",
        "squadds.calcs.qubit",
        "squadds.calcs.transmon_cross"
    ]

    for submodule in submodules:
        test_import(submodule)

if __name__ == "__main__":
    main()

import importlib.util

# Check if streamlit module is installed
try:
    importlib.util.find_spec("streamlit")
    print("streamlit is installed.")
except ImportError:
    print("streamlit is not installed.")
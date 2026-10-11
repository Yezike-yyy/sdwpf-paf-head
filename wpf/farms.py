"""Wind farms of the study: dataset directory and results directory, relative to the wpf/ root.

SDWPF keeps its original paths (dataset/, results/<tag>/...). The two Cubico farms of plan revision 3 (validation
on independent farms) use datasets/<farm>/ (prepare_cubico.py) and results/<farm>/<tag>/...
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FARMS = {
    "sdwpf": dict(data="dataset", results="results"),
    "penmanshiel": dict(data=os.path.join("datasets", "penmanshiel"), results=os.path.join("results", "penmanshiel")),
    "kelmarsh": dict(data=os.path.join("datasets", "kelmarsh"), results=os.path.join("results", "kelmarsh")),
}
NEW_FARMS = ["penmanshiel", "kelmarsh"]


def data_dir(farm):
    return os.path.join(ROOT, FARMS[farm]["data"])


def results_dir(farm):
    return os.path.join(ROOT, FARMS[farm]["results"])


def display_name(farm):
    return "SDWPF" if farm == "sdwpf" else farm.capitalize()


def report_name(farm, name):
    """reports/<name> for SDWPF (unchanged), reports/<farm>_<name> for the other farms."""
    return name if farm == "sdwpf" else "%s_%s" % (farm, name)

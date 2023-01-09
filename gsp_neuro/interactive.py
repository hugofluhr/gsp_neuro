import ipywidgets as widgets
from IPython.display import display


def select_subject(subjects_ids):
    w = widgets.Select(options=subjects_ids, value=subjects_ids[0], description='Subjects :')
    display(w)
    return w

def select_scale():
    scales = [1, 2, 3, 4, 5]
    w = widgets.Select(options=scales, value=scales[0], description = "Scale : ")
    display(w)
    return w
from IPython.core.display import display, HTML
import ase.visualize

def view(atoms):
    
    obj = ase.visualize.view(atoms, viewer="x3d")
    string = obj.data
    string = string.replace("https://www.x3dom.org/x3dom/release/x3dom.css", "https://fullmetalfelix.github.io/ML-CSC-tutorial/x3dom.css")
    string = string.replace("https://www.x3dom.org/x3dom/release/x3dom.js", "https://fullmetalfelix.github.io/ML-CSC-tutorial/x3dom.js")
    display(HTML(string))


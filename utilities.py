import base64
from tempfile import NamedTemporaryFile
from IPython.display import HTML, display
import matplotlib.pyplot as plt

IMG_TAG = """<img src="data:image/gif;base64,{0}" alt="some_text">"""

def anim_to_gif(anim, fps=10):
    data="0"
    with NamedTemporaryFile(suffix='.gif') as f:
        anim.save(f.name, writer='imagemagick', fps=fps);
        data = open(f.name, "rb").read()
        data = str(base64.b64encode(data), 'utf-8')
    return IMG_TAG.format(data)

def display_animation(anim, **kwords):
    plt.close(anim._fig)
    display(HTML(anim_to_gif(anim, **kwords)))

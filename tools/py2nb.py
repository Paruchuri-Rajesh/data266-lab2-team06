"""Convert percent-format .py files (# %% / # %% [markdown]) into .ipynb notebooks.
Usage: python tools/py2nb.py notebooks_src/p1_rag.py notebooks/p1_rag.ipynb
"""
import sys, nbformat

def convert(src, dst):
    cells, kind, buf = [], None, []
    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if kind == "md":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        elif text:
            cells.append(nbformat.v4.new_code_cell(text))
    for line in open(src, encoding="utf-8").read().splitlines():
        if line.startswith("# %%"):
            flush(); buf = []
            kind = "md" if "[markdown]" in line else "code"
        else:
            buf.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"
    nbformat.write(nb, dst)
    print(f"wrote {dst} ({len(cells)} cells)")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])

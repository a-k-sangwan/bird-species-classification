# Builds bird_species_colab.ipynb from bird_species.py, with plots.py
# pasted into the "# [plots.py]" cell.

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def lines(text):
    out = text.strip("\n").split("\n")
    return [l + "\n" for l in out[:-1]] + [out[-1]]


def main():
    src = open(os.path.join(HERE, "bird_species.py")).read()
    plots = open(os.path.join(HERE, "plots.py")).read()
    cells = []
    for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        if header.strip() == "[markdown]":
            md = "\n".join(re.sub(r"^# ?", "", l) for l in body.strip("\n").split("\n"))
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": lines(md)})
            continue
        if body.strip().startswith("# [plots.py]"):
            body = plots
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": lines(body)})
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = os.path.join(HERE, "bird_species_colab.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()

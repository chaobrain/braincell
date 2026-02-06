# _test.py
import random
from _vis_tree import demo_tree, install_tree_vis
install_tree_vis()

trees = [
    demo_tree(depth=random.randint(5, 10), prob_branch=random.uniform(0.6, 0.9))
    for _ in range(3)
]

for i, tree in enumerate(trees):
    dot = tree.vis.plot_2d(
        layout="depth",
        label_mode="id",
        rankdir="TB",
        color_mode="depth",
        color_cfg={"start": (100,200,200), "end": (100,50,150), "edge":"source"},
        save_path=f"./plot_2d/tree_demo_{i}",
        fmt="pdf",
    )
    print(f"Saved tree_demo_{i}.pdf")

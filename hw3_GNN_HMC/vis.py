import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    pos = torch.load("ptlog/ddi_position.ptlog")
    s = torch.load("ptlog/ddi_structure.ptlog")
    i = [i* 5 for i in range(len(pos))]
    plt.plot(i, pos, label="positional representation")
    plt.plot(i, s, label="structural representation")
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Hits@20")
    plt.legend()
    plt.savefig("graph.png")
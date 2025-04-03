import torch

def test(device):
    print(f"---{device}---")
    num_rows = 3
    num_cols = 4
    j = 0

    S = torch.zeros(num_rows, num_cols, 2, device=device)
    indices = torch.tensor([[1], [0], [2]], dtype=torch.long, device=device)

    # using scatter_
    S_scatter = S.clone()
    ones = torch.ones(num_rows, device=device)
    S_scatter[:, :, j].scatter_(1, indices, ones.unsqueeze(1))
    print("using scatter_:")
    print(f"{S_scatter[:, :, j]=}")
    print(f"{S_scatter.count_nonzero()=}")

    # using advanced indexing
    S_indexing = S.clone()
    row_indices = torch.arange(num_rows, device=device)
    S_indexing[row_indices, indices.squeeze(), j] = 1
    print("\nusing advanced indexing:")
    print(f"{S_indexing[:, :, j]=}")
    print(f"{S_indexing.count_nonzero()=}")

    print(f"\n{torch.equal(S_scatter, S_indexing)=}")

test("cpu")
print("")
test("mps")

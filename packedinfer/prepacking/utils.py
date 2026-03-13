from typing import List, Union
import torch
from torch import Tensor
from ortools.linear_solver import pywraplp
import binpacking


# As implemented here:
# https://github.com/pytorch/pytorch/issues/10536#issuecomment-1320935162
def left_pad_sequence(
    sequences: Union[Tensor, List[Tensor]],
    batch_first: bool = True,
    padding_value: float = 0.0,
) -> Tensor:

    sequences = tuple(map(lambda s: s.flip(0), sequences))
    padded_sequence = torch._C._nn.pad_sequence(sequences, batch_first, padding_value)
    _seq_dim = padded_sequence.dim()
    padded_sequence = padded_sequence.flip(-_seq_dim + batch_first)
    return padded_sequence


def greedy_packing(length_dict, max_bin_size):
    return binpacking.to_constant_volume(length_dict, max_bin_size)


# https://developers.google.com/optimization/pack/bin_packing
def integer_program_packing(length_dict, max_bin_size):
    data = {}
    data["items"] = list(length_dict.keys())
    data["weights"] = list(length_dict.values())
    data["bins"] = data["items"]
    data["bin_capacity"] = max_bin_size

    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return
    x = {}
    for i in data["items"]:
        for j in data["bins"]:
            x[(i, j)] = solver.IntVar(0, 1, "x_%i_%i" % (i, j))
    y = {}
    for j in data["bins"]:
        y[j] = solver.IntVar(0, 1, "y[%i]" % j)

    for i in data["items"]:
        solver.Add(sum(x[i, j] for j in data["bins"]) == 1)

    for j in data["bins"]:
        solver.Add(sum(x[(i, j)] * data["weights"][i] for i in data["items"]) <= y[j] * data["bin_capacity"])

    solver.Minimize(solver.Sum([y[j] for j in data["bins"]]))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        result = []
        for j in data["bins"]:
            if y[j].solution_value() == 1:
                bin_dict = {}
                for i in data["items"]:
                    if x[i, j].solution_value() > 0:
                        bin_dict[i] = data["weights"][i]
                result.append(bin_dict)
    else:
        raise ("The problem does not have an optimal solution.")

    return result

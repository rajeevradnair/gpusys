import torch

DTYPE_TOLERANCES = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}

def get_tolerances(dtype):
    if dtype not in DTYPE_TOLERANCES:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return DTYPE_TOLERANCES[dtype]



def compare_tensors_old_1(actual, expected, atol=1e-5, rtol=1e-5):
    diff = torch.abs(actual - expected)

    denominator = torch.clamp(torch.abs(expected), min=1e-12)
    rel_error = diff / denominator

    worst_index = diff.argmax().item()

    passed = torch.allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    )

    return {
        "passed": passed,
        "max_abs_error": diff.max().item(),
        "max_rel_error": rel_error.max().item(),
        "worst_index": worst_index,
        "expected": expected.flatten()[worst_index].item(),
        "actual": actual.flatten()[worst_index].item(),
        "atol": atol,
        "rtol": rtol,
    }


def compare_tensors(actual, expected, atol=None, rtol=None):
    if actual.shape != expected.shape:
        raise ValueError(
            f"Shape mismatch: actual={actual.shape}, expected={expected.shape}"
        )

    if atol is None or rtol is None:
        default_atol, default_rtol = get_tolerances(expected.dtype)

        atol = default_atol if atol is None else atol
        rtol = default_rtol if rtol is None else rtol

    diff = torch.abs(actual.float() - expected.float())

    denominator = torch.clamp(torch.abs(expected.float()), min=1e-12)
    rel_error = diff / denominator

    worst_index = diff.argmax().item()

    passed = torch.allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    )

    return {
        "passed": passed,
        "max_abs_error": diff.max().item(),
        "max_rel_error": rel_error.max().item(),
        "worst_index": worst_index,
        "expected": expected.flatten()[worst_index].item(),
        "actual": actual.flatten()[worst_index].item(),
        "atol": atol,
        "rtol": rtol,
    }
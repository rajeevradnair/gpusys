
import torch

from reference_utils import compare_tensors


expected = torch.tensor([1.0, 2.0, 3.0])

good = torch.tensor([1.0, 2.000001, 3.0])

bad = torch.tensor([1.0, 2.1, 3.0])

print("GOOD:", compare_tensors(good, expected))
print("BAD :", compare_tensors(bad, expected))

# precision of fp32 is better than fp16 is better than fp32
# fp32 and bf16 have a similar range ... same numbe of exponent bits
# so bf16 can stand in for fp32 for reduced precision


for dtype in [
    torch.float32,
    torch.float16,
    torch.bfloat16,
]:
    expected = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=dtype,
    )

    actual = expected.clone()
    actual[1] += 0.0005

    print(dtype)
    print(compare_tensors(actual, expected))

| Memory        | Scope           | Lifetime                | Size intuition | Speed intuition  | Programmer control         |
| ------------- | --------------- | ----------------------- | -------------- | ---------------- | -------------------------- |
| Registers     | One thread      | Thread/kernel execution | Tiny           | Fastest          | Mostly compiler-managed    |
| Shared memory | One block       | Block lifetime          | Small          | Very fast        | Explicit with `__shared__` |
| L1 cache      | Near an SM      | Hardware-managed        | Small          | Fast             | Mostly hardware-managed    |
| L2 cache      | GPU-wide/shared | Hardware-managed        | Larger         | Medium           | Mostly hardware-managed    |
| Global memory | Entire GPU      | Allocation lifetime     | Huge           | Slowest of these | Explicit loads/stores      |

from itertools import product
from pathlib import Path

output_dir = Path(__file__).parent

CU_TEMPLATE = """#include "launch.hpp"

template void launch_packed_attn<{head_dim}, {num_heads}, {num_heads_kv}, {page_size}, {paged_kv}, {qkvpacked}, {returnlse}, {qksquare}, {prefix}>(const Params& params, cudaStream_t stream);
"""

PAGE_DISABLED = None

HEAD_DIMS = [128]
HEAD_PAIRS = [(32, 32), (32, 8), (32, 4), (16, 8), (16, 4), (8, 2)]
PAGE_SIZES = [PAGE_DISABLED, 256, 128]
QKV_PACKED = [False, True]
PREFIX = [(False, False), (False, True), (True, False)]
RETURN_LSE = [False, True]

for (
    head_dim,
    (num_heads, num_heads_kv),
    page_size,
    qkvpacked,
    returnlse,
    (qksquare, prefix),
) in product(
    HEAD_DIMS,
    HEAD_PAIRS,
    PAGE_SIZES,
    QKV_PACKED,
    RETURN_LSE,
    PREFIX,
):
    file_name = (
        "packed_attn"
        f"_hdim{head_dim}"
        f"_heads{num_heads}_kvheads{num_heads_kv}"
        f"_pgsize{page_size if page_size else -1}"
        f"_qkvpacked{'true' if qkvpacked else 'false'}"
        f"_returnlse{'true' if returnlse else 'false'}"
        f"_qksquare{'true' if qksquare else 'false'}"
        f"_prefix{'true' if prefix else 'false'}"
        ".cu"
    )  # fmt: skip

    file_content = CU_TEMPLATE.format(
        head_dim=head_dim,
        num_heads=num_heads,
        num_heads_kv=num_heads_kv,
        page_size=page_size if page_size else -1,
        paged_kv="false" if page_size is None else "true",
        qkvpacked="true" if qkvpacked else "false",
        returnlse="true" if returnlse else "false",
        qksquare="true" if qksquare else "false",
        prefix="true" if prefix else "false",
    )
    (output_dir / file_name).write_text(file_content)

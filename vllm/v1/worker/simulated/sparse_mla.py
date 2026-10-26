# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structure-only sparse MLA for CPU simulation; no attention kernels execute."""

from vllm.v1.attention.backends.mla.cpu_mla import CPUMLABackend, CPUMLAImpl


class SimulatedSparseMLAImpl(CPUMLAImpl):
    is_sparse = True
    # The common metadata builder clones a prefill metadata backend. Keep that
    # construction path available; all execution entrypoints below still fail.
    supports_dense_mha_prefill = True

    def forward(self, *args, **kwargs):
        raise RuntimeError("Simulated sparse MLA cannot execute attention")

    def forward_mqa(self, *args, **kwargs):
        raise RuntimeError("Simulated sparse MLA cannot execute attention")

    def forward_mha(self, *args, **kwargs):
        raise RuntimeError("Simulated sparse MLA cannot execute attention")


class SimulatedSparseMLABackend(CPUMLABackend):
    @staticmethod
    def get_name() -> str:
        return "SIMULATED_SPARSE_MLA"

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type[SimulatedSparseMLAImpl]:
        return SimulatedSparseMLAImpl

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
        # DSA indexer and latent KV must share the indexer's 64-token blocks.
        return [64]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size == 64

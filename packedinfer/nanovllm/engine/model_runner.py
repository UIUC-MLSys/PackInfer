import os
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist

from ..config import Config
from ..engine.sequence import Sequence
from ..layers.sampler import Sampler
from ..models.qwen3 import Qwen3ForCausalLM
from ..models.mistral import MistralForCausalLM
from ..models.qwen3_moe import Qwen3MoeForCausalLM
from ..utils.context import get_context, reset_context, set_context
from ..utils.loader import load_model
from ...prepacking.processor import PrePackProcessor

_EVENTS = []
_RUN_STATS = []


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.use_prepacking = config.use_prepacking

        # Packed Attn
        self.packed_tok_threshold = config.packed_tok_threshold
        self.reserved_page_num = config.reserved_page_num

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        model_type = getattr(hf_config, "model_type", "").lower()
        if model_type == "mistral":
            self.model = MistralForCausalLM(hf_config)
        elif model_type == "qwen3":
            self.model = Qwen3ForCausalLM(hf_config)
        elif model_type == "qwen3_moe":
            self.model = Qwen3MoeForCausalLM(hf_config)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self._run_id = 0
        if self.use_prepacking:
            pad_token_id = hf_config.pad_token_id
            if pad_token_id is None:
                pad_token_id = hf_config.eos_token_id if hf_config.eos_token_id is not None else 0
            self.prepack_processor = PrePackProcessor(None, pad_token_id=pad_token_id)
        else:
            self.prepack_processor = None
        # self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        self.kv_cache[0].fill_(float("nan"))
        self.kv_cache[1].zero_()
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill_prepack(self, seqs: list[Sequence], run_id: int = -1, seq_lens: list[int] | None = None):
        if self.prepack_processor is None:
            raise RuntimeError("PrePackProcessor is not available for prepacking.")
        if seq_lens is None:
            seq_lens = [len(seq) for seq in seqs]
        token_ids = [seq.token_ids[seq.num_cached_tokens :] for seq in seqs]
        token_pos_offsets = [seq.num_cached_tokens for seq in seqs]
        (
            new_tokens,
            new_positions,
            new_mask,
            _restart_indices,
            _original_ids,
            token_source_seqs,
            token_source_pos,
        ) = self.prepack_processor.batch_process_token_ids(
            token_ids,
            token_pos_offsets=token_pos_offsets,
            drop_last_token=False,
        )
        if new_tokens is None:
            raise RuntimeError("Prepacking produced empty inputs.")
        token_source_seqs_cpu = token_source_seqs.cpu().tolist()
        token_source_pos_cpu = token_source_pos.cpu().tolist()
        last_bins = [-1] * len(seqs)
        last_pos = [-1] * len(seqs)
        last_token_pos = [len(seq) - 1 for seq in seqs]
        slot_mapping = []
        for b, (seq_row, pos_row) in enumerate(zip(token_source_seqs_cpu, token_source_pos_cpu)):
            for t, (seq_idx, token_pos) in enumerate(zip(seq_row, pos_row)):
                if seq_idx == -1:
                    slot_mapping.append(-1)
                    continue
                if token_pos == last_token_pos[seq_idx]:
                    last_bins[seq_idx] = b
                    last_pos[seq_idx] = t
                seq = seqs[seq_idx]
                block_id = seq.block_table[token_pos // self.block_size]
                slot_mapping.append(block_id * self.block_size + (token_pos % self.block_size))
        if any(idx == -1 for idx in last_bins) or any(idx == -1 for idx in last_pos):
            raise RuntimeError("Failed to locate last tokens for prepacked sequences.")
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        last_bins = torch.tensor(last_bins, dtype=torch.int64, device=new_tokens.device)
        last_pos = torch.tensor(last_pos, dtype=torch.int64, device=new_tokens.device)
        set_context(
            True,
            seq_lens=seq_lens,
            run_id=run_id,
            slot_mapping=slot_mapping,
        )
        return new_tokens, new_positions, new_mask, (last_bins, last_pos)

    def shared_prefix_block(self, seqs: list[Sequence]):
        min_len = min(len(seq.block_table) for seq in seqs)
        cnt = 0

        for i in range(min_len):
            b0 = seqs[0].block_table[i]
            if all(seq.block_table[i] == b0 for seq in seqs):
                cnt += 1
            else:
                break
        return cnt

    def prepare_prefill(self, seqs: list[Sequence], run_id: int = -1, seq_lens: list[int] | None = None):
        if self.use_prepacking and all(seq.num_cached_tokens == 0 for seq in seqs):
            return self.prepare_prefill_prepack(seqs, run_id=run_id, seq_lens=seq_lens)
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        # Packed Attn
        if seq_lens is None:
            seq_lens = [len(seq) for seq in seqs]

        use_bfd = os.getenv("ENABLE_BFD", "0") == "1"

        class Bin:
            def __init__(self):
                self.seqs = []
                self.used = 0  # token count

        packable_seqs = []
        non_pack_seqs = []
        bin_capacity = self.packed_tok_threshold
        bins: list[Bin] = []

        for seq in seqs:
            if len(seq) <= self.packed_tok_threshold:
                packable_seqs.append(seq)
            else:
                non_pack_seqs.append(seq)

        if use_bfd:
            # Best-Fit Decreasing
            packable_seqs.sort(key=lambda s: len(s), reverse=True)
            for seq in packable_seqs:
                size = len(seq)

                best_bin = None
                min_remaining = float("inf")

                for b in bins:
                    if b.used + size <= bin_capacity:
                        remaining = bin_capacity - (b.used + size)
                        if remaining < min_remaining:
                            min_remaining = remaining
                            best_bin = b

                if best_bin is None:
                    new_bin = Bin()
                    new_bin.seqs.append(seq)
                    new_bin.used = size
                    bins.append(new_bin)
                else:
                    best_bin.seqs.append(seq)
                    best_bin.used += size

        ordered_seqs = non_pack_seqs.copy()
        if use_bfd:
            for b in bins:
                for seq in b.seqs:
                    ordered_seqs.append(seq)
        else:
            ordered_seqs.extend(packable_seqs)

        for seq in ordered_seqs:
            # for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        # Packed Attn
        leftpad = []
        rightlim = []
        packed_cu_seqlens_q = [0]
        packed_cu_seqlens_k = [0]
        packed_block_tables = [[]]
        packed_max_seqlen = 0
        prefix_pad = []

        # Deprecated
        # leftpad_tok_ofs = 0
        # leftpad_pg_ofs = 0
        packed_src_slots = []
        packed_dst_slots = []

        to_be_packed_len = 0
        to_be_packed: list[Sequence] = []

        for seq in ordered_seqs:
            seqlen = len(seq)
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen

            # if seqlen > self.packed_tok_threshold:  # Do not pack
            if seq in non_pack_seqs:
                leftpad.extend([0] * seqlen_q)
                for i in range(seqlen_q):
                    rightlim.append(seq.num_cached_tokens + i + 1)
                prefix_pad.append(0)
                packed_cu_seqlens_q.append(packed_cu_seqlens_q[-1] + seqlen_q)
                packed_cu_seqlens_k.append(packed_cu_seqlens_k[-1] + seqlen_k)
                packed_block_tables[-1] = seq.block_table
                packed_max_seqlen = max(packed_max_seqlen, seqlen_q)
                packed_block_tables.append([])
            else:  # Pack
                to_be_packed_len += seqlen
                to_be_packed.append(seq)

                # leftpad.extend([leftpad_tok_ofs] * seqlen)
                # for i in range(seqlen):
                #     rightlim.append(leftpad_tok_ofs + i + 1)
                # leftpad_tok_ofs += seqlen

                # start_dst = leftpad_pg_ofs * self.block_size + leftpad_tok_ofs
                # for i in range(seq.num_blocks):
                #     start = seq.block_table[i] * self.block_size
                #     if i != seq.num_blocks - 1:
                #         end = self.block_size
                #     else:
                #         end = seq.last_block_num_tokens
                #     packed_src_slots.extend(range(start, end + start))
                # packed_dst_slots.extend(range(start_dst, start_dst + seqlen))

                if to_be_packed_len > self.packed_tok_threshold or seq == ordered_seqs[-1]:
                    prefix_block_num = 0 if len(to_be_packed) == 1 else self.shared_prefix_block(to_be_packed)
                    prefix_token_num = prefix_block_num * self.block_size
                    prefix_pad.append(prefix_token_num)
                    q_token_num = 0
                    leftpad_tok_ofs = prefix_token_num
                    for seq in to_be_packed:
                        seqlen = len(seq)
                        seqlen_q = seqlen - seq.num_cached_tokens
                        seqlen_k = seqlen

                        q_token_num += seqlen_q
                        leftpad.extend([leftpad_tok_ofs] * seqlen_q)
                        for i in range(seqlen_q):
                            rightlim.append(leftpad_tok_ofs + i + 1)

                        packed_block_tables[-1].extend(seq.block_table[prefix_block_num:])

                        leftpad_tok_ofs += seqlen_k - prefix_token_num

                    packed_max_seqlen = max(packed_max_seqlen, q_token_num)
                    packed_cu_seqlens_q.append(packed_cu_seqlens_q[-1] + q_token_num)
                    packed_cu_seqlens_k.append(packed_cu_seqlens_k[-1] + leftpad_tok_ofs)
                    # next_page = leftpad_pg_ofs + (leftpad_tok_ofs + self.block_size - 1) // self.block_size
                    # if next_page >= self.reserved_page_num:
                    #     raise RuntimeError("Not enough kv-cache space for Packed Attention. Please increase reserved_page_num.")
                    # packed_block_tables[-1].extend(range(leftpad_pg_ofs, next_page))
                    packed_block_tables.append([])
                    to_be_packed = []
                    to_be_packed_len = 0
                    # leftpad_pg_ofs = next_page

        packed_block_tables.pop()

        pack_group_lens = []
        pack_group_ratios = []
        pack_group_max = 0
        if len(packed_cu_seqlens_k) > 1:
            pack_group_lens = [packed_cu_seqlens_k[i + 1] - packed_cu_seqlens_k[i] for i in range(len(packed_cu_seqlens_k) - 1)]
            pack_group_max = max(pack_group_lens)
            if pack_group_max > 0:
                pack_group_ratios = [group_len / pack_group_max for group_len in pack_group_lens]

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        leftpad = torch.tensor(leftpad, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        rightlim = torch.tensor(rightlim, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        packed_cu_seqlens_q = torch.tensor(packed_cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        packed_cu_seqlens_k = torch.tensor(packed_cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        max_len = max(len(packed_block_table) for packed_block_table in packed_block_tables)
        packed_block_tables = [block_table + [-1] * (max_len - len(block_table)) for block_table in packed_block_tables]
        packed_block_tables = torch.tensor(packed_block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        packed_src_slots = torch.tensor(packed_src_slots, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        packed_dst_slots = torch.tensor(packed_dst_slots, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        prefix_pad = torch.tensor(prefix_pad, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        if block_tables is None:
            rightlim = None
            prefix_pad = None
            packed_block_tables = None

        set_context(
            True,
            seq_lens=seq_lens,
            run_id=run_id,
            pack_group_lens=pack_group_lens,
            pack_group_ratios=pack_group_ratios,
            pack_group_max=pack_group_max,
            rightlim=rightlim,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=block_tables,
            leftpad=leftpad,
            packed_cu_seqlens_q=packed_cu_seqlens_q,
            packed_cu_seqlens_k=packed_cu_seqlens_k,
            packed_max_seqlen=packed_max_seqlen,
            packed_block_tables=packed_block_tables,
            packed_src_slots=packed_src_slots,
            packed_dst_slots=packed_dst_slots,
            prefix_pad=prefix_pad,
        )

        return input_ids, positions, None, None

    # def prepare_decode(self, seqs: list[Sequence]):
    def prepare_decode(self, seqs: list[Sequence], run_id: int = -1, seq_lens: list[int] | None = None):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        # first sort the seqs: the seqs need to be longest and shortest interleaved
        seqs.sort(key=lambda x: len(x))
        # print(f"Shortest seq length: {len(seqs[0])}, Longest seq length: {len(seqs[-1])}")
        if seq_lens is None:
            seq_lens = [len(seq) for seq in seqs]
        sorted_seqs = []
        i, j = 0, len(seqs) - 1
        while i <= j:
            if i == j:
                sorted_seqs.append(seqs[i])
            else:
                sorted_seqs.append(seqs[j])
                sorted_seqs.append(seqs[i])
            i += 1
            j -= 1
        seqs = sorted_seqs

        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 1
        max_seqlen_k = 0

        for seq in seqs:
            seqlen_q = 1
            seqlen_k = len(seq)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        # Packed Attn
        leftpad = []
        rightlim = []
        packed_cu_seqlens_q = [0]
        packed_cu_seqlens_k = [0]
        packed_block_tables = [[]]
        packed_max_seqlen = 0
        prefix_pad = []

        # Deprecated
        # leftpad_tok_ofs = 0
        # leftpad_pg_ofs = 0
        packed_src_slots = []
        packed_dst_slots = []

        to_be_packed_len = 0
        to_be_packed: list[Sequence] = []

        for seq in seqs:
            seqlen = len(seq)
            seqlen_q = 1
            seqlen_k = seqlen

            if seqlen > self.packed_tok_threshold:  # Do not pack
                leftpad.extend([0] * seqlen_q)
                for i in range(seqlen_q):
                    rightlim.append(seq.num_cached_tokens + i + 1)
                prefix_pad.append(0)
                packed_cu_seqlens_q.append(packed_cu_seqlens_q[-1] + seqlen_q)
                packed_cu_seqlens_k.append(packed_cu_seqlens_k[-1] + seqlen_k)
                packed_block_tables[-1] = seq.block_table
                packed_max_seqlen = max(packed_max_seqlen, seqlen_q)
                packed_block_tables.append([])
            else:  # Pack
                to_be_packed_len += seqlen
                to_be_packed.append(seq)

                if to_be_packed_len > self.packed_tok_threshold or seq == seqs[-1]:
                    prefix_block_num = 0 if len(to_be_packed) == 1 else self.shared_prefix_block(to_be_packed)
                    prefix_token_num = prefix_block_num * self.block_size
                    prefix_pad.append(prefix_token_num)
                    q_token_num = 0
                    leftpad_tok_ofs = prefix_token_num
                    for seq in to_be_packed:
                        seqlen = len(seq)
                        seqlen_q = 1
                        seqlen_k = seqlen

                        q_token_num += seqlen_q
                        leftpad.extend([leftpad_tok_ofs] * seqlen_q)
                        for i in range(seqlen_q):
                            rightlim.append(leftpad_tok_ofs + i + 1)

                        packed_block_tables[-1].extend(seq.block_table[prefix_block_num:])

                        leftpad_tok_ofs += seqlen_k - prefix_token_num

                    packed_max_seqlen = max(packed_max_seqlen, q_token_num)
                    packed_cu_seqlens_q.append(packed_cu_seqlens_q[-1] + q_token_num)
                    packed_cu_seqlens_k.append(packed_cu_seqlens_k[-1] + leftpad_tok_ofs)
                    # next_page = leftpad_pg_ofs + (leftpad_tok_ofs + self.block_size - 1) // self.block_size
                    # if next_page >= self.reserved_page_num:
                    #     raise RuntimeError("Not enough kv-cache space for Packed Attention. Please increase reserved_page_num.")
                    # packed_block_tables[-1].extend(range(leftpad_pg_ofs, next_page))
                    packed_block_tables.append([])
                    to_be_packed = []
                    to_be_packed_len = 0
                    # leftpad_pg_ofs = next_page

        packed_block_tables.pop()

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)

        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        leftpad = torch.tensor(leftpad, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        rightlim = torch.tensor(rightlim, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        packed_cu_seqlens_q = torch.tensor(packed_cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        packed_cu_seqlens_k = torch.tensor(packed_cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        max_len = max(len(packed_block_table) for packed_block_table in packed_block_tables)
        packed_block_tables = [block_table + [-1] * (max_len - len(block_table)) for block_table in packed_block_tables]
        packed_block_tables = torch.tensor(packed_block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        packed_src_slots = torch.tensor(packed_src_slots, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        packed_dst_slots = torch.tensor(packed_dst_slots, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        prefix_pad = torch.tensor(prefix_pad, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(
            False,
            seq_lens=seq_lens,
            run_id=run_id,
            pack_group_lens=None,
            pack_group_ratios=None,
            pack_group_max=0,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            leftpad=leftpad,
            packed_cu_seqlens_q=packed_cu_seqlens_q,
            packed_cu_seqlens_k=packed_cu_seqlens_k,
            packed_max_seqlen=packed_max_seqlen,
            packed_block_tables=packed_block_tables,
            packed_src_slots=packed_src_slots,
            packed_dst_slots=packed_dst_slots,
            prefix_pad=prefix_pad,
            rightlim=rightlim,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        attention_mask: torch.Tensor | None = None,
        prepack_last_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            if attention_mask is None:
                return self.model.compute_logits(self.model(input_ids, positions))
            hidden_states = self.model(input_ids, positions, attention_mask=attention_mask)
            if prepack_last_indices is None:
                raise RuntimeError("prepack_last_indices must be provided when using prepacking.")
            last_bins, last_pos = prepack_last_indices
            last_hidden = hidden_states[last_bins, last_pos]
            return self.model.compute_logits(last_hidden, use_context=False)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        self._run_id += 1
        seq_lens = [len(seq) for seq in seqs]
        if is_prefill:
            input_ids, positions, attention_mask, prepack_last_indices = self.prepare_prefill(
                seqs, run_id=self._run_id, seq_lens=seq_lens
            )
        else:
            input_ids, positions = self.prepare_decode(seqs, run_id=self._run_id, seq_lens=seq_lens)
            attention_mask = None
            prepack_last_indices = None
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        if os.getenv("PROFILE_TIMING", "0") == "1":
            context = get_context()
            _RUN_STATS.append(
                {
                    "run_id": self._run_id,
                    "is_prefill": is_prefill,
                    "batch_lens": seq_lens,
                    "batch_size": len(seqs),
                    "attn_backend": os.getenv("ATTN_BACKEND", "PACKED_ATTN"),
                    "pack_groups": len(context.pack_group_lens or []),
                    "pack_group_lens": context.pack_group_lens,
                    "pack_group_ratios": context.pack_group_ratios,
                    "pack_group_max": context.pack_group_max,
                }
            )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        logits = self.run_model(
            input_ids,
            positions,
            is_prefill,
            attention_mask=attention_mask,
            prepack_last_indices=prepack_last_indices,
        )
        end_event.record()
        _EVENTS.append((start_event, end_event))
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

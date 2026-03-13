from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch
from .utils import greedy_packing


class PrePackProcessor:
    def __init__(self, tokenizer, packing_fn=None, pad_token_id=None):
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        elif pad_token_id is not None:
            self.pad_token = pad_token_id
        else:
            raise ValueError("tokenizer or pad_token_id must be provided.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if packing_fn:
            self.packing_fn = packing_fn
        else:
            self.packing_fn = greedy_packing

    def process(
        self,
        length_dict,
        packing_dict,
        token_dict,
        drop_last_token: bool = True,
        token_pos_offsets: dict | None = None,
        track_token_sources: bool = False,
    ):
        '''
        Takes batch of tokens and packs them according to the bin-packing algorithm.
    
        Args:
            length_dict (dict): maps original batch index to its prompt length
            packing_dict (dict): a mapping between prompt length and bin index
            token_dict (dict): maps original batch index to its tokenized prompt
    
        Returns:
            new_tokens (Tensor): packed sequence of tokens
            new_positions (Tensor): restart positions for the new_tokens
            new_mask (Tensor): independent mask for the new_tokens
            restart_dict (dict): mapping restart index and original batch index
        '''
        new_positions = []
        new_tokens = []
        restart_dict = {0: -1}  # -1 is a placeholder
        restart_index = 0
        new_token_sources = []
        token_pos_offsets = token_pos_offsets or {}

        for key in packing_dict:
            seq_tokens = token_dict[key]
            keep_len = len(seq_tokens) - 1 if drop_last_token else len(seq_tokens)
            new_tokens += seq_tokens[:keep_len]  # optionally omit final token for generation
            restart_index += keep_len
            start_pos = token_pos_offsets.get(key, 0)
            new_positions += list(range(start_pos, start_pos + keep_len))
            if track_token_sources:
                new_token_sources += [(key, start_pos + i) for i in range(keep_len)]
            restart_dict[restart_index] = key

        restart_indices = list(restart_dict.keys())
        size = len(new_tokens)
        new_mask = torch.zeros(size, size, device=self.device)

        for i in range(len(restart_indices) - 1):
            start = restart_indices[i]
            end = restart_indices[i + 1]
            new_mask[start:end, start:end] = torch.tril(torch.ones((end - start, end - start)))

        new_tokens = torch.tensor(new_tokens, device=self.device)
        new_positions = torch.tensor(new_positions, device=self.device)
        new_mask = new_mask.clone().detach()

        if track_token_sources:
            return new_tokens, new_positions, new_mask, restart_dict, new_token_sources
        return new_tokens, new_positions, new_mask, restart_dict

    def batch_process(self, sentences):

        original_ids = self.tokenizer(sentences).input_ids
        token_dict = dict(enumerate(original_ids))
        length_dict = {index: len(toks) for index, toks in enumerate(original_ids)}

        max_bin_size = max(length_dict.values())
        packing_lst = self.packing_fn(length_dict, max_bin_size)

        batch_new_tokens = []
        batch_new_positions = []
        batch_new_mask = []
        batch_restart_indices = []
        for packing_dict in packing_lst:
            new_tokens, new_positions, new_mask, restart_indices = self.process(
                length_dict, packing_dict, token_dict
            )
            batch_new_tokens.append(new_tokens)
            batch_new_positions.append(new_positions)
            batch_new_mask.append(new_mask)
            batch_restart_indices.append(restart_indices)

        batch_new_tokens = pad_sequence(batch_new_tokens, batch_first=True, padding_value=self.pad_token)
        batch_new_positions = pad_sequence(batch_new_positions, batch_first=True, padding_value=1)

        max_size = max(tensor.shape[1:] for tensor in batch_new_mask)[0]
        padded_masks = [
            F.pad(tensor, (0, max_size - tensor.size(0), 0, max_size - tensor.size(1)))
            for tensor in batch_new_mask
        ]
        batch_new_mask = torch.stack(padded_masks)
        # Convert packed 2D masks into a 4D additive mask expected by SDPA/eager attention.
        min_dtype = torch.finfo(batch_new_mask.dtype).min
        batch_new_mask = (1.0 - batch_new_mask) * min_dtype
        batch_new_mask = batch_new_mask.unsqueeze(1)

        return batch_new_tokens, batch_new_positions, batch_new_mask, batch_restart_indices, original_ids

    def batch_process_token_ids(
        self,
        token_ids: list[list[int]],
        token_pos_offsets: list[int] | None = None,
        drop_last_token: bool = False,
    ):
        token_dict = dict(enumerate(token_ids))
        length_dict = {index: len(toks) for index, toks in enumerate(token_ids)}
        token_pos_offsets = token_pos_offsets or [0] * len(token_ids)
        token_pos_offsets = {index: offset for index, offset in enumerate(token_pos_offsets)}

        max_bin_size = max(length_dict.values()) if length_dict else 0
        packing_lst = self.packing_fn(length_dict, max_bin_size) if length_dict else []

        batch_new_tokens = []
        batch_new_positions = []
        batch_new_mask = []
        batch_restart_indices = []
        batch_token_sources = []

        for packing_dict in packing_lst:
            new_tokens, new_positions, new_mask, restart_indices, token_sources = self.process(
                length_dict,
                packing_dict,
                token_dict,
                drop_last_token=drop_last_token,
                token_pos_offsets=token_pos_offsets,
                track_token_sources=True,
            )
            batch_new_tokens.append(new_tokens)
            batch_new_positions.append(new_positions)
            batch_new_mask.append(new_mask)
            batch_restart_indices.append(restart_indices)
            batch_token_sources.append(token_sources)

        if not batch_new_tokens:
            return None, None, None, None, None, None, None

        batch_new_tokens = [torch.tensor(tokens, device=self.device) for tokens in batch_new_tokens]
        batch_new_positions = [torch.tensor(pos, device=self.device) for pos in batch_new_positions]
        batch_new_tokens = pad_sequence(batch_new_tokens, batch_first=True, padding_value=self.pad_token)
        batch_new_positions = pad_sequence(batch_new_positions, batch_first=True, padding_value=0)

        max_size = max(tensor.shape[1:] for tensor in batch_new_mask)[0]
        padded_masks = [
            F.pad(tensor, (0, max_size - tensor.size(0), 0, max_size - tensor.size(1)))
            for tensor in batch_new_mask
        ]
        batch_new_mask = torch.stack(padded_masks)
        min_dtype = torch.finfo(batch_new_mask.dtype).min
        batch_new_mask = (1.0 - batch_new_mask) * min_dtype
        batch_new_mask = batch_new_mask.unsqueeze(1)

        token_source_seqs = torch.full(
            (len(batch_token_sources), max_size),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        token_source_pos = torch.full(
            (len(batch_token_sources), max_size),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        for i, token_sources in enumerate(batch_token_sources):
            for j, (seq_idx, token_pos) in enumerate(token_sources):
                token_source_seqs[i, j] = seq_idx
                token_source_pos[i, j] = token_pos

        return (
            batch_new_tokens,
            batch_new_positions,
            batch_new_mask,
            batch_restart_indices,
            token_ids,
            token_source_seqs,
            token_source_pos,
        )

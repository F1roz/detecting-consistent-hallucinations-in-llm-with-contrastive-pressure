"""Hugging Face transformers backend. Works on any CUDA GPU; uses 4-bit NF4 when memory is short."""

from __future__ import annotations

from itertools import groupby

from ..io_utils import get_logger
from .base import GenRequest, GenResult, group_key

log = get_logger(__name__)


class HFBackend:
    name = "transformers"

    def __init__(self, hf_id: str, revision: str | None, low_memory: bool, batch_size: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = {"revision": revision, "device_map": "auto"}
        if low_memory:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
            self.precision = "nf4-4bit"
        else:
            kwargs["dtype"] = dtype
            self.precision = "bf16" if dtype == torch.bfloat16 else "fp16"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
        except TypeError:  # older transformers name the argument torch_dtype
            kwargs["torch_dtype"] = kwargs.pop("dtype", None)
            self.model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
        self.model.eval()
        self.eos_ids = self._eos_ids()
        log.info("Loaded %s with transformers (%s)", hf_id, self.precision)

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        results: list[GenResult | None] = [None] * len(requests)
        order = sorted(range(len(requests)), key=lambda i: group_key(requests[i]))
        for _, members in groupby(order, key=lambda i: group_key(requests[i])):
            indices = list(members)
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start:start + self.batch_size]
                for index, result in zip(chunk, self._batch([requests[i] for i in chunk])):
                    results[index] = result
        return results  # type: ignore[return-value]

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()

    def _batch(self, requests: list[GenRequest]) -> list[GenResult]:
        torch = self.torch
        first = requests[0]
        prompts = [self.tokenizer.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True)
                   for r in requests]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        inputs = inputs.to(self.model.device)
        sampling = first.temperature > 0
        if first.seed is not None:
            torch.manual_seed(first.seed)
        gen_kwargs = dict(
            max_new_tokens=first.max_new_tokens,
            do_sample=sampling,
            num_return_sequences=first.n,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_ids,
            repetition_penalty=1.0,
        )
        if sampling:
            gen_kwargs.update(temperature=first.temperature, top_p=1.0, top_k=0)
        else:
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        generated = out.sequences[:, prompt_len:]
        transition = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        first_step = torch.log_softmax(out.scores[0].float(), dim=-1)

        results = []
        for row, request in enumerate(requests):
            texts, logprobs = [], []
            for k in range(request.n):
                seq_index = row * request.n + k
                ids, lps = self._trim(generated[seq_index].tolist(), transition[seq_index].tolist())
                texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
                logprobs.append(lps)
            top = {}
            if request.first_token_top_k:
                values, token_ids = first_step[row * request.n].topk(request.first_token_top_k)
                for value, token_id in zip(values.tolist(), token_ids.tolist()):
                    token = self.tokenizer.decode([token_id]).strip()
                    top.setdefault(token, value)
            results.append(GenResult(texts=texts, token_logprobs=logprobs if request.want_logprobs else None,
                                     first_token_logprobs=top))
        return results

    def _trim(self, ids: list[int], lps: list[float]) -> tuple[list[int], list[float]]:
        kept_ids, kept_lps = [], []
        for token_id, lp in zip(ids, lps):
            if token_id in self.eos_ids or token_id == self.tokenizer.pad_token_id:
                break
            kept_ids.append(token_id)
            kept_lps.append(float(lp))
        return kept_ids, kept_lps

    def _eos_ids(self) -> list[int]:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        return list(eos) if isinstance(eos, (list, tuple)) else [eos]

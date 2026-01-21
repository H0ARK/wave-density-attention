from __future__ import annotations

from typing import Any

import torch


class TeacherProvider:
    def __init__(
        self,
        *,
        student: torch.nn.Module,
        teacher: torch.nn.Module | None = None,
        ema_model_params: dict[str, torch.Tensor] | None = None,
        ema_lm_params: dict[str, torch.Tensor] | None = None,
        functional_call: Any | None = None,
    ):
        self.student = student
        self.teacher = teacher
        self.ema_model_params = ema_model_params
        self.ema_lm_params = ema_lm_params
        self.functional_call = functional_call
        self.use_ema = False

    def set_use_ema(self, value: bool) -> None:
        self.use_ema = bool(value)

    def set_teacher(self, teacher: torch.nn.Module | None) -> None:
        self.teacher = teacher

    def has_teacher(self) -> bool:
        if self.use_ema:
            return self.ema_model_params is not None
        return self.teacher is not None

    def forward_backbone(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...] | None]:
        if self.use_ema:
            if self.functional_call is None or self.ema_model_params is None:
                raise RuntimeError(
                    "EMA teacher requested but functional_call/ema_model_params not available."
                )
            t_out = self.functional_call(
                self.student.model,
                self.ema_model_params,
                (),
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "output_hidden_states": output_hidden_states,
                },
            )
        else:
            if self.teacher is None:
                return None, None
            t_out = self.teacher.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )

        t_hidden = t_out[0]
        t_all_h = t_out.hidden_states if output_hidden_states else None
        return t_hidden, t_all_h

    def logits_from_hidden(self, t_hidden: torch.Tensor) -> torch.Tensor | None:
        if self.use_ema:
            if self.functional_call is None or self.ema_lm_params is None:
                raise RuntimeError(
                    "EMA teacher requested but functional_call/ema_lm_params not available."
                )
            return self.functional_call(
                self.student.lm_head, self.ema_lm_params, (t_hidden,), {}
            ).float()
        if self.teacher is None:
            return None
        return self.teacher.lm_head(t_hidden).float()

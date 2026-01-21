from __future__ import annotations

from typing import Any

from wave_dencity.transplant.parallel_attn import ParallelAttentionWrapper


class FeatureExtractor:
    def __init__(self, targets: list[str] | None = None):
        self.targets = set(targets or [])

    def wants_wrapper_features(self) -> bool:
        return bool(
            self.targets.intersection(
                {"attn_out", "resid_delta", "mixed_out", "wda_out"}
            )
        )

    def set_wrapper_collection(self, enabled: bool) -> None:
        ParallelAttentionWrapper.collect_features = bool(enabled)

    def extract(
        self,
        *,
        wrappers: list[ParallelAttentionWrapper],
        student_hidden_states: tuple[Any, ...] | None = None,
        teacher_hidden_states: tuple[Any, ...] | None = None,
    ) -> dict[str, dict[str, Any]]:
        student: dict[str, Any] = {}
        teacher: dict[str, Any] = {}

        if "hidden_states" in self.targets:
            if student_hidden_states is not None:
                student["hidden_states"] = student_hidden_states
            if teacher_hidden_states is not None:
                teacher["hidden_states"] = teacher_hidden_states

        if "resid_energy" in self.targets:
            if student_hidden_states is not None and len(student_hidden_states) > 1:
                for i in range(1, len(student_hidden_states)):
                    delta = student_hidden_states[i] - student_hidden_states[i - 1]
                    energy = delta.float().pow(2).mean(dim=-1).sqrt()
                    student[f"layer_{i - 1}.resid_energy"] = energy
            if teacher_hidden_states is not None and len(teacher_hidden_states) > 1:
                for i in range(1, len(teacher_hidden_states)):
                    delta = teacher_hidden_states[i] - teacher_hidden_states[i - 1]
                    energy = delta.float().pow(2).mean(dim=-1).sqrt()
                    teacher[f"layer_{i - 1}.resid_energy"] = energy

        if self.wants_wrapper_features():
            for m in wrappers:
                if m.layer_idx is None:
                    continue
                idx = int(m.layer_idx)
                if "attn_out" in self.targets and m.last_attn_out is not None:
                    teacher[f"layer_{idx}.attn_out"] = m.last_attn_out
                if "wda_out" in self.targets and m.last_wda_out is not None:
                    student[f"layer_{idx}.wda_out"] = m.last_wda_out
                if "mixed_out" in self.targets and m.last_mixed_out is not None:
                    student[f"layer_{idx}.mixed_out"] = m.last_mixed_out
                if "resid_delta" in self.targets and m.last_mixed_out is not None:
                    student[f"layer_{idx}.resid_delta"] = m.last_mixed_out
                    if m.last_attn_out is not None:
                        teacher[f"layer_{idx}.resid_delta"] = m.last_attn_out

        return {"student": student, "teacher": teacher}

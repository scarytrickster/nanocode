from dataclasses import dataclass


@dataclass
class RLMBudget:
    """Hard limits for recursive execution."""

    max_depth: int = 3
    max_children: int = 4
    max_iterations: int = 10

    children_created: int = 0
    iterations: int = 0

    def can_spawn_child(self, depth: int) -> bool:
        return (
            depth <= self.max_depth
            and self.children_created < self.max_children
        )

    def consume_child(self) -> None:
        if self.children_created >= self.max_children:
            raise RuntimeError("RLM child budget exceeded")

        self.children_created += 1

    def consume_iteration(self) -> None:
        if self.iterations >= self.max_iterations:
            raise RuntimeError("RLM iteration budget exceeded")

        self.iterations += 1
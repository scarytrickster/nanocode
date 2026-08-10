from dataclasses import dataclass


@dataclass
class Experience:
    """A lesson learned from a previous agent execution."""

    task: str
    diagnosis: str
    improvement: str
    success: bool


class Memory:
    """Simple in-memory experience store."""

    def __init__(self) -> None:
        self.experiences: list[Experience] = []

    def add(self, experience: Experience) -> None:
        """Store an experience."""

        self.experiences.append(experience)

    def get_all(self) -> list[Experience]:
        """Return all stored experiences."""

        return self.experiences.copy()

    def retrieve(
        self,
        task: str,
        limit: int = 3,
    ) -> list[Experience]:
        """Retrieve experiences relevant to a task."""

        task_words = set(task.lower().split())

        matches = []

        for experience in self.experiences:
            experience_words = set(
                experience.task.lower().split()
            )

            overlap = len(task_words & experience_words)

            if overlap > 0:
                matches.append(
                    (overlap, experience)
                )

        matches.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            experience
            for _, experience in matches[:limit]
        ]

    def clear(self) -> None:
        """Clear all stored experiences."""

        self.experiences.clear()

    def count(self) -> int:
        """Return the number of stored experiences."""

        return len(self.experiences)
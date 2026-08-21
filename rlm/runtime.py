from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

from rlm.budget import RLMBudget
from rlm.call import RLMCallHandler
from rlm.context import RLMContext
from rlm.result import RLMResult
from rlm.synthesizer import RLMSynthesizer
from langfuse import get_client

# Children run concurrently, but never more than this many at once. Three
# matches what the deterministic decomposer produces, and the provider has
# already shown it is sensitive to bursts: more parallelism would mostly buy
# more 429s.
DEFAULT_MAX_CONCURRENCY = 3


class RLMRuntime:
    """Runtime responsible for controlled recursive RLM execution."""

    def __init__(
        self,
        call_handler: RLMCallHandler,
        budget: RLMBudget | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError(
                f"max_concurrency must be positive, got {max_concurrency}"
            )

        self.call_handler = call_handler
        self.budget = budget or RLMBudget()
        self.langfuse = get_client()

        self.max_concurrency = max_concurrency

        # Budget checks and consumption must be one atomic decision, or two
        # threads could both pass can_spawn_child on the last slot.
        self._budget_lock = threading.Lock()

        # Observed concurrency, so a caller can see what actually overlapped
        # rather than what was permitted.
        self._active_lock = threading.Lock()
        self._active_children = 0
        self.peak_active_children = 0

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def _reserve_child(
        self,
        parent: RLMContext,
        task: str,
        content: str = "",
    ) -> RLMContext:
        """Claim a budget slot for one child, or refuse it.

        Reservation happens on the submitting thread, before any worker
        starts, so the budget decides how many children exist rather than
        discovering the violation inside a worker that already ran.
        """

        child = parent.child(
            task=task,
            content=content,
        )

        with self._budget_lock:
            if not self.budget.can_spawn_child(child.depth):
                raise RuntimeError(
                    "RLM recursion budget exceeded"
                )

            self.budget.consume_child()
            self.budget.consume_iteration()

        return child

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _enter(self) -> None:
        with self._active_lock:
            self._active_children += 1
            self.peak_active_children = max(
                self.peak_active_children, self._active_children
            )

    def _leave(self) -> None:
        with self._active_lock:
            self._active_children -= 1

    def _execute_child(self, child: RLMContext) -> RLMResult:
        """Run one already-reserved child."""

        self._enter()

        try:
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="rlm-child",
                input={
                    "task": child.task,
                    "content": child.content,
                    "depth": child.depth,
                },
            ) as span:

                result = self.call_handler.call(child)

                span.update(
                    output={
                        "success": result.success,
                        "answer": result.answer,
                        "depth": result.depth,
                    }
                )

                return result
        finally:
            self._leave()

    def call(
        self,
        parent: RLMContext,
        task: str,
        content: str = "",
    ) -> RLMResult:
        """Execute one controlled child RLM call."""

        child = self._reserve_child(parent, task, content)

        return self._execute_child(child)

    def call_many(
        self,
        parent: RLMContext,
        tasks: list[tuple[str, str]],
    ) -> list[RLMResult]:
        """
        Execute multiple child RLM calls, up to max_concurrency at a time.

        Each tuple contains:
            (task, content)

        Results come back in decomposition order regardless of the order the
        children finished in, so scheduling can never change what synthesis
        sees.
        """

        if not tasks:
            return []

        # Child identity is fixed here, from the decomposition order, before
        # anything runs: child 2 stays child 2 even if it finishes first.
        self._announce_children(tasks)

        reserved: list[tuple[int, RLMContext]] = []
        budget_error: RuntimeError | None = None

        for index, (task, content) in enumerate(tasks):
            try:
                reserved.append(
                    (index, self._reserve_child(parent, task, content))
                )
            except RuntimeError as error:
                # The budget refused this child. Children already reserved
                # still run to completion; the refusal is raised afterwards.
                budget_error = error
                break

        results: list[RLMResult | None] = [None] * len(reserved)

        if not reserved:
            # Every child was refused: nothing to run, no pool to create.
            pass

        elif self.max_concurrency == 1 or len(reserved) == 1:
            # One worker means ordinary sequential execution, on this thread.
            for slot, (_, child) in enumerate(reserved):
                results[slot] = self._execute_child(child)
        else:
            self._run_concurrently(reserved, results)

        if budget_error is not None:
            raise budget_error

        return [result for result in results if result is not None]

    def _run_concurrently(
        self,
        reserved: list[tuple[int, RLMContext]],
        results: list[RLMResult | None],
    ) -> None:
        """Run reserved children on a bounded pool, filling results by slot."""

        workers = min(self.max_concurrency, len(reserved))

        # The pool is closed on exit, so no worker outlives the call.
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="rlm-child",
        ) as pool:

            futures = {}

            for slot, (_, child) in enumerate(reserved):
                # Langfuse and OpenTelemetry track the current span in a
                # context variable, which a new thread does not inherit.
                # Copying the context keeps child spans under the RLM trace.
                context = contextvars.copy_context()

                futures[
                    pool.submit(context.run, self._execute_child, child)
                ] = slot

            for future, slot in futures.items():
                # A worker that raised is not allowed to cancel its siblings:
                # every slot is collected, and the failure becomes a result.
                try:
                    results[slot] = future.result()
                except Exception as error:
                    results[slot] = RLMResult(
                        answer="",
                        success=False,
                        depth=reserved[slot][1].depth,
                        metadata={
                            "error": str(error),
                            "error_type": "error",
                            "rate_limited": False,
                            "task": reserved[slot][1].task,
                        },
                    )

    def _announce_children(self, tasks: list[tuple[str, str]]) -> None:
        """Tell the handler the decomposition order, if it tracks identity."""

        setter = getattr(self.call_handler, "set_child_order", None)

        if setter is None:
            return

        setter([task for task, _ in tasks])

    def call_and_synthesize(
        self,
        parent: RLMContext,
        tasks: list[tuple[str, str]],
        synthesizer: RLMSynthesizer | None = None,
    ) -> RLMResult:
        """
        Execute multiple child RLM calls and synthesize their results
        into a single parent result.
        """

        results = self.call_many(
            parent=parent,
            tasks=tasks,
        )

        if synthesizer is None:
            synthesizer = RLMSynthesizer()

        with self.langfuse.start_as_current_observation(
            as_type="span",
            name="rlm-synthesis",
            input={
                "parent_task": parent.task,
                "child_count": len(results),
                "successful_children": sum(
                    result.success for result in results
                ),
            },
        ) as span:

            result = synthesizer.synthesize(results)

            span.update(
                output={
                    "success": result.success,
                    "answer": result.answer,
                }
            )

            return result

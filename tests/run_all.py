import subprocess
import sys


TESTS = [
    "test.test_state",
    "test.test_executor",
    "test.test_planner",
    "test.test_tracer",
    "test.test_evaluator",
    "test.test_agent_evaluation",
    "test.test_reflector",
    "test.test_memory",
    "test.test_planner_memory",
    "test.test_agent_memory",
    "test.test_agent_memory_integration",
    "test.test_agent_memory_e2e",
    "test.test_planner_retry",
    "test.test_agent_retry",
    "test.test_agent_retry_memory",
]


def main() -> None:
    print("=" * 60)
    print("Running NanoCode test suite")
    print("=" * 60)

    failed = []

    for test in TESTS:
        print(f"\n▶ Running {test}")

        result = subprocess.run(
            [sys.executable, "-m", test]
        )

        if result.returncode != 0:
            failed.append(test)
            print(f"❌ FAILED: {test}")
        else:
            print(f"✅ PASSED: {test}")

    print("\n" + "=" * 60)

    if failed:
        print("❌ Test suite failed")
        print("\nFailed tests:")

        for test in failed:
            print(f"  - {test}")

        sys.exit(1)

    print("✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
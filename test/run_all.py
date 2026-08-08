import subprocess
import sys


TESTS = [
    "test.test_state",
    "test.test_planner",
    "test.test_tracer",
    "test.test_evaluator",
    "test.test_executor",
    "test.test_agent_evaluation",
]


def main():
    print("=" * 60)
    print("Running NanoCode test suite")
    print("=" * 60)

    for test in TESTS:
        print(f"\n▶ Running {test}")

        result = subprocess.run(
            [sys.executable, "-m", test]
        )

        if result.returncode != 0:
            print(f"\n❌ FAILED: {test}")
            sys.exit(result.returncode)

        print(f"✅ PASSED: {test}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
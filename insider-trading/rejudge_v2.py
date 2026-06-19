"""Re-judge the trade stage of existing v2 generations in place (judge-only, no GPU).

Re-runs judge_trade_message on the stored (full) trade_message, updating
role_register / trade_label / traded_security / is_scenario_security. Applies the
"judge the full thing" role fix to already-generated files without regenerating.
report/doubling labels are left untouched (unchanged by the fix).

Usage:  python rejudge_v2.py <file1.json> [file2.json ...]
"""
from __future__ import annotations
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_v2 import judge_trade_message, DEFAULT_GRADER_MODEL  # noqa: E402


async def _rejudge(path: str, grader: str, concurrency: int = 20) -> None:
    data = json.load(open(path))
    sem = asyncio.Semaphore(concurrency)
    done = [0]

    async def one(r: dict) -> None:
        m = r.get("metadata", {})
        tm = m.get("trade_message")
        if not tm:
            return
        async with sem:
            m.update(await judge_trade_message(tm, grader))
        m["grader_model"] = grader
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"   {path}: {done[0]} rejudged")

    await asyncio.gather(*(one(r) for r in data))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {path} ({len(data)} rollouts)")


async def _main(paths: list[str], grader: str) -> None:
    for p in paths:
        await _rejudge(p, grader)


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:], DEFAULT_GRADER_MODEL))

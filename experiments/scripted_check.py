"""Quality gates without a real model: drive the taskset through Verifiers' Env API with a scripted
OpenAI-compatible model, so the tools, state replay, stop rule and reward all run for real.

Policies:
  oracle   replays the reference re-planner's orders, step by step       expect 1.0 (cost identical)
  nothing  never gives an order, just advances the clock                  expect the 'nothing' baseline
  silent   answers in text and never calls a tool                         expect the 'nothing' baseline
           (the rest of the day runs on the midnight orders)

usage: uv run python experiments/scripted_check.py [--tasks JSONL] [--policies oracle nothing silent]
"""

import argparse
import asyncio
import json
import os
import socket
from pathlib import Path

from aiohttp import web

from grid_dispatch.taskset import DEFAULT_TASKS
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.types import SamplingConfig
from verifiers.v1.utils.loaders import harness_config_type, load_environment

TASKS = DEFAULT_TASKS
ROWS: list[dict] = []


def schedule_orders(step: dict) -> list[dict]:
    """The re-planner sets whole schedules; each becomes one {"unit", "on"} order."""
    out = []
    for a in step["actions"]:
        if "battery" in a:
            out.append({"battery": {str(step["at"] + i): v for i, v in enumerate(a["battery"])}})
            continue
        on, x, h0 = [], a["schedule"], a["from"]
        for i, v in enumerate(x):
            if v and (i == 0 or not x[i - 1]):
                on.append([h0 + i, h0 + i + 1])
            elif v:
                on[-1][1] = h0 + i + 1
        out.append({"unit": a["unit"], "on": on})
    return out


def match_row(system_text: str) -> dict:
    """Identify the task from get_system's last line (grid area and date) plus its unit list."""
    for row in ROWS:
        w = row["world"]
        if f"Grid area {w['ba']}, {w['date']}." in system_text and all(u["id"] in system_text for u in w["units"]):
            return row
    raise LookupError("no task matches this system description")


def scripted_reply(policy: str, body: dict) -> dict:
    msgs = body["messages"]
    calls = [c for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    advances = sum(1 for c in calls if c["function"]["name"] == "advance_clock")
    message: dict = {"role": "assistant", "content": None}
    if policy == "silent" or advances >= 6:
        message["content"] = "Done."
    elif not calls:
        message["tool_calls"] = [_call(0, "get_system", {})]
    elif policy == "nothing":
        message["tool_calls"] = [_call(len(calls), "advance_clock", {})]
    else:   # oracle: at each step, the reference's orders, then advance
        system = next(m for m in msgs if m["role"] == "tool" and m.get("tool_call_id") == calls[0]["id"])
        row = match_row(system["content"] if isinstance(system["content"], str)
                        else "".join(p.get("text", "") for p in system["content"]))
        step = row["reference"]["actions"][advances]
        last = calls[-1]["function"]["name"]
        if last == "act":
            message["tool_calls"] = [_call(len(calls), "advance_clock", {})]
        else:
            message["tool_calls"] = [_call(len(calls), "act", {"actions": schedule_orders(step)})]
    return {"id": "chatcmpl-scripted", "object": "chat.completion", "created": 0, "model": body.get("model", "scripted"),
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _call(i: int, name: str, args: dict) -> dict:
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


async def serve_model(policy: str) -> tuple[web.AppRunner, str]:
    dialect = ChatDialect()

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        raw = scripted_reply(policy, body)
        if not dialect.streaming(body):
            return web.json_response(raw)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in dialect.stream_events(raw):
            await response.write(event)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner, f"http://127.0.0.1:{port}/v1"


async def run_policy(policy: str) -> list[tuple[str, float, dict]]:
    runner, url = await serve_model(policy)
    os.environ["SCRIPTED_API_KEY"] = "scripted"
    try:
        config = EvalConfig(
            env={"taskset": {"id": "grid_dispatch", "tasks_path": str(TASKS)},
                 "agent": {"harness": harness_config_type("null")(id="null"),
                           "runtime": {"type": "subprocess"}, "max_turns": 40}},
            select={"limit": len(ROWS)}, num_rollouts=1, model="scripted",
            sampling=SamplingConfig(max_tokens=1024),
            client=EvalClientConfig(base_url=url, api_key_var="SCRIPTED_API_KEY"),
        )
        env = load_environment(config.env)
        tasks = list(env.taskset.select(config.select))
        ctx = ModelContext(client=config.client, model=config.model, sampling=config.sampling)
        async with env.serving():
            episodes = await asyncio.gather(*(env.run_slot(slot, ctx) for task in tasks for slot in env.slots(task, 1)))
    finally:
        await runner.cleanup()
    out = []
    for ep in episodes:
        for tr in ep.traces:
            for err in tr.errors:
                print(f"  ERROR {tr.task.data.id}: {err}")
            out.append((tr.task.data.id, tr.reward, dict(tr.metrics)))
    return out


async def main():
    global TASKS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--policies", nargs="+", default=["oracle", "nothing", "silent"])
    args = ap.parse_args()
    TASKS = args.tasks
    ROWS.extend(json.loads(line) for line in TASKS.read_text().splitlines())
    by_id = {r["id"]: r for r in ROWS}
    ok = True
    for policy in args.policies:
        results = await run_policy(policy)
        print(f"== {policy}: {len(results)} episodes")
        for tid, reward, metrics in sorted(results):
            ref = by_id[tid]["reference"]
            want = 1.0 if policy == "oracle" else ref["baselines"]["nothing"]["score"]
            good = reward is not None and abs(reward - want) < 1e-4   # costs are stored to the cent
            ok &= good
            stats = {k: round(v, 3) for k, v in metrics.get("shift_stats", metrics).items() if k in ("reliability", "economy", "shed_mwh", "steps_played")}
            print(f"  {'ok ' if good else 'BAD'} {tid}: reward {reward} (expected {want:.4f}) {stats}")
        ok &= len(results) == len(ROWS)
    print("ALL GATES PASS" if ok else "SOME GATES FAILED")


if __name__ == "__main__":
    asyncio.run(main())

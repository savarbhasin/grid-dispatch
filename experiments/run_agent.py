"""Play one live shift with a model through an OpenAI-compatible API (tool calling), and score it.

usage: uv run python experiments/run_agent.py TASKS_JSONL TASK_ID --model M --base-url URL --api-key-var VAR
           [--max-tokens 65536] [--max-calls 80] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

from grid_dispatch.check import task_from_row
from grid_dispatch.shift import SIMULATE_PER_STEP, Shift
from grid_dispatch.world import World

PROMPT = """You are the dispatcher on duty at a power grid control room for one full day, 00:00 to 24:00.

Your job: keep the lights on at the lowest total cost. Every hour, the units you have switched on are \
dispatched automatically at least cost to meet the real demand, after free wind and solar. If the units \
that are on cannot cover demand, the shortfall is unserved energy and costs $5,000/MWh, far more than any fuel. \
Running units costs fuel, and every start-up has a start-up cost.

Your decisions are which units are on in each hour, and the battery. Rules:
- A unit needs `lead` hours of notice to start: from time T you can only order starts at T+lead or later.
- Once on, a unit must run at least `min_up` hours; once off, it must stay off at least `min_down` hours.
- Units can trip without warning and are then out for the day.
- Forecasts of demand and wind+solar are imperfect and get better as the hour approaches.

The day runs in steps: you look, check and act, then call `advance_clock`, which runs the next 4 hours of the \
real day and tells you what happened. Orders you give stay in force until you change them. Each step you may \
call `simulate` up to {sim} times to test changes on the current forecast before committing. The shift ends \
when the clock reaches 24:00; your score depends on the total cost of the whole day compared with a skilled \
dispatcher. Start with `get_system`, `get_status` and `get_forecast`."""

ACTION = {"type": "object", "description": "One order: {\"start\": unit, \"hour\": h} keeps the unit on from hour h "
          "(add \"until\": e to stop at e); {\"stop\": unit, \"hour\": h} keeps it off from h (optionally until e); "
          "{\"unit\": unit, \"on\": [[6, 14]]} replaces the unit's whole schedule from now; "
          "{\"battery\": {\"17\": 80, \"18\": -40}} sets battery MW for those hours (+ discharge, - charge).",
          "properties": {"start": {"type": "string"}, "stop": {"type": "string"}, "hour": {"type": "integer"},
                         "until": {"type": "integer"}, "unit": {"type": "string"},
                         "on": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                         "battery": {"type": "object", "additionalProperties": {"type": "number"}}}}
TOOLS = [
    {"type": "function", "function": {"name": "get_system", "description": "Unit data, battery and costs (fixed for the day).",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_status", "description": "Clock, each unit's state and schedule, battery, costs so far, what happened last step.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_forecast", "description": "Demand and wind+solar by hour as known now (actuals for past hours).",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "simulate", "description": "Test orders on a copy of the grid using the current forecast (the real world is unchanged). Limited calls per step.",
                                      "parameters": {"type": "object", "properties": {"actions": {"type": "array", "items": ACTION},
                                                                                      "until": {"type": "integer", "description": "Simulate up to this hour (default 24)."}},
                                                     "required": ["actions"]}}},
    {"type": "function", "function": {"name": "act", "description": "Give orders for real: starts, stops, battery. Each order is confirmed or refused with the reason.",
                                      "parameters": {"type": "object", "properties": {"actions": {"type": "array", "items": ACTION}}, "required": ["actions"]}}},
    {"type": "function", "function": {"name": "advance_clock", "description": "Run the next 4 hours of the real day and report what happened.",
                                      "parameters": {"type": "object", "properties": {}}}},
]


def call_tool(shift: Shift, name: str, args: dict) -> str:
    if name == "get_system":
        return shift.get_system()
    if name == "get_status":
        return shift.get_status()
    if name == "get_forecast":
        return shift.get_forecast()
    if name == "simulate":
        return shift.simulate(args.get("actions", []), args.get("until", 24))
    if name == "act":
        return shift.act(args.get("actions", []))
    if name == "advance_clock":
        return shift.advance_clock()
    return f"unknown tool {name}"


def complete(client: OpenAI, model: str, messages: list, max_tokens: int) -> tuple[dict, dict]:
    """One streamed completion; returns the assistant message (with tool calls) and usage."""
    stream = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, max_tokens=max_tokens,
                                            stream=True, stream_options={"include_usage": True})
    content, reasoning, calls, usage, finish = [], [], {}, {}, None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage.model_dump()
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        d = ch.delta
        finish = ch.finish_reason or finish
        if d.content:
            content.append(d.content)
        r = getattr(d, "reasoning_content", None) or (d.model_extra or {}).get("reasoning_content")
        if r:
            reasoning.append(r)
        for tc in d.tool_calls or []:
            c = calls.setdefault(tc.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            c["id"] = tc.id or c["id"]
            if tc.function and tc.function.name:
                c["function"]["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                c["function"]["arguments"] += tc.function.arguments
    msg = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg, {**usage, "finish": finish, "reasoning_chars": len("".join(reasoning))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", type=Path)
    ap.add_argument("task_id")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-var", required=True)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--max-calls", type=int, default=80)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    row = next(json.loads(l) for l in args.tasks.read_text().splitlines() if json.loads(l)["id"] == args.task_id)
    shift = Shift(World(task_from_row(row)), row["reference"]["cost"], row["reference"]["floor"])
    client = OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_var], timeout=1800)
    messages = [{"role": "user", "content": PROMPT.format(sim=SIMULATE_PER_STEP)}]
    usages, nudged, t0 = [], False, time.perf_counter()
    for i in range(args.max_calls):
        if shift.w.now >= 24:
            break
        msg, usage = complete(client, args.model, messages, args.max_tokens)
        usages.append(usage)
        messages.append(msg)
        names = [c["function"]["name"] for c in msg.get("tool_calls", [])]
        print(f"[{i:02d}] {time.perf_counter() - t0:6.0f}s clock {shift.w.now:02d}:00 out {usage.get('completion_tokens')} "
              f"tok finish={usage['finish']} calls={names}", flush=True)
        if not msg.get("tool_calls"):
            if nudged:
                break
            nudged = True
            messages.append({"role": "user", "content": f"The shift is not over: it is {shift.w.now:02d}:00. Keep going; call advance_clock when ready."})
            continue
        for c in msg["tool_calls"]:
            try:
                a = json.loads(c["function"]["arguments"] or "{}")
                out = call_tool(shift, c["function"]["name"], a)
            except Exception as e:   # malformed arguments go back to the model
                out = f"error: {e!r}"
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    auto = shift.w.now < 24
    while shift.w.now < 24:   # the shift doesn't stop: the rest of the day runs on the current orders
        shift.advance_clock()
    res = {**shift.result(), "auto_finished": auto, "model_calls": len(usages),
           "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usages),
           "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usages),
           "baseline_scores": row["reference"]["scores"], "seconds": round(time.perf_counter() - t0)}
    print(json.dumps({k: v for k, v in res.items() if k != "parts"}, indent=1))
    out = args.out or Path("experiments/results") / f"{args.task_id}--{args.model.split('/')[-1]}.json"
    out.write_text(json.dumps({"result": res, "usage": usages, "messages": messages}, indent=1))
    print(f"transcript: {out}")


if __name__ == "__main__":
    main()

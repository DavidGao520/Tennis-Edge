"""Phase 2 Agent package: shadow-trade → human-in-loop → restricted auto-execute.

Structure (per Phase 2 eng review):
  decisions.py — AgentDecision/EvAnalysis schemas + JSONL append log
  llm.py       — LLMProvider ABC + Gemini impl + BudgetTracker
  safety.py    — 5 kill switches (LLM 3x, WS stale+live, tick-logger
                 stale, budget cap, daily P&L floor)
  loop.py      — DB-tail reader, candidate queue, text-only prompt,
                 shadow/human-in-loop/auto executor modes
"""

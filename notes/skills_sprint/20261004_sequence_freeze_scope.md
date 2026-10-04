# Experimental hierarchical demo: freeze scope

Architecture: task-request freshness → discrete supervisor → select walking1499 or hold499 →123D actor input/37D action →Isaac simulation. State feedback supplies speed, actual Euler heading rate, tilt and feet contact for the handoff readiness gate. PPO trained the actors; the supervisor and watchdog are explicit rules. This is two policies with bidirectional skill handoffs, not a single actor merely receiving zero commands.

Freeze immutable weight/model/interface hashes, prospective gate values and evaluation configuration, plus results. Selection remains direct handoff from the prior seed42 protocol comparison; held-out retest data must not tune it. This is an experimental demonstration version; repository default controllers are not changed.

Scope: low-speed0.5m/s, plane, fixed material, no observation noise/external forces, inherited random world root pose, fixed initial joints/velocities. Normal stop/hold/restart and task-message transient/persistent loss. No actual perception stack, disturbance, high-speed, rough-terrain or MuJoCo success claim. No autonomous physical emergency action; persistent observation terminates after12s holding. Original skill/baseline acceptance remains separate.

Timing: simulation control period20ms, physics5ms; heartbeat100ms, freshness TTL300ms, two messages to recover freshness. Readiness trailing window500ms, confirmation1s, additional retention2s. Timeout12s holding, resumed-walk observation10s. These are simulated-time protocol budgets, not measured real-time CPU/GPU deadline compliance. Report communication detection delay, physical stopping confirmation, readiness/retention delay and post-resume tracking confirmation independently.

Evidence: raw first-episode states/obs/proposals/actions/contacts/stage events/input hashes; optimized independent actor/gate/metric/watchdog reconstruction; full denominators including failures. Supplemental physical distances exclude post-reset and post-terminal samples. An absent stopping confirmation has no invented zero drift. Net displacement and path length remain distinct.

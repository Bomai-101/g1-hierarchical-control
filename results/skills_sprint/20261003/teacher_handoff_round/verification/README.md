# Latest round independent reconstruction

216 cases checked under python -O. 162113 first-episode action rows reconstructed fromsaved observations andNPZ actors; maxTorch/NumPy proposalerror 8.5830688e-06. Base123D state/gravity/commands/joint mapping/previous applied action, march125D phase,0.5s smoothstep blend verified. All216recorded metrics/footedges/lift/fullpass recomputed.

old_direct_trace_pairs=36 refers to36 matched1499 *prefix* tracepairs (eachIsaactrace contains16envs), notwholecandidate traces. Prefixstate/obs/actions matchfirst-round beforeactualswitch; candidateweightsdiffer. Evidence source SHA andalloriginalevalinputSHAverified.

Fixed2s split givesinitial andremainingwindows. Splitpath sums equalwholepath; endpointnet magnitudes do notsum. Onlyfull10s survivors usedforwindowcomparisons. No assertion thatskillenteredafter2s. No model/control changes or performance-threshold changes.

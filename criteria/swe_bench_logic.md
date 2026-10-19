# SWE-bench Verified — Logic-Risk Ablation Criteria

## Ground Truth Note

Do NOT trust the agent's self-assessment or claims that "the patch looks correct". Agents routinely declare success on patches that fix the wrong file, address only a symptom, or are subtly broken.

## Criteria

### Root Cause Analysis {#root_cause}

Read the issue, identify the buggy behavior it describes, and trace it to the code that produces it. Decide whether the patch modifies the actual code path responsible for the bug, or only its symptoms. A patch that edits the buggy function or branch should score HIGH; a patch that catches the bad output downstream, special-cases the literal example in the issue, edits a caller to work around a buggy callee, or changes a default to dodge the broken path should score LOW. Judge by WHERE the change lands in the call stack — both small and larger fixes are valid as long as the edited lines are the ones whose behavior the issue actually depends on.

### L1 — Shadow Override {#logic_l1}

Trace every identifier used by the changed code to the definition that actually resolves at runtime. Score HIGH when imports, local bindings, closures, attributes, and inherited names resolve unambiguously to the intended entities. Score LOW when shadowing, rebinding, or an unbound name makes the patch call or read a different entity than intended. Ignore type, boundary, and control-flow issues unless name resolution is their direct cause.

### L2 — Type Contract Breach {#logic_l2}

Trace the concrete types entering and leaving the changed code, including nullable paths and implicit coercions. Score HIGH when arguments, operators, returns, and assigned values satisfy their declared or established contracts. Score LOW when a value can reach an operation or callee with an incompatible type. Ignore name resolution and boundary behavior that remains type-correct.

### L3 — Boundary Blindspot {#logic_l3}

Trace empty, zero, negative, null, single-element, first-iteration, last-iteration, and bounds paths affected by the patch. Score HIGH when every applicable boundary has defined correct behavior. Score LOW for unchecked indexing, off-by-one ranges, division by zero, or missing empty/null handling. Ignore an explicit designed rejection at a capacity limit.

### L4 — State Mutation Hazard {#logic_l4}

Trace reads, writes, and aliases of mutable state in execution order within one execution context. Score HIGH when mutation order preserves invariants and ownership is clear. Score LOW when a write invalidates a later read, iteration is mutated unsafely, or aliases expose an unexpected in-place change. Ignore multi-threaded or async interleavings that require another execution context.

### L5 — Control Flow Escape {#logic_l5}

List the non-resource postconditions required by the changed code, then inspect every return, raise, break, continue, and implicit exception path. Score HIGH when every exit performs required validation, state updates, markers, and notifications. Score LOW when an exit skips one of those operations. Ignore resource release and rollback pairing; that is outside this criterion.

### L6 — Callee Contract Mismatch {#logic_l6}

For each call affected by the patch, compare the caller's assumptions with the callee's actual return, exception, idempotency, ordering, and side-effect contract. Score HIGH when every relied-on guarantee is established by local code or authoritative API behavior. Score LOW when the patch assumes a guarantee the callee does not provide. Ignore purely local type or boundary errors already covered by another criterion.

### Empirical Verification {#verification}

Look at the commands the agent actually ran and what they printed, not what the agent claimed in its narration. Reward agents that (a) constructed a reproducer for the failure described in the issue, (b) observed the failure before applying the fix, (c) observed the expected correct behavior after the fix, and (d) ran the existing tests in the affected module without breaking them. Trust observed command output over the agent's narration of it. Penalize agents that declared success without running anything, misread their own command output (e.g. compared a literal string to itself, ignored a traceback, claimed a test passed when it errored), or edited the code again after the last successful verification step so the final patch is untested.

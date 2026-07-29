# Haiqu SDK — agent orientation

Python SDK for the Haiqu quantum cloud. The public API lives on the `Haiqu` class in
`haiqu/sdk/quantum_haiqu.py`, exposed as a module-level singleton:

```python
from haiqu.sdk import haiqu   # NOT `import haiqu` — that is an empty namespace package
```

## Canonical workflow

Every workflow is these explicit steps. The SDK already provides each one — check the `Haiqu`
class methods before writing any helper or wrapper.

```python
haiqu.login()                  # reads HAIQU_API_KEY when no key is passed; must be called
# or: haiqu.login(api_access_key="your api key")
haiqu.init("my experiment")    # otherwise jobs land in a shared default experiment

# 1. Build a LOGICAL circuit (plain Qiskit).
# 2. Optional logical-level features (act BEFORE transpilation):
#    state_compression, vector_loading, distribution_loading, mps_loading,
#    function_loading, observable_backpropagation
# 3. device = haiqu.get_device("fake_torino")
#    tqc = haiqu.transpile(qc, device)          # optional cloud-side transpilation
# 4. job = haiqu.run(tqc, shots=1000, device=device, use_mitigation=True)
#    results = job.result()
```

## The `HaiquCircuitGate` — the #1 trap

Data-loading job results return a `HaiquCircuitGate` (`haiqu/sdk/gates.py`),
whose `definition` is `None`, as it is primarily intended as a client-side handle for a server-side
circuit. State-compression job results return a `CircuitModel`, which does include a `qpy` attribute with the circuit contents;
call `.to_gate()` if you need to embed that cloud circuit in a larger Qiskit circuit. A
`HaiquCircuitGate` carries a `circuit_id`; a `CircuitModel` stores the same cloud circuit ID as
`.id`.

| Task | Use | Never |
|---|---|---|
| Depth / gate counts | `haiqu.transpile(qc, device).analytics` (e.g. `.depth`, `.depth_2q`, `.gates_2q`) or `.core_metrics(widget=False)` | `qc.depth()` / local counts on `HaiquCircuitGate` |
| Transpile | `haiqu.transpile(qc, device)` | relying on local `qiskit.transpile` / `qc.decompose()` to expand `HaiquCircuitGate` |
| Execute | `haiqu.run(...)` | local Aer / `Statevector` |
| Embed in a larger circuit | `circuit.append(gate, ...)` / `circuit.compose(...)` (works client-side) | rebuilding the sub-circuit manually |
| Persist a cloud circuit | store `circuit.id`; retrieve with `haiqu.get_circuit(circuit_id)` | pickling `CircuitModel` objects as durable storage |

## Quick reference

- All `device_id` values, including `"aer_simulator"`, dispatch to the Haiqu cloud; there is no
  local execution path.
- `haiqu.run(...).result()` returns measurement probability distributions (`dict[str, float]`;
  named `quasi_dists` in the wire schema) when no observables are supplied, and expectation
  values when observables are supplied. It does not return integer counts.
- Simulator qubit limits: statevector 24; MPS (`options={"method": "matrix_product_state"}`)
  has no strict limit (including with fake devices); noisy simulation without MPS
  (`fake_*` devices or a `noise_model`) is limited to 12.
- `list_*` methods render a Jupyter widget and return `None` by default; in scripts always pass
  `widget=False`.
- Estimate before spending: `*_loading_estimates(...)` (for example
  `distribution_loading_estimates`, `vector_loading_estimates`) and `haiqu.run(..., dry_run=True)` with
  `job.estimated_qpu_cost`.
- Job metrics: loading jobs expose `job.quality` (state fidelity vs. the target); compression
  jobs expose `job.quality` (noiseless proxy — validate observables, do not treat it as accuracy).
- `CircuitModel.analytics` is computed cloud-side and may be `None` until ready; prefer
  `core_metrics(widget=False)` / `wait_for_analytics(widget=False)`, which block until populated.
- Multiple calls to job methods (such as data loading) with identical inputs will generate
  multiple results. To save time, persist the job ID rather than re-running the job.
- All state loaders produce statevectors in Qiskit little-endian order (rightmost bit of a basis
  index = qubit 0), but their inputs are indexed differently: `vector_loading` /
  `block_vector_loading` take `data[i]` = amplitude for basis index `i`; `function_loading` /
  `distribution_loading` discretize `[interval_start, interval_end]` into bins, mapping increasing
  bin `i` to amplitude index `i`; `mps_loading` maps site tensor `A_i` to qubit `q_i`.
  `entangled_manifold_embedding` uses feature list order, not amplitude indexing.
- `run(parameters=...)` binds values in `circuit.parameters` order (usually alphabetical when parameters are created one-by-one, independent of gate order; `ParameterVector` uses vector index order — inspect `list(circuit.parameters)` to confirm).
- `NonlinearVariationalProblem` term strings (Pauli and projector `0`/`1` symbols) use Qiskit's
  reversed-order convention: rightmost character = qubit 0. Projector terms must use
  `(term_string, coefficient)` pairs; `SparsePauliOp` cannot represent `0`/`1`.
- Job/circuit model classes are not re-exported at package level; import them from the schemas
  module for type annotations: `from haiqu.sdk.schemas import RunJobModel, CircuitModel`.

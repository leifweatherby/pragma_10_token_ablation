Run Outputs
===========

Outputs from generation runs are stored as individual JSON files under a common
subdirectory. There are two different output types, which correspond to the two
run types described in the [run types documentation](./runtypes.md).

Onepass
-------

File tree:

```
/path/to/data-dir/model-name
├── run_info.json               # Basic metadata about the run
├── cot_off.json                # Array of QAWithResponseObjects from CoT-OFF
└── cot_on.json                 # Array of QAWithResponseObjects from CoT-ON
```

Ablations
---------

File tree:

```
/path/to/data-dir/model-name
├── baseline.json               # CoT-OFF responses and verdicts
├── run_info.json               # Basic metadata about the run
├── step_000.json               # CoT-ON responses and verdicts with no ablations
├── step_001.json               # CoT-ON responses and verdicts with ablations
├── step_XXX.json               # Further ablation steps
└── step_n.json                 # Last step of CoT-ON with ablations, where acc(CoT-ON) < acc(CoT-OFF)
```

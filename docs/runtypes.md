Runtypes
========

We enable two different runtypes:

Onepass
-------

A `onepass` run generates two types of model responses to QA pairs:

1. CoT-OFF: Reasoning was not enabled for the response
2. CoT-ON: Reasoning was enabled for the response

This run type does not perform any ablations. It is just for comparing how an
LLM responds when reasoning is turned on/off.

Ablations
---------

An `ablations` run performs ablations on a model's reasoning traces, using a
second LLM to judge the results:

1. Run CoT-OFF to establish a baseline accuracy
2. Run CoT-ON without ablations and judge performance
3. Find distinctive n-grams in CoT-ON reasoning traces + responses and ablate
   them based on a cutoff value; right now, this cutoff value is a quantile
   (e.g., n-grams with fighting words scores >= 98th quantile)
4. Rerun CoT-ON with the ablated n-grams
5. Repeat steps 3-4 until CoT-ON accuracy falls below CoT-OFF or max steps is
   reached

Flow Diagram
------------

```
┌──────────────────────────────────────────────────────────────┐
│                 Main Entry (__main__.py)                     │
│                                                              │
│  1. Load config from TOML                                    │
│  2. Load model & tokenizer                                   │
│  3. Load examples from dataset                               │
└────────────────────────┬─────────────────────────────────────┘
                         │
                    Parse runtype
                         │
         ┌───────────────┴────────────────┐
         │                                │
         ▼                                ▼
    ┌─────────────┐             ┌──────────────────┐
    │   ONEPASS   │             │    ABLATIONS     │
    └─────────────┘             └──────────────────┘
         │                                │
         │                                ├─► Initialize LLMJudge
         │                                │   (requires API key)
         │                                │
         ▼                                ▼
    ┌──────────────────────┐    ┌──────────────────────────┐
    │  run_onepass()       │    │  AblationRunner.run()    │
    ├──────────────────────┤    ├──────────────────────────┤
    │ For each run config: │    │ 1. Run CoT-OFF baseline  │
    │                      │    │    (all examples)        │
    │ • CoT-OFF (if OFF)   │    │                          │
    │   - Generate         │    │ 2. Judge baseline        │
    │   - Store responses  │    │                          │
    │                      │    │ 3. Run iterative steps:  │
    │ • CoT-ON (if ON)     │    │    (step 0 to max_steps) │
    │   - Generate         │    │                          │
    │   - Store responses  │    │                          │
    └──────────────────────┘    └────────┬─────────────────┘
         │                               │
         │                               ▼
         │                      ┌────────────────────────────────┐
         │                      │  AblationRunner._process_step()│
         │                      ├────────────────────────────────┤
         │                      │ For each step:           │
         │                      │                          │
         │                      │ a) Filter active exs     │
         │                      │    (only passing ones)   │
         │                      │                          │
         │                      │ b) Generate CoT-ON with  │
         │                      │    NGramMasking on       │
         │                      │    ablated n-grams       │
         │                      │                          │
         │                      │ c) Judge new responses   │
         │                      │                          │
         │                      │ d) Calculate             │
         │                      │    distinctiveness using │
         │                      │    metric (e.g. fighting │
         │                      │    words)                │
         │                      │                          │
         │                      │ e) Find n-grams above    │
         │                      │    cutoff threshold      │
         │                      │                          │
         │                      │ f) Check if performance  │
         │                      │    improved (is_better)  │
         │                      │                          │
         │                      │ g) Record step results   │
         │                      │                          │
         │                      │ h) Decide: continue or         │
         │                      │    stop ablation?              │
         │                      └─────────┬──────────────────────┘
         │                                │
         │                      ┌─────────┴──────────┐
         │                      │                    │
         │             Accuracy > CoT-OFF?  Accuracy < Cot-OFF?
         │                      │                    │
         │                      │                    │
         │                      ▼                    ▼
         │           _process_step() [loop]    Stop ablating
         │                                           │
         │                                           │
         │                                           │
         └────────────────────┬──────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  results.save_run()  │
                    ├──────────────────────┤
                    │ Output to directory: │
                    │                      │
                    │ OnePass:             │
                    │ • run_info.json      │
                    │ • cot_off.json       │
                    │ • cot_on.json        │
                    │                      │
                    │ Ablations:           │
                    │ • run_info.json      │
                    │ • baseline.json      │
                    │ • step_000.json      │
                    │ • step_001.json      │
                    │ • ...                │
                    │ • step_NNN.json      │
                    └──────────────────────┘
```

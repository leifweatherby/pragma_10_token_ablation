/**
 * StepViewer component - Navigate through ablation steps and view performance metrics.
 */

import { createSignal, createResource, For, Show, createMemo } from 'solid-js';
import { api } from '../api/client';
import type { Step } from '../types';
import { StepNgramAnalysis } from './StepNgramAnalysis';
import { CollapsibleQuestionsSection } from './CollapsibleQuestionsSection';

export function StepViewer(props: {
  runName: string;
}) {
  const [selectedStep, setSelectedStep] = createSignal<number | 'baseline'>('baseline');
  const [stepList] = createResource(() => api.listSteps(props.runName));
  const [baseline] = createResource(() => api.getBaseline(props.runName));
  const [runInfo] = createResource(() => api.getRunInfo(props.runName));

  // Load all step data for comparison
  const [allStepsData] = createResource(
    () => stepList(),
    async (steps) => {
      if (!steps) return [];
      const promises = steps.map(s => api.getStep(props.runName, s.step));
      return Promise.all(promises);
    }
  );

  const [stepData] = createResource(
    () => {
      const step = selectedStep();
      return step === 'baseline' ? null : step;
    },
    (stepNum) => (stepNum !== null ? api.getStep(props.runName, stepNum) : null)
  );

  const currentData = () => {
    return selectedStep() === 'baseline' ? baseline() : stepData();
  };

  const currentVerdicts = createMemo(() => {
    if (selectedStep() === 'baseline') {
      return baseline()?.verdicts || [];
    } else {
      return stepData()?.verdicts || [];
    }
  });

  return (
    <div>
      {/* Performance Comparison (Sticky Container) */}
      <div class="sticky top-0 z-20 bg-gray-50 dark:bg-gray-900 shadow-md">
        {/* Performance Comparison Bar Graph */}
        <div class="max-w-[650px] mx-auto px-6 py-6">
          <Show when={baseline.loading || stepList.loading || allStepsData.loading}>
            <div class="bg-white dark:bg-gray-800 shadow p-6">
              <h3 class="text-lg font-semibold mb-4">Performance Comparison vs Baseline</h3>
              <div class="flex items-center justify-center py-8">
                <div class="animate-spin h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
            </div>
          </Show>

          <Show when={baseline() && allStepsData()}>
            <div class="bg-white dark:bg-gray-800 shadow p-6">
              <h3 class="text-lg font-semibold mb-4">Performance Comparison vs Baseline</h3>
              <div class="space-y-3">
                {/* Baseline Bar */}
                <PerformanceBar
                  label="Baseline"
                  numCorrect={baseline()!.performance.num_correct}
                  numTotal={baseline()!.performance.num_total}
                  baselineCorrect={baseline()!.performance.num_correct}
                  isSelected={selectedStep() === 'baseline'}
                  onClick={() => setSelectedStep('baseline')}
                />

                {/* Step Bars */}
                <For each={allStepsData()}>
                  {(step: Step) => (
                    <PerformanceBar
                      label={`Step ${String(step.step).padStart(3, '0')}`}
                      numCorrect={step.performance.num_correct}
                      numTotal={step.performance.num_total}
                      baselineCorrect={baseline()!.performance.num_correct}
                      isSelected={selectedStep() === step.step}
                      onClick={() => setSelectedStep(step.step)}
                    />
                  )}
                </For>
              </div>
            </div>
          </Show>
        </div>
      </div>

      {/* Step content - show loading or data */}
      <div class="max-w-[650px] mx-auto px-6 py-6 space-y-6 mb-[100px]">
        <Show
          when={!((selectedStep() === 'baseline' && baseline.loading) || (selectedStep() !== 'baseline' && stepData.loading))}
          fallback={
            <div class="bg-white dark:bg-gray-800 shadow p-6">
              <div class="flex items-center justify-center py-8">
                <div class="animate-spin h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
            </div>
          }
        >
          <Show when={currentData()}>
            {(data) => {
              const isBaseline = selectedStep() === 'baseline';
              const stepData = data() as Step;
              const stepNumber = selectedStep() as number;

              // Calculate comparison metrics for steps
              const accuracyDelta = !isBaseline && baseline()
                ? stepData.performance.accuracy - baseline()!.performance.accuracy
                : 0;
              const correctDelta = !isBaseline && baseline()
                ? stepData.performance.num_correct - baseline()!.performance.num_correct
                : 0;

              return (
                <div class="space-y-6">
                  {/* Step Header with comparison to baseline */}
                  <Show when={!isBaseline}>
                    <div class="bg-white dark:bg-gray-800 shadow p-6 border-l-4 border-blue-600">
                      <h2 class="text-2xl font-bold mb-3">Step {String(stepNumber).padStart(3, '0')}</h2>
                      <div class="space-y-2">
                        <MetricCard
                          label="Accuracy"
                          value={`${stepData.performance.accuracy.toFixed(1)}%`}
                          valueClass={accuracyDelta >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}
                        />
                        <MetricCard
                          label="Change"
                          value={`${accuracyDelta >= 0 ? '+' : ''}${accuracyDelta.toFixed(1)}%`}
                          valueClass={accuracyDelta >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}
                        />
                        <MetricCard
                          label="Correct"
                          value={`${stepData.performance.num_correct}/${stepData.performance.num_total}`}
                        />
                        <MetricCard
                          label="Delta"
                          value={`${correctDelta >= 0 ? '+' : ''}${correctDelta}`}
                          valueClass={correctDelta >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}
                        />
                      </div>
                    </div>
                  </Show>

                  {/* Step-specific info */}
                  <Show when={!isBaseline && stepData.distinctiveness_metrics}>
                    {(metrics) => (
                      <div class="bg-white dark:bg-gray-800 shadow p-6">
                        <h3 class="text-xl font-semibold mb-4">Distinctiveness Metrics</h3>
                        <div class="space-y-2">
                          <Show when={runInfo()}>
                            <MetricCard
                              label="Metric"
                              value={runInfo()!.metric}
                            />
                          </Show>
                          <MetricCard
                            label="Cutoff"
                            value={metrics().cutoff.toFixed(3)}
                          />
                          <MetricCard
                            label="Min Score"
                            value={metrics().score_min.toFixed(2)}
                          />
                          <MetricCard
                            label="Max Score"
                            value={metrics().score_max.toFixed(2)}
                          />
                          <MetricCard
                            label="Mean Score"
                            value={metrics().score_mean.toFixed(2)}
                          />
                          <MetricCard
                            label="Median Score"
                            value={metrics().score_median.toFixed(2)}
                          />
                        </div>
                      </div>
                    )}
                  </Show>

                  {/* N-gram Analysis */}
                  <Show when={!isBaseline && runInfo()}>
                    <StepNgramAnalysis
                      stepData={data() as Step}
                      runName={props.runName}
                      modelName={runInfo()!.model}
                    />
                  </Show>

                  {/* Questions Section - only show for steps, not baseline */}
                  <Show when={!isBaseline}>
                    <div class="bg-white dark:bg-gray-800 shadow p-6">
                      <h3 class="text-xl font-semibold mb-4">Questions</h3>
                      <CollapsibleQuestionsSection
                        verdicts={currentVerdicts()}
                        stepName={`step ${String(selectedStep()).padStart(3, '0')}`}
                        defaultExpanded={false}
                      />
                    </div>
                  </Show>
                </div>
              );
            }}
          </Show>
        </Show>

        <Show when={baseline.error || stepData.error}>
          <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-6">
            <div class="text-red-600 dark:text-red-400">Error: {baseline.error?.message || stepData.error?.message}</div>
          </div>
        </Show>
      </div>
    </div>
  );
}

function MetricCard(props: { label: string; value: string | number; valueClass?: string }) {
  return (
    <div class="text-sm">
      <span class="font-bold">{props.label}:</span>{' '}
      <span class={props.valueClass || ''}>{props.value}</span>
    </div>
  );
}

interface PerformanceBarProps {
  label: string;
  numCorrect: number;
  numTotal: number;
  baselineCorrect: number;
  isSelected: boolean;
  onClick: () => void;
}

function PerformanceBar(props: PerformanceBarProps) {
  const numIncorrect = props.numTotal - props.numCorrect;
  const correctPct = (props.numCorrect / props.numTotal) * 100;
  const incorrectPct = (numIncorrect / props.numTotal) * 100;
  const accuracy = (props.numCorrect / props.numTotal) * 100;

  // Calculate opacity based on performance vs baseline
  const baselineAccuracy = (props.baselineCorrect / props.numTotal) * 100;
  const isBetter = accuracy > baselineAccuracy;
  const isBaseline = props.numCorrect === props.baselineCorrect;
  const opacity = isBaseline ? 1 : isBetter ? 0.9 : 0.6;

  return (
    <button
      onClick={props.onClick}
      class={`w-full text-left group transition-all ${props.isSelected
        ? 'border-2 border-blue-600 bg-blue-50 dark:bg-blue-900/20 shadow-md'
        : 'border border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
        }`}
      title={`${props.numCorrect} correct, ${numIncorrect} incorrect (${accuracy.toFixed(1)}%)`}
    >
      <div class="flex items-center p-2">
        <div class="w-32 text-sm font-medium flex-shrink-0 font-mono">{props.label}</div>
        <div class="flex-1 flex h-8 overflow-hidden">
          {/* Green bar for correct */}
          <div
            class="flex items-center justify-center text-xs font-semibold text-white transition-all hover:opacity-100"
            style={{ width: `${correctPct}%`, opacity, 'background-color': '#22c55e' }}
          >
            <span class="opacity-0 group-hover:opacity-100">{props.numCorrect}</span>
          </div>
          {/* Red bar for incorrect */}
          <div
            class="flex items-center justify-center text-xs font-semibold text-white transition-all hover:opacity-100"
            style={{ width: `${incorrectPct}%`, opacity, 'background-color': '#ef4444' }}
          >
            <span class="opacity-0 group-hover:opacity-100">{numIncorrect}</span>
          </div>
        </div>
        <div class="w-20 text-sm text-right flex-shrink-0 ml-2 font-mono">
          {accuracy.toFixed(1)}%
        </div>
      </div>
    </button>
  );
}

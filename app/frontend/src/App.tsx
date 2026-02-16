import { createSignal, Show, createResource } from 'solid-js';
import { RunOverview } from './components/RunOverview';
import { StepViewer } from './components/StepViewer';
import { api } from './api/client';

function App() {
  const [selectedRun, setSelectedRun] = createSignal<string | null>(null);

  return (
    <div class="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <div class="max-w-[650px] mx-auto">
        <Show when={!selectedRun()} fallback={<RunDetail runName={selectedRun()!} onBack={() => setSelectedRun(null)} />}>
          <RunOverview onSelectRun={setSelectedRun} />
        </Show>
      </div>
    </div>
  );
}

function RunDetail(props: { runName: string; onBack: () => void }) {
  const [runInfo] = createResource(() => api.getRunInfo(props.runName));

  const formatAccuracy = (acc: number) => `${(acc * 100).toFixed(1)}%`;
  const formatDelta = (delta: number) => {
    const sign = delta >= 0 ? '+' : '';
    return `${sign}${(delta * 100).toFixed(1)}%`;
  };

  return (
    <div>
      {/* Sticky top bar */}
      <div class="sticky top-0 z-10 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm">
        <div class="max-w-[650px] mx-auto px-6 py-4">
          <div class="flex items-center justify-between mb-3">
            <button
              onClick={props.onBack}
              class="text-blue-600 hover:text-blue-700 flex items-center text-sm"
            >
              ← Back to Runs
            </button>
          </div>
          <h1 class="text-2xl font-bold mb-4">{props.runName}</h1>

          {/* Run Metadata */}
          <Show when={runInfo()}>
            {(info) => (
              <div class="max-w-[300px] mx-auto border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
                <div class="space-y-2 text-sm">
                  <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400">Model:</span>
                    <span class="font-medium" title={info().model}>
                      {info().model.split('/').pop()}
                    </span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400">Steps:</span>
                    <span class="font-medium">
                      {info().total_steps_completed} / {info().max_steps}
                    </span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400">N-grams ablated:</span>
                    <span class="font-medium">{info().total_ngrams_ablated?.toLocaleString() ?? 'N/A'}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400">Final accuracy:</span>
                    <span class="font-medium">{formatAccuracy(info().final_accuracy)}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400">Accuracy delta:</span>
                    <span
                      class={`font-medium ${
                        info().final_accuracy_delta >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                      }`}
                    >
                      {formatDelta(info().final_accuracy_delta)}
                    </span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400">Status:</span>
                    <span
                      class={`font-medium ${
                        info().termination_reason === 'degraded'
                          ? 'text-yellow-600 dark:text-yellow-400'
                          : 'text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {info().termination_reason}
                    </span>
                  </div>
                  <Show when={info().strategy}>
                    <div class="flex justify-between">
                      <span class="text-gray-600 dark:text-gray-400">Strategy:</span>
                      <span class="font-medium">{info().strategy}</span>
                    </div>
                  </Show>
                  <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400">Examples:</span>
                    <span class="font-medium">{info().num_examples}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-gray-600 dark:text-gray-400">Seed:</span>
                    <span class="font-medium">{info().seed}</span>
                  </div>
                </div>
              </div>
            )}
          </Show>
        </div>
      </div>

      <StepViewer runName={props.runName} />
    </div>
  );
}

export default App;

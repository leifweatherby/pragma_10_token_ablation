/**
 * RunOverview component - Lists available runs and displays metadata.
 */
import { createResource, For, Show } from 'solid-js';
import { api } from '../api/client';
import type { Run } from '../types';

/**
 * Extract date from run name (assumes format like "pragma-xyz" where xyz is a timestamp-based suffix)
 * Falls back to the run name for sorting if no date pattern is found.
 */
function extractDateFromRunName(runName: string): Date {
  // Try to extract a date pattern from the run name
  // Common patterns might be: pragma-20231015, pragma-2023-10-15, etc.
  const dateMatch = runName.match(/(\d{4})-?(\d{2})-?(\d{2})/);
  if (dateMatch) {
    return new Date(parseInt(dateMatch[1]), parseInt(dateMatch[2]) - 1, parseInt(dateMatch[3]));
  }

  // If no explicit date, try to use the suffix as a timestamp indicator
  // This assumes run names are created chronologically
  return new Date(0); // Fallback for sorting
}

/**
 * Sort runs by date (newest first), extracted from run name
 */
function sortRunsByDate(runs: Run[]): Run[] {
  return [...runs].sort((a, b) => {
    const dateA = extractDateFromRunName(a.name);
    const dateB = extractDateFromRunName(b.name);
    return dateB.getTime() - dateA.getTime();
  });
}

export function RunOverview(props: { onSelectRun: (runName: string) => void }) {
  const [runs] = createResource(api.getRuns);

  return (
    <div class="p-6 max-w-4xl mx-auto">
      <div class="mb-12 text-center">
        <h1 class="text-5xl font-bold mb-3 ">
          Pragma
        </h1>
        <p class="text-lg text-gray-600 dark:text-gray-400">
          Interpreting tokens in LLM reasoning traces
        </p>
      </div>

      <Show when={runs.loading}>
        <div class="bg-white dark:bg-gray-800 shadow p-6 rounded-lg">
          <div class="flex items-center justify-center space-x-3">
            <div class="animate-spin h-6 w-6 border-b-2 border-blue-600"></div>
            <span class="text-gray-600 dark:text-gray-400">Loading runs...</span>
          </div>
        </div>
      </Show>

      <Show when={runs.error}>
        <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-6 rounded-lg">
          <div class="text-red-600 dark:text-red-400">Error loading runs: {runs.error.message}</div>
        </div>
      </Show>

      <Show when={runs()}>
        <div class="flex flex-col gap-3">
          <For each={sortRunsByDate(runs()!)}>
            {(run: Run) => (
              <RunCard run={run} onSelect={() => props.onSelectRun(run.name)} />
            )}
          </For>
        </div>
      </Show>
    </div>
  );
}

function RunCard(props: { run: Run; onSelect: () => void }) {
  const { run } = props;

  const formatDate = (runName: string): string => {
    const date = extractDateFromRunName(runName);
    if (date.getTime() === 0) {
      // No date found, return the run name suffix
      return runName.split('-').pop() || runName;
    }
    return date.toLocaleDateString();
  };

  return (
    <button
      onClick={props.onSelect}
      class="bg-white dark:bg-gray-800 shadow-md p-4 text-left hover:shadow-lg hover:scale-[1.01] transition-all border border-gray-200 dark:border-gray-700 rounded-lg"
    >
      <div class="flex items-baseline justify-between">
        <h2 class="text-lg font-semibold truncate flex-1" title={run.name}>
          {run.name}
        </h2>
        <span class="text-sm text-gray-500 dark:text-gray-400 ml-4 whitespace-nowrap">
          {formatDate(run.name)}
        </span>
      </div>
    </button>
  );
}

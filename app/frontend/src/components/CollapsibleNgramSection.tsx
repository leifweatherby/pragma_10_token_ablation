/**
 * CollapsibleNgramSection component - Reusable accordion for displaying collapsible n-gram lists.
 */

import { createSignal, For, Show, createResource } from 'solid-js';
import { decodeNgramsForModel } from '../utils/tokenizer';

interface CollapsibleNgramSectionProps {
  title: string;
  count: number;
  ngrams: number[][];
  modelName: string;
  defaultExpanded?: boolean;
  colorClass?: string;
}

export function CollapsibleNgramSection(props: CollapsibleNgramSectionProps) {
  const [expanded, setExpanded] = createSignal(props.defaultExpanded ?? false);

  // Decode n-grams when expanded
  const [decodedNgrams] = createResource(
    () => (expanded() ? props.ngrams : null),
    async (ngrams) => {
      if (!ngrams || ngrams.length === 0) return [];
      try {
        return await decodeNgramsForModel(ngrams, props.modelName);
      } catch (error) {
        console.error('Error decoding n-grams:', error);
        // Return token IDs as fallback
        return ngrams.map(ngram => ngram.join(','));
      }
    }
  );

  const toggleExpanded = () => {
    setExpanded(!expanded());
  };

  // Determine color classes
  const headerColorClass = props.colorClass || 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700';

  return (
    <div class="border border-gray-300 dark:border-gray-600">
      <button
        onClick={toggleExpanded}
        class={`w-full text-left p-2 sm:p-3 transition-colors ${headerColorClass}`}
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="text-gray-400 text-lg">{expanded() ? '▼' : '▶'}</span>
            <span class="font-semibold text-sm sm:text-base">{props.title}</span>
            <span class="text-xs sm:text-sm text-gray-500">({props.count})</span>
          </div>
        </div>
      </button>

      <Show when={expanded()}>
        <div class="border-t border-gray-300 dark:border-gray-600 p-2 sm:p-3">
          {/* Loading state */}
          <Show when={decodedNgrams.loading}>
            <div class="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
              <div class="animate-spin h-4 w-4 border-b-2 border-blue-600"></div>
              <span>Decoding n-grams...</span>
            </div>
          </Show>

          {/* Error state */}
          <Show when={decodedNgrams.error}>
            <div class="text-sm text-red-600 dark:text-red-400">
              Error decoding n-grams: {decodedNgrams.error?.message}
            </div>
          </Show>

          {/* Empty state */}
          <Show when={!decodedNgrams.loading && !decodedNgrams.error && props.count === 0}>
            <div class="text-sm text-gray-500 italic">No n-grams to display</div>
          </Show>

          {/* N-grams list */}
          <Show when={!decodedNgrams.loading && !decodedNgrams.error && decodedNgrams()}>
            <div class="space-y-1 max-h-96 overflow-y-auto">
              <For each={decodedNgrams()}>
                {(ngram, idx) => (
                  <div class="text-xs sm:text-sm font-mono bg-white dark:bg-gray-900 p-2 border border-gray-200 dark:border-gray-700 break-words">
                    <div class="grid grid-cols-[1fr_auto] gap-3">
                      <div class="break-words">
                        {ngram || props.ngrams[idx()].join(',')}
                      </div>
                      <div class="text-gray-500 dark:text-gray-400 text-right whitespace-nowrap">
                        [{props.ngrams[idx()].join(', ')}]
                      </div>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
}

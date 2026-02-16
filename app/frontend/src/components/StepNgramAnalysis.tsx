/**
 * StepNgramAnalysis component - Integrated n-gram analysis for a specific step.
 * Displays beneath step details in the Steps tab.
 */

import { createSignal, Show, createMemo } from 'solid-js';
import { decodeTokenIdsForModel } from '../utils/tokenizer';
import { CollapsibleNgramSection } from './CollapsibleNgramSection';
import type { Step } from '../types';

interface StepNgramAnalysisProps {
  stepData: Step;
  runName: string;
  modelName: string;
}

export function StepNgramAnalysis(props: StepNgramAnalysisProps) {
  const [searchQuery, _setSearchQuery] = createSignal('');
  const [sortBy, _setSortBy] = createSignal<'score' | 'ngram'>('score');

  // Decode n-grams to human-readable text
  const [decodedNgrams, setDecodedNgrams] = createSignal<Map<string, string>>(new Map());

  // Decode all n-grams when step data changes
  createMemo(async () => {
    const step = props.stepData;
    if (!step || !step.distinctiveness_results) return;

    const decoded = new Map<string, string>();
    for (const result of step.distinctiveness_results) {
      const key = JSON.stringify(result.ngram);
      try {
        const text = await decodeTokenIdsForModel(result.ngram, props.modelName);
        decoded.set(key, text);
      } catch (e) {
        decoded.set(key, `[${result.ngram.join(', ')}]`);
      }
    }
    setDecodedNgrams(decoded);
  });

  const sortedNgrams = createMemo(() => {
    const step = props.stepData;
    if (!step || !step.distinctiveness_results) return [];

    let ngrams = [...step.distinctiveness_results];
    const query = searchQuery().toLowerCase();

    // Filter by search (search in decoded text)
    if (query) {
      ngrams = ngrams.filter((n) => {
        const key = JSON.stringify(n.ngram);
        const decoded = decodedNgrams().get(key) || '';
        return decoded.toLowerCase().includes(query);
      });
    }

    // Sort
    if (sortBy() === 'score') {
      ngrams.sort((a, b) => b.score - a.score);
    } else {
      ngrams.sort((a, b) => {
        const keyA = JSON.stringify(a.ngram);
        const keyB = JSON.stringify(b.ngram);
        const decodedA = decodedNgrams().get(keyA) || '';
        const decodedB = decodedNgrams().get(keyB) || '';
        return decodedA.localeCompare(decodedB);
      });
    }

    return ngrams;
  });

  // const _topNgrams = createMemo(() => sortedNgrams().slice(0, 20));

  const ablatedNgrams = createMemo(() => {
    return props.stepData.ablated_ngrams || [];
  });

  const newlyFoundNgrams = createMemo(() => {
    return props.stepData.newly_found_ngrams || [];
  });

  // Helper to check if two n-grams are equal
  // const _ngramsEqual = (a: number[], b: number[]) => {
  //   if (a.length !== b.length) return false;
  //   return a.every((val, idx) => val === b[idx]);
  // };

  return (
    <div class="space-y-6 pt-6 border-t border-gray-300 dark:border-gray-600">
      <h3 class="text-xl font-semibold">N-gram Analysis</h3>

      {/* Collapsible sections for newly found and ablated n-grams */}
      <div class="space-y-3">
        <CollapsibleNgramSection
          title="Newly Found N-grams"
          count={newlyFoundNgrams().length}
          ngrams={newlyFoundNgrams()}
          modelName={props.modelName}
          defaultExpanded={false}
          colorClass="bg-green-50 dark:bg-green-900/20 hover:bg-green-100 dark:hover:bg-green-900/30"
        />
        <CollapsibleNgramSection
          title="Total Ablated N-grams"
          count={ablatedNgrams().length}
          ngrams={ablatedNgrams()}
          modelName={props.modelName}
          defaultExpanded={false}
          colorClass="bg-red-50 dark:bg-red-900/20 hover:bg-red-100 dark:hover:bg-red-900/30"
        />
      </div>

      {/* Summary stats */}
      <div class="bg-white dark:bg-gray-800 shadow p-6">
        <div class="space-y-2 text-sm">
          <div>
            <span class="font-bold">Total Ablated:</span> {ablatedNgrams().length}
          </div>
          <div>
            <span class="font-bold">Newly Found:</span>{' '}
            <span class="text-green-600 dark:text-green-400">{newlyFoundNgrams().length}</span>
          </div>
          <div>
            <span class="font-bold">Distinctive N-grams:</span> {sortedNgrams().length}
          </div>
          <Show when={props.stepData.distinctiveness_metrics}>
            {(metrics) => (
              <div>
                <span class="font-bold">Score Cutoff:</span> {metrics().cutoff.toFixed(3)}
              </div>
            )}
          </Show>
        </div>
      </div>

    </div>
  );
}

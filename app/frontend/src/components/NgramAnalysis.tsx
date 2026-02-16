/**
 * NgramAnalysis component - Display and analyze ablated n-grams.
 */

import { createSignal, createResource, For, Show, createMemo } from 'solid-js';
import { api } from '../api/client';
import { decodeTokenIds } from '../utils/tokenizer';

export function NgramAnalysis(props: { runName: string }) {
 const [selectedStep, setSelectedStep] = createSignal<number>(0);
 const [searchQuery, setSearchQuery] = createSignal('');
 const [sortBy, setSortBy] = createSignal<'score' | 'ngram'>('score');

 const [stepList] = createResource(() => api.listSteps(props.runName));
 const [stepData] = createResource(
  () => selectedStep(),
  (stepNum) => api.getStep(props.runName, stepNum)
 );

 // Decode n-grams to human-readable text
 const [decodedNgrams, setDecodedNgrams] = createSignal<Map<string, string>>(new Map());

 // Decode all n-grams when step data changes
 createMemo(async () => {
  const step = stepData();
  if (!step || !step.distinctiveness_results) return;

  const decoded = new Map<string, string>();
  for (const result of step.distinctiveness_results) {
   const key = JSON.stringify(result.ngram);
   try {
    const text = await decodeTokenIds(result.ngram);
    decoded.set(key, text);
   } catch (e) {
    decoded.set(key, `[${result.ngram.join(', ')}]`);
   }
  }
  setDecodedNgrams(decoded);
 });

 const sortedNgrams = createMemo(() => {
  const step = stepData();
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

 const topNgrams = createMemo(() => sortedNgrams().slice(0, 20));

 const ablatedNgrams = createMemo(() => {
  const step = stepData();
  if (!step) return [];
  return step.ablated_ngrams || [];
 });

 const newlyFoundNgrams = createMemo(() => {
  const step = stepData();
  if (!step) return [];
  return step.newly_found_ngrams || [];
 });

 // Helper to check if two n-grams are equal
 const ngramsEqual = (a: number[], b: number[]) => {
  if (a.length !== b.length) return false;
  return a.every((val, idx) => val === b[idx]);
 };

 return (
  <div class="space-y-6">
   <h2 class="text-2xl font-bold">N-gram Analysis</h2>

   {/* Controls */}
   <div class="flex flex-wrap gap-4 items-center">
    <select
     value={selectedStep()}
     onChange={(e) => setSelectedStep(Number(e.currentTarget.value))}
     class="px-3 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800"
    >
     <Show when={stepList()}>
      <For each={stepList()}>
       {(step) => (
        <option value={step.step}>Step {String(step.step).padStart(3, '0')}</option>
       )}
      </For>
     </Show>
    </select>

    <input
     type="text"
     placeholder="Search n-grams..."
     value={searchQuery()}
     onInput={(e) => setSearchQuery(e.currentTarget.value)}
     class="px-3 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 flex-1 min-w-[200px]"
    />

    <select
     value={sortBy()}
     onChange={(e) => setSortBy(e.currentTarget.value as any)}
     class="px-3 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800"
    >
     <option value="score">Sort by Score</option>
     <option value="ngram">Sort Alphabetically</option>
    </select>
   </div>

   {/* Loading state */}
   <Show when={stepData.loading}>
    <div class="bg-white dark:bg-gray-800 shadow p-6">
     <div class="flex items-center justify-center space-x-3">
      <div class="animate-spin h-6 w-6 border-b-2 border-blue-600"></div>
      <span class="text-gray-600 dark:text-gray-400">Loading n-gram data...</span>
     </div>
    </div>
   </Show>

   {/* Error state */}
   <Show when={stepData.error}>
    <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-6">
     <div class="text-red-600 dark:text-red-400">Error: {stepData.error.message}</div>
    </div>
   </Show>

   {/* Summary stats */}
   <Show when={!stepData.loading && !stepData.error}>
   <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="bg-white dark:bg-gray-800 shadow p-4">
     <div class="text-sm text-gray-600 dark:text-gray-400">Total Ablated</div>
     <div class="text-2xl font-bold">{ablatedNgrams().length}</div>
    </div>
    <div class="bg-white dark:bg-gray-800 shadow p-4">
     <div class="text-sm text-gray-600 dark:text-gray-400">Newly Found</div>
     <div class="text-2xl font-bold text-green-600">{newlyFoundNgrams().length}</div>
    </div>
    <div class="bg-white dark:bg-gray-800 shadow p-4">
     <div class="text-sm text-gray-600 dark:text-gray-400">Distinctive N-grams</div>
     <div class="text-2xl font-bold">{sortedNgrams().length}</div>
    </div>
    <Show when={stepData()?.distinctiveness_metrics}>
     {(metrics) => (
      <div class="bg-white dark:bg-gray-800 shadow p-4">
       <div class="text-sm text-gray-600 dark:text-gray-400">Score Cutoff</div>
       <div class="text-2xl font-bold">{metrics().cutoff.toFixed(3)}</div>
      </div>
     )}
    </Show>
   </div>

   {/* Top N-grams table */}
   <div class="bg-white dark:bg-gray-800 shadow">
    <div class="p-4 border-b border-gray-200 dark:border-gray-700">
     <h3 class="font-semibold">Top 20 Distinctive N-grams</h3>
     <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
      Showing {Math.min(20, sortedNgrams().length)} of {sortedNgrams().length} n-grams
     </p>
    </div>
    <div class="overflow-x-auto">
     <table class="w-full">
      <thead class="bg-gray-50 dark:bg-gray-700">
       <tr>
        <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
         Rank
        </th>
        <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
         N-gram
        </th>
        <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
         Score
        </th>
        <th class="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
         Status
        </th>
       </tr>
      </thead>
      <tbody class="divide-y divide-gray-200 dark:divide-gray-700">
       <For each={topNgrams()}>
        {(ngram, idx) => {
         const isAblated = ablatedNgrams().some((ablated) => ngramsEqual(ablated, ngram.ngram));
         const isNew = newlyFoundNgrams().some((newly) => ngramsEqual(newly, ngram.ngram));
         const ngramKey = JSON.stringify(ngram.ngram);
         const decodedText = decodedNgrams().get(ngramKey) || '[loading...]';
         const tokenIds = `[${ngram.ngram.join(', ')}]`;
         return (
          <tr class="hover:bg-gray-50 dark:hover:bg-gray-750">
           <td class="px-4 py-3 text-sm text-gray-500">{idx() + 1}</td>
           <td class="px-4 py-3 text-sm" title={tokenIds}>
            <span class="font-medium">{decodedText}</span>
            <Show when={isNew}>
             <span class="ml-2 px-2 py-0.5 text-xs bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 ">
              NEW
             </span>
            </Show>
           </td>
           <td class="px-4 py-3 text-sm text-right font-medium">
            {ngram.score.toFixed(3)}
           </td>
           <td class="px-4 py-3 text-center">
            <span
             class={`px-2 py-1 text-xs ${
              isAblated
               ? 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200'
               : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
             }`}
            >
             {isAblated ? 'Ablated' : 'Active'}
            </span>
           </td>
          </tr>
         );
        }}
       </For>
      </tbody>
     </table>
    </div>
   </div>

   <Show when={sortedNgrams().length === 0 && !stepData.loading}>
    <div class="text-center text-gray-500 py-8">
     {searchQuery() ? 'No n-grams match your search' : 'No n-gram data available'}
    </div>
   </Show>
   </Show>
  </div>
 );
}

/**
 * RisingTokens component - Track how token distinctiveness ranks shift across ablation steps.
 *
 * Shows tokens that were NOT in the initial top-K at step 0 but rose in rank as ablation
 * progressed — answering the question: "do other tokens compensate when we mask the top ones?"
 */

import { createMemo, createResource, For, Show } from 'solid-js';
import { decodeNgramsForModel } from '../utils/tokenizer';
import type { Step } from '../types';

interface RisingTokensProps {
  allStepsData: Step[];
  modelName: string;
  topK?: number;      // how many tokens were being ablated per step (default 10)
  showTop?: number;   // how many rising tokens to surface (default 15)
}

interface TokenTrajectory {
  ngram: number[];
  rankByStep: (number | null)[];  // rank at each step (null if not seen)
  scoreByStep: (number | null)[];
  startRank: number | null;       // rank at step 0
  bestRank: number | null;        // best (lowest) rank achieved
  rankImprovement: number;        // startRank - bestRank (positive = rose)
  wasAblated: boolean;            // was this token ever in the ablated set
}

export function RisingTokens(props: RisingTokensProps) {
  const topK = () => props.topK ?? 10;
  const showTop = () => props.showTop ?? 15;

  // Build cross-step rank trajectories from allStepsData
  const trajectories = createMemo<TokenTrajectory[]>(() => {
    const steps = props.allStepsData;
    if (!steps || steps.length === 0) return [];

    // Collect all unique ngram keys across all steps
    const allNgramKeys = new Set<string>();
    for (const step of steps) {
      for (const dr of step.distinctiveness_results ?? []) {
        allNgramKeys.add(JSON.stringify(dr.ngram));
      }
    }

    // Build a map: ngramKey -> trajectory
    const map = new Map<string, TokenTrajectory>();
    for (const key of allNgramKeys) {
      const ngram: number[] = JSON.parse(key);
      map.set(key, {
        ngram,
        rankByStep: steps.map(() => null),
        scoreByStep: steps.map(() => null),
        startRank: null,
        bestRank: null,
        rankImprovement: 0,
        wasAblated: false,
      });
    }

    // Fill in rank/score for each step
    for (let si = 0; si < steps.length; si++) {
      const step = steps[si];
      // Build a set of ablated ngram keys for this step
      const ablatedKeys = new Set((step.ablated_ngrams ?? []).map((ng) => JSON.stringify(ng)));

      for (const dr of step.distinctiveness_results ?? []) {
        const key = JSON.stringify(dr.ngram);
        const traj = map.get(key);
        if (!traj) continue;
        traj.rankByStep[si] = dr.rank ?? null;
        traj.scoreByStep[si] = dr.score ?? null;
        if (ablatedKeys.has(key)) traj.wasAblated = true;
      }
    }

    // Compute derived fields and filter
    const result: TokenTrajectory[] = [];
    for (const traj of map.values()) {
      traj.startRank = traj.rankByStep[0];
      traj.bestRank = traj.rankByStep.reduce<number | null>((best, r) => {
        if (r === null) return best;
        if (best === null) return r;
        return r < best ? r : best;
      }, null);

      if (traj.startRank !== null && traj.bestRank !== null) {
        traj.rankImprovement = traj.startRank - traj.bestRank;
      }

      // Keep tokens that:
      //  1. Were NOT already in the ablated top-K at step 0 (startRank > topK)
      //  2. Rose at least a little (rankImprovement > 0)
      if (
        traj.startRank !== null &&
        traj.startRank >= topK() &&
        traj.rankImprovement > 0
      ) {
        result.push(traj);
      }
    }

    // Sort by rank improvement descending (biggest risers first)
    result.sort((a, b) => b.rankImprovement - a.rankImprovement);

    return result.slice(0, showTop());
  });

  // Decode the rising token ngrams for display
  const [decodedTokens] = createResource(
    () => trajectories().map((t) => t.ngram),
    async (ngrams) => {
      if (!ngrams || ngrams.length === 0) return [];
      try {
        return await decodeNgramsForModel(ngrams, props.modelName);
      } catch {
        return ngrams.map((ng) => ng.join(','));
      }
    }
  );

  const stepCount = () => props.allStepsData.length;

  return (
    <div class="bg-white dark:bg-gray-800 shadow p-6 space-y-4">
      <div>
        <h3 class="text-xl font-semibold">Rising Tokens</h3>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Tokens outside the initial top-{topK()} that rose in distinctiveness rank as ablation
          progressed — potential compensation candidates.
        </p>
      </div>

      <Show when={trajectories().length === 0}>
        <div class="text-sm text-gray-500 italic">
          {stepCount() < 2
            ? 'Need at least 2 ablation steps to compute token trajectories.'
            : 'No rising tokens found — the vocabulary distribution appears stable across steps.'}
        </div>
      </Show>

      <Show when={trajectories().length > 0}>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead class="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th class="px-3 py-2 text-left font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider text-xs">
                  Token
                </th>
                <th class="px-3 py-2 text-right font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider text-xs">
                  Rank↑
                </th>
                <th class="px-3 py-2 text-right font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider text-xs">
                  Step 0
                </th>
                <th class="px-3 py-2 text-right font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider text-xs">
                  Best
                </th>
                <For each={props.allStepsData}>
                  {(_, si) => (
                    <th class="px-2 py-2 text-right font-medium text-gray-400 dark:text-gray-500 text-xs">
                      S{si()}
                    </th>
                  )}
                </For>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-200 dark:divide-gray-700">
              <For each={trajectories()}>
                {(traj, ti) => {
                  const decoded = () =>
                    decodedTokens.loading
                      ? '[…]'
                      : (decodedTokens()?.[ti()] ?? traj.ngram.join(','));

                  // Colour the rank improvement badge
                  const improvementClass = () => {
                    if (traj.rankImprovement > 50) return 'bg-orange-100 dark:bg-orange-900 text-orange-800 dark:text-orange-200';
                    if (traj.rankImprovement > 20) return 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200';
                    return 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300';
                  };

                  return (
                    <tr class="hover:bg-gray-50 dark:hover:bg-gray-750">
                      {/* Token text */}
                      <td class="px-3 py-2 font-mono font-medium max-w-[120px] truncate" title={decoded()}>
                        {decoded()}
                      </td>
                      {/* Rank improvement */}
                      <td class="px-3 py-2 text-right">
                        <span class={`px-1.5 py-0.5 text-xs font-semibold ${improvementClass()}`}>
                          +{traj.rankImprovement}
                        </span>
                      </td>
                      {/* Start rank */}
                      <td class="px-3 py-2 text-right text-gray-500 dark:text-gray-400">
                        {traj.startRank ?? '—'}
                      </td>
                      {/* Best rank achieved */}
                      <td class="px-3 py-2 text-right font-semibold text-blue-600 dark:text-blue-400">
                        {traj.bestRank ?? '—'}
                      </td>
                      {/* Per-step rank cells */}
                      <For each={traj.rankByStep}>
                        {(rank) => (
                          <td class="px-2 py-2 text-right text-xs text-gray-400 dark:text-gray-500">
                            {rank !== null ? rank : '·'}
                          </td>
                        )}
                      </For>
                    </tr>
                  );
                }}
              </For>
            </tbody>
          </table>
        </div>

        <p class="text-xs text-gray-400 dark:text-gray-500">
          Rank↑ = how many positions the token climbed from step 0 to its best rank.
          Lower rank number = more distinctive. S0…S{stepCount() - 1} = rank at each step.
        </p>
      </Show>
    </div>
  );
}

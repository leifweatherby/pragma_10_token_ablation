/**
 * API client for pragma viewer - uses static JSON files.
 * Supports both regular and gzip-compressed JSON files.
 */

import type { Run, RunInfo, Baseline, StepListItem, Step } from '../types';

/**
 * Fetch and parse a JSON file, with automatic support for gzipped files.
 * Tries .gz version first, falls back to uncompressed if not found.
 */
async function fetchJSON<T>(url: string): Promise<T> {
  // Try gzipped version first
  const gzipUrl = url + '.gz';
  try {
    const response = await fetch(gzipUrl);
    if (response.ok) {
      // Check if the server already decompressed the content
      const contentEncoding = response.headers.get('Content-Encoding');

      if (contentEncoding === 'gzip') {
        // Server is serving compressed content, browser will decompress automatically
        return response.json();
      } else {
        // Server sent the .gz file as-is, we need to decompress it manually
        const ds = new DecompressionStream('gzip');
        const decompressedStream = response.body!.pipeThrough(ds);
        const blob = await new Response(decompressedStream).blob();
        const text = await blob.text();
        return JSON.parse(text);
      }
    }
  } catch (err) {
    // Fall through to try uncompressed version
    console.warn(`Failed to load gzipped version of ${url}:`, err);
  }

  // Fall back to uncompressed version
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}`);
  return response.json();
}

export const api = {
  async getRuns(): Promise<Run[]> {
    return fetchJSON('/data/index.json');
  },

  async getRunInfo(runName: string): Promise<RunInfo> {
    return fetchJSON(`/data/${runName}/info.json`);
  },

  async getBaseline(runName: string): Promise<Baseline> {
    return fetchJSON(`/data/${runName}/baseline.json`);
  },

  async listSteps(runName: string): Promise<StepListItem[]> {
    // Get run info to determine how many steps exist
    const runInfo = await this.getRunInfo(runName);
    const steps: StepListItem[] = [];

    for (let i = 0; i < runInfo.total_steps_completed; i++) {
      steps.push({
        step: i,
        filename: `step_${i.toString().padStart(3, '0')}.json`
      });
    }

    return steps;
  },

  async getStep(runName: string, stepNum: number): Promise<Step> {
    const filename = `step_${stepNum.toString().padStart(3, '0')}.json`;
    return fetchJSON(`/data/${runName}/${filename}`);
  },
};

/**
 * TypeScript types for pragma run data structures.
 */

export interface RunInfo {
  model: string;
  judge: string;
  num_examples: number;
  seed: number;
  cutoff: number;
  max_steps: number;
  ngram_range: [number, number];
  metric: string;
  strategy?: string;
  penalty_weight?: number;
  top_k?: number;
  total_steps_completed: number;
  total_ngrams_ablated: number;
  termination_reason: string;
  final_accuracy: number;
  final_accuracy_delta: number;
}

export interface Performance {
  is_better: boolean;
  num_correct: number;
  num_total: number;
  accuracy: number;
  accuracy_delta: number;
}

export interface Response {
  id: string;
  question: string;
  subject: string;
  choices: string[];
  choices_labels: string[];
  correct_answer: string;
  prompt: string;
  reasoning: string;
  response: string;
  reasoning_enabled: boolean;
}

export interface Verdict extends Response {
  judge: string;
  correct: boolean;
  extracted_answer: string;
  explanation: string;
  raw_response: string;
  error: string | null;
}

export interface Baseline {
  performance: Performance;
  verdicts: Verdict[];
}

export interface DistinctivenessMetrics {
  cutoff: number;
  score_min: number;
  score_max: number;
  score_mean: number;
  score_median: number;
}

export interface DistinctivenessResult {
  ngram: number[];
  score: number;
  rank?: number;
  above_cutoff?: boolean;
}

export interface Step {
  step: number;
  performance: Performance;
  verdicts: Verdict[];
  ablated_ngrams: number[][];
  newly_found_ngrams: number[][];
  distinctiveness_metrics: DistinctivenessMetrics;
  distinctiveness_results: DistinctivenessResult[];
}

export interface Run {
  name: string;
  info: RunInfo;
}

export interface StepListItem {
  step: number;
  filename: string;
}

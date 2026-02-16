/**
 * CollapsibleQuestionsSection component - Reusable accordion for displaying questions within steps.
 */

import { createSignal, createMemo, For, Show } from 'solid-js';
import { marked } from 'marked';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import type { Verdict } from '../types';

// Helper to render LaTeX in text
function renderLatex(text: string): string {
  if (!text) return '';

  // Replace display math ($$...$$)
  let result = text.replace(/\$\$([\s\S]+?)\$\$/g, (match, latex) => {
    try {
      return katex.renderToString(latex, { displayMode: true, throwOnError: false });
    } catch (e) {
      console.error('LaTeX render error:', e);
      return match;
    }
  });

  // Replace inline math ($...$)
  result = result.replace(/\$([^$\n]+?)\$/g, (match, latex) => {
    try {
      return katex.renderToString(latex, { displayMode: false, throwOnError: false });
    } catch (e) {
      console.error('LaTeX render error:', e);
      return match;
    }
  });

  return result;
}

// Helper to render text that might be raw LaTeX (without delimiters)
function renderRawLatex(text: string): string {
  if (!text) return '';

  // If already has delimiters, use regular renderLatex
  if (text.includes('$')) {
    return renderLatex(text);
  }

  // Check if text contains LaTeX commands (backslash followed by letters)
  const hasLatexCommands = /\\[a-zA-Z]+/.test(text);

  if (hasLatexCommands) {
    try {
      return katex.renderToString(text, {
        displayMode: false,
        throwOnError: false,
        strict: false
      });
    } catch (e) {
      console.error('LaTeX render error for:', text, e);
      return text;
    }
  }

  // Plain text, return as-is
  return text;
}

// Helper to handle marked.parse() which can return a Promise
function parseMarkdown(text: string): string {
  const result = marked.parse(text);
  return typeof result === 'string' ? result : '';
}

// Helper to render text with both LaTeX and Markdown
function renderTextWithLatex(text: string): string {
  if (!text) return '';
  // First render LaTeX, then pass through markdown
  const withLatex = renderLatex(text);
  return parseMarkdown(withLatex);
}

interface CollapsibleQuestionsSectionProps {
  verdicts: Verdict[];
  defaultExpanded?: boolean;
  stepName: string;
}

export function CollapsibleQuestionsSection(props: CollapsibleQuestionsSectionProps) {
  const [expanded, setExpanded] = createSignal(props.defaultExpanded ?? false);
  const [searchQuery, setSearchQuery] = createSignal('');
  const [filterCorrect, setFilterCorrect] = createSignal<'all' | 'correct' | 'incorrect'>('all');
  const [expandedQuestions, setExpandedQuestions] = createSignal<Set<string>>(new Set());

  const filteredVerdicts = createMemo(() => {
    let filtered = props.verdicts;

    // Apply search filter
    const query = searchQuery().toLowerCase();
    if (query) {
      filtered = filtered.filter(
        (v) =>
          v.id.toLowerCase().includes(query) ||
          v.question.toLowerCase().includes(query)
      );
    }

    // Apply correctness filter
    if (filterCorrect() !== 'all') {
      const shouldBeCorrect = filterCorrect() === 'correct';
      filtered = filtered.filter((v) => v.correct === shouldBeCorrect);
    }

    return filtered;
  });

  const toggleExpanded = () => {
    setExpanded(!expanded());
  };

  const toggleQuestionExpanded = (id: string) => {
    const newSet = new Set(expandedQuestions());
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setExpandedQuestions(newSet);
  };

  return (
    <div class="border border-gray-300 dark:border-gray-600">
      <button
        onClick={toggleExpanded}
        class="w-full text-left p-2 sm:p-3 transition-colors bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
      >
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="text-gray-400 text-lg">{expanded() ? '▼' : '▶'}</span>
            <span class="font-semibold text-sm sm:text-base">Questions</span>
            <span class="text-xs sm:text-sm text-gray-500">({props.verdicts.length})</span>
          </div>
        </div>
      </button>

      <Show when={expanded()}>
        <div class="border-t border-gray-300 dark:border-gray-600 p-2 sm:p-3">
          {/* Controls */}
          <div class="flex flex-wrap gap-4 items-center mb-4">
            {/* Search */}
            <input
              type="text"
              placeholder="Search questions..."
              value={searchQuery()}
              onInput={(e) => setSearchQuery(e.currentTarget.value)}
              class="px-3 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 flex-1 min-w-[200px]"
            />

            {/* Filter */}
            <select
              value={filterCorrect()}
              onChange={(e) => setFilterCorrect(e.currentTarget.value as any)}
              class="px-3 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800"
            >
              <option value="all">All</option>
              <option value="correct">Correct only</option>
              <option value="incorrect">Incorrect only</option>
            </select>

            <div class="text-sm text-gray-600 dark:text-gray-400">
              {filteredVerdicts().length} of {props.verdicts.length} questions
            </div>
          </div>

          {/* Empty state */}
          <Show when={props.verdicts.length === 0}>
            <div class="text-sm text-gray-500 italic">No questions to display</div>
          </Show>

          {/* Questions list */}
          <Show when={props.verdicts.length > 0}>
            <div class="space-y-3 max-h-96 overflow-y-auto">
              <For each={filteredVerdicts()}>
                {(verdict: Verdict) => (
                  <QuestionCard
                    verdict={verdict}
                    expanded={expandedQuestions().has(verdict.id)}
                    onToggle={() => toggleQuestionExpanded(verdict.id)}
                  />
                )}
              </For>
            </div>

            <Show when={filteredVerdicts().length === 0 && props.verdicts.length > 0}>
              <div class="text-center text-gray-500 py-4">No questions match your filters</div>
            </Show>
          </Show>
        </div>
      </Show>
    </div>
  );
}

function QuestionCard(props: { verdict: Verdict; expanded: boolean; onToggle: () => void }) {
  const { verdict } = props;
  const isCorrect = verdict.correct === true;
  const bgColor = isCorrect ? '#22c55e' : '#ef4444';
  const bgOpacity = isCorrect ? 0.1 : 0.1;
  const borderColor = isCorrect ? '#22c55e' : '#ef4444';
  const hoverOpacity = isCorrect ? 0.15 : 0.15;

  return (
    <div
      class="shadow border"
      style={{
        'background-color': `${bgColor}${Math.round(bgOpacity * 255).toString(16).padStart(2, '0')}`,
        'border-color': `${borderColor}66`
      }}
    >
      <button
        onClick={props.onToggle}
        class="w-full text-left p-4 transition-colors"
        style={{
          '--hover-bg': `${bgColor}${Math.round(hoverOpacity * 255).toString(16).padStart(2, '0')}`
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = `${bgColor}${Math.round(hoverOpacity * 255).toString(16).padStart(2, '0')}`}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
      >
        <div class="flex items-start justify-between">
          <div class="flex-1 pr-4">
            <div class="flex items-center gap-2 mb-2">
              <span class="text-sm font-mono text-gray-500">{verdict.id}</span>
              <span
                class="px-2 py-1 text-xs font-semibold text-white"
                style={{
                  'background-color': isCorrect ? '#22c55e' : '#ef4444'
                }}
              >
                {verdict.correct ? 'Correct' : 'Incorrect'}
              </span>
              <Show when={verdict.extracted_answer}>
                <span class="text-xs text-gray-500">
                  Answer: {verdict.extracted_answer}
                  {verdict.correct_answer !== verdict.extracted_answer && (
                    <span class="text-red-600"> (expected: {verdict.correct_answer})</span>
                  )}
                </span>
              </Show>
            </div>
            <div class="text-sm" innerHTML={renderLatex(verdict.question)} />
          </div>
          <div class="text-gray-400">
            {props.expanded ? '▼' : '▶'}
          </div>
        </div>
      </button>

      <Show when={props.expanded}>
        <div class="p-4 pt-0 space-y-4 border-t border-gray-200 dark:border-gray-700 mt-2">
          {/* Choices */}
          <div>
            <h4 class="font-semibold text-sm mb-2">Choices:</h4>
            <div class="space-y-1">
              <For each={verdict.choices}>
                {(choice, idx) => (
                  <div
                    class={`text-sm p-2 ${
                      verdict.choices_labels[idx()] === verdict.correct_answer
                        ? 'bg-green-50 dark:bg-green-900/20 font-medium'
                        : ''
                    }`}
                  >
                    <span innerHTML={renderRawLatex(verdict.choices_labels[idx()])} />
                    <span>: </span>
                    <span innerHTML={renderRawLatex(choice)} />
                  </div>
                )}
              </For>
            </div>
          </div>

          {/* Reasoning */}
          <Show when={verdict.reasoning}>
            <div>
              <h4 class="font-semibold text-sm mb-2">Reasoning:</h4>
              <div
                class="text-xs bg-gray-100 dark:bg-gray-900 p-3 overflow-x-auto prose prose-sm dark:prose-invert max-w-none"
                innerHTML={renderTextWithLatex(verdict.reasoning)}
              />
            </div>
          </Show>

          {/* Response */}
          <div>
            <h4 class="font-semibold text-sm mb-2">Response:</h4>
            <div
              class="text-sm bg-gray-100 dark:bg-gray-900 p-3 prose prose-sm dark:prose-invert max-w-none"
              innerHTML={renderTextWithLatex(verdict.response)}
            />
          </div>

          {/* Judge explanation */}
          <Show when={verdict.explanation}>
            <div>
              <h4 class="font-semibold text-sm mb-2">Judge Explanation:</h4>
              <div
                class="text-sm bg-blue-50 dark:bg-blue-900/20 p-3 prose prose-sm dark:prose-invert max-w-none"
                innerHTML={renderTextWithLatex(verdict.explanation)}
              />
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
}

/**
 * Verifier core: LLM-as-a-judge pairwise comparisons over `ctx.llm`, plus a
 * Probabilistic-Pivot-Tournament-style aggregation (ring pass, pivot
 * selection, pivot rounds, win-mass ranking). This is the harness-side
 * counterpart of `llm_verifier/pivot_tournament.py` in the
 * llm-as-a-verifier project.
 *
 * The judge is asked for integer scores 1-20 per candidate on every directed
 * comparison; repeated evaluations alternate the A/B prompt slots so
 * positional bias cancels. A failed call scores a tie (0.5 / 0.5), matching
 * the Python verifier's `on_error="tie"` policy.
 *
 * @module llm-verifier-multipath/verifier
 */

import { randomUUID } from 'node:crypto'
import type { Context } from '@deepseek-ai/cordis'
import { BlockAssembler, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, Message, MessageId, TokenUsage } from '@deepseek-ai/dsh-llm'
// Type-only: brings the ctx.llm service augmentation.
import type {} from '@deepseek-ai/dsh-llm'

/** Source stamp on every verifier-authored message. */
const PLUGIN_SOURCE = { kind: 'plugin', plugin: 'llm-verifier-multipath' } as const

/** Zeroed usage for calls that reported none (TokenUsage requires both totals). */
const EMPTY_USAGE: TokenUsage = { inputTokens: 0, outputTokens: 0 }

/** One directed comparison outcome: normalized [scoreA, scoreB] in [0, 1]. */
export type PairScore = readonly [number, number]

export interface VerifierCallOptions {
  /** Plugin context carrying the `llm` service. */
  ctx: Context
  /** Task description every candidate solves. */
  task: string
  /** Candidate paths (full text), index-aligned. */
  paths: readonly string[]
  /** Evaluation criteria text the judge applies. */
  criteria: string
  /** Repeated verifications per directed pair; repeats alternate A/B slots. */
  nEvaluations: number
  /** Pivot count k for the pivot tournament (O(Nk) comparisons). */
  pivots: number
  /** Provider route for verifier calls. */
  provider: string
  /** Verifier model id. */
  model: string
  /** Per-candidate character cap inside the judge prompt. */
  maxCandidateChars: number
  /** Caller-owned cancellation. */
  signal: AbortSignal
}

export interface RankedPath {
  index: number
  score: number
}

export interface ComparisonOutcome {
  /** Index of the winning candidate. */
  winner: number
  /** All candidates ranked best-first with their mean preference. */
  ranking: RankedPath[]
  /** Per-candidate mean preference (w_i / c_i); 0 for candidates never scored. */
  scores: number[]
  /** Number of directed judge comparisons performed. */
  nComparisons: number
  /** Summed token usage of every verifier call. */
  usage: TokenUsage
}

/** The pairwise judge prompt, slot-labeled like the Python verifier template. */
export function comparisonPrompt(
  task: string,
  criteria: string,
  candidateA: string,
  candidateB: string,
): string {
  return [
    'You are an expert reviewer. You will see a task description and two candidate solutions (execution paths).',
    '',
    'Evaluation Criteria:',
    criteria,
    '',
    'Task:',
    task,
    '',
    'Candidate A:',
    candidateA,
    '',
    'Candidate B:',
    candidateB,
    '',
    'Carefully analyze each candidate against the criteria, then provide your final scores:',
    '<score_A> INTEGER_1_TO_20 </score_A>',
    '<score_B> INTEGER_1_TO_20 </score_B>',
    '',
    'Rating Rules: Rate correctness on a 1-20 scale based on the evaluation criteria (1 = incorrect, 10 = borderline, 20 = correct). Output nothing but the two score tags.',
  ].join('\n')
}

/** Ring pass: ordered adjacent pairs so every candidate sits in slot A once and slot B once. */
export function ringCycle(n: number): Array<readonly [number, number]> {
  const pairs: Array<readonly [number, number]> = []
  for (let i = 0; i < n; i++) pairs.push([i, (i + 1) % n])
  return pairs
}

/** Pivot rounds: every ordered pair involving at least one pivot (both directions). */
export function pivotRoundPairs(n: number, pivots: readonly number[]): Array<readonly [number, number]> {
  const pivotSet = new Set(pivots)
  const pairs: Array<readonly [number, number]> = []
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && (pivotSet.has(i) || pivotSet.has(j))) pairs.push([i, j])
    }
  }
  return pairs
}

/** One judge call result: normalized scores (or null = tie) plus token usage. */
interface JudgeResult {
  scores: PairScore | null
  usage: TokenUsage
}

/**
 * Run the full tournament: ring pass, pivot selection by ring scores, pivot
 * rounds, then win-mass aggregation (Bradley-Terry style mean preference).
 */
export async function comparePaths(options: VerifierCallOptions): Promise<ComparisonOutcome> {
  const { paths } = options
  const n = paths.length
  if (n <= 0) throw new Error('comparePaths requires at least one candidate')
  if (n === 1) {
    return { winner: 0, ranking: [{ index: 0, score: 1 }], scores: [1], nComparisons: 0, usage: EMPTY_USAGE }
  }

  const ring = ringCycle(n)
  const ringPhase = await scorePairs(ring, options)

  const w = new Array<number>(n).fill(0)
  const c = new Array<number>(n).fill(0)
  accumulate(ring, ringPhase.scores, w, c)
  const mean = (i: number): number => (c[i] === 0 ? 0 : w[i] / c[i])

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => mean(b) - mean(a) || a - b)
  const k = Math.min(Math.max(options.pivots, 1), n - 1)
  const pivotSet = order.slice(0, k)

  const pivotPairs = pivotRoundPairs(n, pivotSet)
  const pivotPhase = await scorePairs(pivotPairs, options)
  accumulate(pivotPairs, pivotPhase.scores, w, c)

  let winner = 0
  for (let i = 1; i < n; i++) {
    if (mean(i) > mean(winner) || (mean(i) === mean(winner) && i < winner)) winner = i
  }
  const scores = Array.from({ length: n }, (_, i) => mean(i))
  const ranking = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => scores[b] - scores[a] || a - b)
    .map(index => ({ index, score: scores[index]! }))

  return {
    winner,
    ranking,
    scores,
    nComparisons: ring.length + pivotPairs.length,
    usage: addUsage(ringPhase.usage, pivotPhase.usage),
  }
}

/** Accumulate win mass and counts from scored pairs; a null outcome ties 0.5/0.5. */
function accumulate(
  pairs: readonly (readonly [number, number])[],
  scores: readonly (PairScore | null)[],
  w: number[],
  c: number[],
): void {
  for (let idx = 0; idx < pairs.length; idx++) {
    const [a, b] = pairs[idx]!
    const score = scores[idx]
    const [sa, sb] = score ?? [0.5, 0.5]
    w[a] += sa
    w[b] += sb
    c[a] += 1
    c[b] += 1
  }
}

/** Score every directed pair with nEvaluations repeats, alternating A/B slots. */
async function scorePairs(
  pairs: readonly (readonly [number, number])[],
  options: VerifierCallOptions,
): Promise<{ scores: Array<PairScore | null>; usage: TokenUsage }> {
  const jobs: Array<{ pair: readonly [number, number]; swap: boolean }> = []
  for (const pair of pairs) {
    for (let repeat = 0; repeat < options.nEvaluations; repeat++) {
      jobs.push({ pair, swap: repeat % 2 === 1 })
    }
  }
  const results = await mapLimit(jobs, 4, job => judgePair(job.pair, job.swap, options))

  const averaged: Array<PairScore | null> = []
  let usage: TokenUsage = EMPTY_USAGE
  for (let i = 0; i < pairs.length; i++) {
    const repeats = results.slice(i * options.nEvaluations, (i + 1) * options.nEvaluations)
    let a = 0
    let b = 0
    for (const repeat of repeats) {
      usage = addUsage(usage, repeat.usage)
      if (repeat.scores === null) {
        a += 0.5
        b += 0.5
      } else {
        a += repeat.scores[0]
        b += repeat.scores[1]
      }
    }
    averaged.push([a / repeats.length, b / repeats.length])
  }
  return { scores: averaged, usage }
}

/** One directed judge call over `ctx.llm`; returns normalized scores or null on failure. */
async function judgePair(
  pair: readonly [number, number],
  swap: boolean,
  options: VerifierCallOptions,
): Promise<JudgeResult> {
  const { ctx, task, paths, criteria, provider, model, maxCandidateChars, signal } = options
  const aText = truncate(paths[pair[0]]!, maxCandidateChars)
  const bText = truncate(paths[pair[1]]!, maxCandidateChars)
  // swap alternates which candidate is labeled "A" so position bias cancels
  const prompt = comparisonPrompt(task, criteria, swap ? bText : aText, swap ? aText : bText)
  const assembler = new BlockAssembler()
  try {
    for await (const chunk of ctx.llm.stream({
      provider,
      model,
      messages: [userMessage(prompt)],
      temperature: 0,
      // The judge needs its score tags in the text output: deepseek-v4-flash is
      // a reasoning model and otherwise spends the whole token budget on
      // reasoning_content, leaving content empty and the call a max-tokens tie.
      reasoningEffort: ReasoningEffortId('off'),
      maxTokens: 4096,
      signal,
    })) {
      assembler.push(chunk)
    }
  } catch {
    return { scores: null, usage: EMPTY_USAGE }
  }
  // A max-tokens cutoff still carries whatever tags were emitted before the
  // truncation; only hard failures forfeit the comparison.
  if (assembler.finish.kind !== 'stop' && assembler.finish.kind !== 'max-tokens') {
    return { scores: null, usage: assembler.usage ?? EMPTY_USAGE }
  }
  const text = textOf(assembler.message(PLUGIN_SOURCE).content)
  const scoreA = parseScore(text, 'A')
  const scoreB = parseScore(text, 'B')
  if (scoreA === null || scoreB === null) return { scores: null, usage: assembler.usage ?? EMPTY_USAGE }
  const normalized: PairScore = [scoreA / 20, scoreB / 20]
  const scores: PairScore = swap ? [normalized[1], normalized[0]] : normalized
  return { scores, usage: assembler.usage ?? EMPTY_USAGE }
}

/** Parse one `<score_X> n </score_X>` tag; clamps to [1, 20]. */
export function parseScore(text: string, slot: 'A' | 'B'): number | null {
  const match = new RegExp(`<score_${slot}>\\s*(\\d{1,3})\\s*</score_${slot}>`, 'i').exec(text)
  if (match === null) return null
  const value = Number(match[1])
  if (!Number.isFinite(value)) return null
  return Math.min(20, Math.max(1, Math.round(value)))
}

/** Join the text blocks of a content array, falling back to reasoning text. */
export function textOf(blocks: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    if ((block.type === 'text' || block.type === 'reasoning') && block.text) parts.push(block.text)
  }
  return parts.join('\n\n')
}

/** Sum two token-usage records field by field. */
function addUsage(a: TokenUsage, b: TokenUsage): TokenUsage {
  const out: TokenUsage = {
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
  }
  const cacheRead = (a.cacheReadTokens ?? 0) + (b.cacheReadTokens ?? 0)
  if (cacheRead > 0) out.cacheReadTokens = cacheRead
  const cacheWrite = (a.cacheWriteTokens ?? 0) + (b.cacheWriteTokens ?? 0)
  if (cacheWrite > 0) out.cacheWriteTokens = cacheWrite
  const reasoning = (a.reasoningTokens ?? 0) + (b.reasoningTokens ?? 0)
  if (reasoning > 0) out.reasoningTokens = reasoning
  return out
}

function truncate(text: string, limit: number): string {
  if (limit <= 0 || text.length <= limit) return text
  return text.slice(0, limit) + '\n... (truncated)'
}

function userMessage(text: string): Message {
  const content: ContentBlock[] = [{ type: 'text', text }]
  return {
    id: randomUUID() as unknown as MessageId,
    role: 'user',
    content,
    source: PLUGIN_SOURCE,
  }
}

/** Run `fn` over `items` with at most `limit` promises in flight. */
async function mapLimit<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const out = new Array<R>(items.length)
  let next = 0
  async function worker(): Promise<void> {
    while (next < items.length) {
      const index = next
      next += 1
      out[index] = await fn(items[index]!, index)
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => worker()))
  return out
}

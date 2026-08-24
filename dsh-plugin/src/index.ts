/**
 * LLM-as-a-Verifier multi-path plugin for DeepSeek Harness.
 *
 * Turns the harness into a "generate many execution paths, then self-select
 * the best one" loop. The model-facing surface:
 *
 * - a system-prompt workflow section (order 150, tool-guidance band) that
 *   tells the agent to prefer multi-path generation + verifier selection for
 *   open-ended or correctness-critical tasks;
 * - `run_paths` — spawn N independent one-shot subagents (one fresh execution
 *   path each) and register their final answers as candidates of a new run;
 * - `submit_path` — register a candidate the agent produced itself;
 * - `list_paths` — inspect the candidates of a run;
 * - `select_best_path` — pairwise LLM-as-a-judge tournament over `ctx.llm`
 *   (ring pass + pivot rounds with A/B slot alternation) that ranks the
 *   candidates and returns the winner.
 *
 * Candidate runs are kept in a process-local registry keyed by run id; the
 * registry lives for the lifetime of the runtime process.
 *
 * @module llm-verifier-multipath
 */

import { randomUUID } from 'node:crypto'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { JsonValue } from '@deepseek-ai/dsh-tools'
// Type-only imports bring the ctx.* service augmentations.
import type {} from '@deepseek-ai/dsh-agent'
import type {} from '@deepseek-ai/dsh-llm'
import type {} from '@deepseek-ai/dsh-subagent'
import type {} from '@deepseek-ai/dsh-system-prompt'
import { comparePaths, textOf } from './verifier.js'

export const name = 'llm-verifier-multipath'

export const inject = ['tools', 'llm', 'subagents', 'systemPrompt', 'agents']

export interface Config {
  /** Default number of independent candidate paths for `run_paths`. */
  nPaths?: number
  /** Default pivot count k for the selection tournament. */
  pivots?: number
  /** Default repeated verifications per directed pair. */
  nEvaluations?: number
  /** Provider route used by verifier calls. */
  provider?: string
  /** Verifier model id. */
  model?: string
  /** Default evaluation criteria text. */
  criteria?: string
  /** Per-candidate character cap in judge prompts and tool results. */
  maxCandidateChars?: number
}

export const Config: z<Config> = z.object({
  nPaths: z.number().min(1).max(16).default(3),
  pivots: z.number().min(1).default(2),
  nEvaluations: z.number().min(1).max(8).default(1),
  provider: z.string().default('deepseek-official'),
  model: z.string().default('deepseek-v4-flash'),
  criteria: z.string().default('Does the candidate solve the task correctly and completely?'),
  maxCandidateChars: z.number().min(512).default(12000),
})

const DEFAULT_CRITERIA = 'Does the candidate solve the task correctly and completely?'

/** System-prompt workflow section (tool-guidance band: order 100-199). */
const WORKFLOW_SECTION = [
  'Multi-path verification (llm-verifier-multipath): for open-ended or correctness-critical tasks, prefer generating several INDEPENDENT candidate attempts and then selecting the best one with an LLM judge:',
  '1. run_paths — spawn N independent subagent attempts for the task (or submit_path to register an attempt you produced yourself);',
  '2. list_paths — inspect the registered candidates;',
  '3. select_best_path — run the pairwise judge tournament and adopt the winner as the final answer, explaining in your reply why it won.',
].join('\n')

interface CandidatePath {
  id: string
  content: string
  source: 'subagent' | 'agent'
  stopReason?: string
}

interface PathRun {
  runId: string
  ownerSessionId: string
  task: string
  paths: CandidatePath[]
  createdAt: number
}

/** Process-local run registry. */
const runs = new Map<string, PathRun>()

export function apply(ctx: Context, config: Config): void {
  const cfg = {
    nPaths: config.nPaths ?? 3,
    pivots: config.pivots ?? 2,
    nEvaluations: config.nEvaluations ?? 1,
    provider: config.provider ?? 'deepseek-official',
    model: config.model ?? 'deepseek-v4-flash',
    criteria: (config.criteria ?? '').trim() || DEFAULT_CRITERIA,
    maxCandidateChars: config.maxCandidateChars ?? 12000,
  }

  ctx.systemPrompt.section({
    name: 'llm-verifier-multipath',
    order: 150,
    text: WORKFLOW_SECTION,
  })

  ctx.tools.register(
    defineTool({
      name: 'run_paths',
      description:
        'Spawn N independent agent attempts for a task, each in its own fresh subagent session, and register their final answers as candidate paths of a new run. Use this before select_best_path when the task is open-ended or correctness is hard to judge: several independent attempts plus verifier selection beats a single attempt.',
      parameters: {
        task: { type: 'string', required: true, description: 'The task every candidate attempt must solve.' },
        n: { type: 'number', description: 'Number of independent attempts (default 3, max 16).' },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            runId: { type: 'string', required: true, description: 'Identifier of the new run.' },
            task: { type: 'string', required: true, description: 'The task the candidates solve.' },
            paths: {
              type: 'array',
              required: true,
              description: 'One entry per attempt.',
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  id: { type: 'string', required: true, description: 'Candidate path id.' },
                  content: { type: 'string', required: true, description: 'Final answer of the attempt (truncated in this result; the full text is kept for verification).' },
                  source: { type: 'string', required: true, description: 'How the candidate was produced (subagent or agent).' },
                  stopReason: { type: 'string', description: 'How the attempt ended (completed, error, max-tokens, ...).' },
                },
              },
            },
          },
        },
        render: (_args, value) => [{ type: 'text', text: renderPaths(value) }],
      },
      async execute(args, exec) {
        const agent = exec.agent
        if (agent === undefined) throw new Error('run_paths requires an agent context')
        const n = clampInt(args.n, cfg.nPaths, 1, 16)
        const provider = pickSubagentProvider(ctx)
        const taskText = String(args.task ?? '').trim()
        if (!taskText) throw new Error('run_paths requires a non-empty task')
        const prompt: ContentBlock[] = [{ type: 'text', text: taskText }]

        const started = await Promise.allSettled(
          Array.from({ length: n }, (_, i) =>
            ctx.subagents.start(provider, {
              label: `llm-verifier-path-${i + 1}`,
              prompt,
              parent: agent,
              signal: exec.signal,
            })),
        )

        const paths: CandidatePath[] = []
        for (let i = 0; i < started.length; i++) {
          const settled = started[i]!
          if (settled.status === 'rejected') {
            paths.push({
              id: randomUUID(),
              content: `(attempt ${i + 1} failed to start: ${String(settled.reason)})`,
              source: 'subagent',
              stopReason: 'error',
            })
            continue
          }
          const run = settled.value
          try {
            const result = await run.result
            paths.push({
              id: randomUUID(),
              content: textOf(result.output) || '(attempt produced no text output)',
              source: 'subagent',
              stopReason: String(result.stopReason),
            })
          } catch (error) {
            paths.push({
              id: randomUUID(),
              content: `(attempt ${i + 1} errored: ${String(error)})`,
              source: 'subagent',
              stopReason: 'error',
            })
          } finally {
            await run.dispose()
          }
        }

        const runId = randomUUID()
        runs.set(runId, {
          runId,
          ownerSessionId: String(agent.id),
          task: taskText,
          paths,
          createdAt: Date.now(),
        })
        return {
          runId,
          task: taskText,
          paths: paths.map(capPath(cfg.maxCandidateChars)),
        }
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'submit_path',
      description:
        "Register the agent's own candidate answer as a path of a run (appending to runId when given, else creating a new run). Call this with the final answer you produced yourself, then call select_best_path to rank it against other candidates.",
      parameters: {
        content: { type: 'string', required: true, description: 'The candidate answer to register.' },
        runId: { type: 'string', description: 'Run to append to; omit to create a new run.' },
        task: { type: 'string', description: 'Task description for a newly created run.' },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            runId: { type: 'string', required: true, description: 'The run the path was added to.' },
            pathId: { type: 'string', required: true, description: 'The new candidate path id.' },
            nPaths: { type: 'number', required: true, description: 'Total candidate count of the run after this submission.' },
          },
        },
        render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
      },
      async execute(args, exec) {
        const agent = exec.agent
        if (agent === undefined) throw new Error('submit_path requires an agent context')
        const content = String(args.content ?? '').trim()
        if (!content) throw new Error('submit_path requires non-empty content')
        let run = typeof args.runId === 'string' ? runs.get(args.runId) : undefined
        if (run === undefined) {
          const runId = typeof args.runId === 'string' ? args.runId : randomUUID()
          run = {
            runId,
            ownerSessionId: String(agent.id),
            task: String(args.task ?? '(agent-submitted path)'),
            paths: [],
            createdAt: Date.now(),
          }
          runs.set(runId, run)
        }
        const path: CandidatePath = { id: randomUUID(), content, source: 'agent' }
        run.paths.push(path)
        return { runId: run.runId, pathId: path.id, nPaths: run.paths.length }
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'list_paths',
      description:
        'List the candidate paths registered in a run (ids, sources, stop reasons, and truncated content). Use it to inspect what run_paths or submit_path registered before calling select_best_path.',
      parameters: {
        runId: { type: 'string', required: true, description: 'The run to inspect.' },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            runId: { type: 'string', required: true, description: 'The inspected run.' },
            task: { type: 'string', required: true, description: 'The run task.' },
            paths: {
              type: 'array',
              required: true,
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  id: { type: 'string', required: true, description: 'Candidate path id.' },
                  source: { type: 'string', required: true, description: 'subagent or agent.' },
                  stopReason: { type: 'string', description: 'How the attempt ended.' },
                  content: { type: 'string', required: true, description: 'Truncated candidate content.' },
                },
              },
            },
          },
        },
        render: (_args, value) => [{ type: 'text', text: renderPaths(value) }],
      },
      async execute(args) {
        const run = typeof args.runId === 'string' ? runs.get(args.runId) : undefined
        if (run === undefined) throw new Error(`unknown run ${String(args.runId)}; call run_paths or submit_path first`)
        return {
          runId: run.runId,
          task: run.task,
          paths: run.paths.map(capPath(cfg.maxCandidateChars)),
        }
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'select_best_path',
      description:
        'Rank the candidate paths of a run with an LLM-as-a-judge tournament (pairwise comparisons, ring pass + pivot rounds, A/B slot alternation) and return the winner. Call it after run_paths / submit_path, then adopt the winner as the final answer.',
      parameters: {
        runId: { type: 'string', required: true, description: 'The run whose candidates to rank.' },
        criteria: { type: 'string', description: 'Evaluation criteria the judge applies; defaults to the configured criteria.' },
        nEvaluations: { type: 'number', description: 'Repeated verifications per directed pair; repeats alternate the A/B slots to cancel position bias.' },
        pivots: { type: 'number', description: 'Pivot count k for the tournament (cost O(Nk)); clamped to [1, N-1].' },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            runId: { type: 'string', required: true, description: 'The ranked run.' },
            winner: {
              type: 'object',
              required: true,
              additionalProperties: false,
              properties: {
                index: { type: 'number', required: true, description: 'Winner index in the run paths.' },
                id: { type: 'string', required: true, description: 'Winner path id.' },
              },
            },
            ranking: {
              type: 'array',
              required: true,
              description: 'All candidates best-first.',
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  index: { type: 'number', required: true, description: 'Candidate index in the run paths.' },
                  id: { type: 'string', required: true, description: 'Candidate path id.' },
                  score: { type: 'number', required: true, description: 'Mean preference in [0, 1].' },
                },
              },
            },
            nComparisons: { type: 'number', required: true, description: 'Directed judge comparisons performed.' },
            criteria: { type: 'string', required: true, description: 'The criteria used.' },
            usage: {
              type: 'object',
              required: true,
              description: 'Summed verifier token usage.',
              additionalProperties: true,
            },
          },
        },
        render: (_args, value) => [{ type: 'text', text: renderSelection(value) }],
      },
      async execute(args, exec) {
        const run = typeof args.runId === 'string' ? runs.get(args.runId) : undefined
        if (run === undefined) throw new Error(`unknown run ${String(args.runId)}; call run_paths or submit_path first`)
        const criteria = String(args.criteria ?? '').trim() || cfg.criteria
        const outcome = await comparePaths({
          ctx,
          task: run.task,
          paths: run.paths.map(p => p.content),
          criteria,
          nEvaluations: clampInt(args.nEvaluations, cfg.nEvaluations, 1, 8),
          pivots: clampInt(args.pivots, cfg.pivots, 1, Math.max(1, run.paths.length - 1)),
          provider: cfg.provider,
          model: cfg.model,
          maxCandidateChars: cfg.maxCandidateChars,
          signal: exec.signal,
        })
        const usage: Record<string, JsonValue> = {
          inputTokens: outcome.usage.inputTokens,
          outputTokens: outcome.usage.outputTokens,
        }
        if (outcome.usage.cacheReadTokens !== undefined) usage.cacheReadTokens = outcome.usage.cacheReadTokens
        if (outcome.usage.reasoningTokens !== undefined) usage.reasoningTokens = outcome.usage.reasoningTokens
        return {
          runId: run.runId,
          winner: { index: outcome.winner, id: run.paths[outcome.winner]!.id },
          ranking: outcome.ranking.map(entry => ({ index: entry.index, id: run.paths[entry.index]!.id, score: entry.score })),
          nComparisons: outcome.nComparisons,
          criteria,
          usage,
        }
      },
    }),
  )

  ctx.logger.info(
    `[llm-verifier-multipath] loaded: run_paths / submit_path / list_paths / select_best_path (provider=${cfg.provider}, model=${cfg.model})`,
  )
}

/** Prefer the in-process spawn provider, then fork, then any registered provider. */
function pickSubagentProvider(ctx: Context): string {
  for (const preferred of ['spawn', 'fork']) {
    if (ctx.subagents.getProvider(preferred) !== undefined) return preferred
  }
  const providers = ctx.subagents.list()
  if (providers.length === 0) throw new Error('no subagent provider is registered in this composition')
  return providers[0]!
}

function clampInt(value: unknown, fallback: number, min: number, max: number): number {
  const raw = typeof value === 'number' && Number.isFinite(value) ? Math.round(value) : fallback
  return Math.min(max, Math.max(min, raw))
}

function capPath(limit: number): (path: CandidatePath) => { id: string; content: string; source: string; stopReason?: string } {
  return path => ({
    id: path.id,
    content: path.content.length > limit ? path.content.slice(0, limit) + '\n... (truncated)' : path.content,
    source: path.source,
    ...(path.stopReason === undefined ? {} : { stopReason: path.stopReason }),
  })
}

/** Human-readable render of a run_paths / list_paths value. */
function renderPaths(value: JsonValue): string {
  const record = value as { runId?: unknown; task?: unknown; paths?: Array<{ id?: unknown; content?: unknown; stopReason?: unknown }> }
  const lines = [`run: ${String(record.runId ?? '?')}`, `task: ${String(record.task ?? '')}`, 'paths:']
  for (const [i, path] of (record.paths ?? []).entries()) {
    lines.push(`[${i}] id=${String(path.id ?? '?')}${path.stopReason === undefined ? '' : ` stopReason=${String(path.stopReason)}`}`)
    lines.push(String(path.content ?? ''))
  }
  return lines.join('\n')
}

/** Human-readable render of a select_best_path value. */
function renderSelection(value: JsonValue): string {
  const record = value as {
    runId?: unknown
    winner?: { index?: unknown; id?: unknown }
    ranking?: Array<{ index?: unknown; id?: unknown; score?: unknown }>
    nComparisons?: unknown
    criteria?: unknown
  }
  const lines = [
    `run: ${String(record.runId ?? '?')}`,
    `winner: path #${String(record.winner?.index ?? '?')} (${String(record.winner?.id ?? '?')})`,
    'ranking (best first):',
  ]
  for (const entry of record.ranking ?? []) {
    lines.push(`  #${String(entry.index)} score=${Number(entry.score).toFixed(4)} id=${String(entry.id)}`)
  }
  lines.push(`comparisons: ${String(record.nComparisons ?? 0)}`)
  lines.push(`criteria: ${String(record.criteria ?? '')}`)
  return lines.join('\n')
}

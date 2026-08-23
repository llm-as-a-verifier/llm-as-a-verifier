// Repro: compile the run_paths output schema with the plugin's dsh-tools copy
// and validate a sample value, to see whether `source` survives compilation.
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)

const pluginTools = require('C:/Users/cc/ws/llm-as-a-verifier/dsh-plugin/.dev-store/.pnpm/@deepseek-ai+dsh-tools@0.0._cabb709635092bb2ecfb9ec252624f49/node_modules/@deepseek-ai/dsh-tools/package.json')
console.log('plugin dsh-tools:', pluginTools.version)

const { defineTool } = await import('file:///C:/Users/cc/ws/llm-as-a-verifier/dsh-plugin/.dev-store/.pnpm/@deepseek-ai+dsh-tools@0.0._cabb709635092bb2ecfb9ec252624f49/node_modules/@deepseek-ai/dsh-tools/lib/index.js')

const tool = defineTool({
  name: 'run_paths_repro',
  description: 'repro',
  parameters: {
    task: { type: 'string', required: true, description: 'task' },
    n: { type: 'number', description: 'n' },
  },
  output: {
    schema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        runId: { type: 'string', required: true, description: 'run id' },
        task: { type: 'string', required: true, description: 'task' },
        paths: {
          type: 'array',
          required: true,
          description: 'paths',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              id: { type: 'string', required: true, description: 'id' },
              content: { type: 'string', required: true, description: 'content' },
              source: { type: 'string', required: true, description: 'source' },
              stopReason: { type: 'string', description: 'stop' },
            },
          },
        },
      },
    },
    render: () => [{ type: 'text', text: 'ok' }],
  },
  async execute() {
    return {}
  },
})

console.log(JSON.stringify(tool.output.schema, null, 2))

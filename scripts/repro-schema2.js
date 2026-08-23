// Repro 2: compile + validate with the DSH-installed dsh-tools copy.
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)

const mainTools = require('C:/Users/cc/AppData/Roaming/npm/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai/dsh-tools/package.json')
console.log('main dsh-tools:', mainTools.version)

const dshTools = await import('file:///C:/Users/cc/AppData/Roaming/npm/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai/dsh-tools/lib/index.js')
const { defineTool, validateJsonSchemaValue } = dshTools

const tool = defineTool({
  name: 'run_paths_repro',
  description: 'repro',
  parameters: {
    task: { type: 'string', required: true, description: 'task' },
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

console.log('compiled schema paths items properties:', Object.keys(tool.output.schema.properties.paths.items.properties).join(', '))

const sample = {
  runId: 'r1',
  task: 't',
  paths: [
    { id: 'a', content: 'x', source: 'subagent', stopReason: 'completed' },
    { id: 'b', content: 'y', source: 'subagent', stopReason: 'completed' },
    { id: 'c', content: 'z', source: 'subagent' },
  ],
}
const violations = validateJsonSchemaValue(tool.output.schema, sample, 'value')
console.log('violations:', JSON.stringify(violations))

// Repro: call the DeepSeek API exactly like the plugin's judgePair does
// and inspect the RAW response (why parseScore might fail).
import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'

const yaml = readFileSync(`${homedir()}/.dsh/.credentials.yaml`, 'utf8')
const line = yaml.split('\n').find(l => /^\s*DEEPSEEK_API_KEY\s*:/.test(l))
if (!line) throw new Error('no DEEPSEEK_API_KEY in credentials')
const key = line.replace(/^\s*DEEPSEEK_API_KEY\s*:\s*/, '').replace(/^["']|["']$/g, '').trim()

const prompt = [
  'You are an expert reviewer. You will see a task description and two candidate solutions (execution paths).',
  '',
  'Evaluation Criteria:',
  'Does the candidate solve the task correctly and completely?',
  '',
  'Task:',
  '实现一个 Python 函数 reverse_string(s: str) -> str，正确处理空字符串和 Unicode 字符。请给出完整、可直接运行的代码实现，并简要说明你的实现思路。',
  '',
  'Candidate A:',
  'def reverse_string(s): return s[::-1]  # simple code-point reversal',
  '',
  'Candidate B:',
  'def reverse_string(s):\n    clusters = []\n    for ch in s:\n        clusters.append(ch)\n    return "".join(reversed(clusters))',
  '',
  'Carefully analyze each candidate against the criteria, then provide your final scores:',
  '<score_A> INTEGER_1_TO_20 </score_A>',
  '<score_B> INTEGER_1_TO_20 </score_B>',
  '',
  'Rating Rules: Rate correctness on a 1-20 scale based on the evaluation criteria (1 = incorrect, 10 = borderline, 20 = correct). Output nothing but the two score tags.',
].join('\n')

const body = {
  model: 'deepseek-v4-flash',
  messages: [{ role: 'user', content: prompt }],
  temperature: 0,
  max_tokens: 4096,
  stream: false,
  thinking: { type: 'disabled' },
}

console.log('POST https://api.deepseek.com/chat/completions')
const res = await fetch('https://api.deepseek.com/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
  body: JSON.stringify(body),
})
console.log('HTTP status:', res.status)
const text = await res.text()
if (!res.ok) {
  console.log('error body:', text.slice(0, 2000))
  process.exit(0)
}
const json = JSON.parse(text)
const msg = json.choices?.[0]?.message ?? {}
console.log('finish_reason:', json.choices?.[0]?.finish_reason)
console.log('--- content ---')
console.log(JSON.stringify(msg.content))
console.log('--- reasoning_content ---')
console.log(JSON.stringify(msg.reasoning_content ?? null))

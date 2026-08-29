module.exports = {
  apps: [{
    name: 'orchestrator',
    script: 'venv/bin/python',
    args: 'three_brain_ai/cli.py serve --host 0.0.0.0 --port 8005',
    env: {
      MINISTRAL_BASE_URL: 'https://wider-radio-ruled-apartments.trycloudflare.com/v1',
      DEEPSEEK_BASE_URL: 'https://constitutional-balance-produce-unexpected.trycloudflare.com/v1',
      OPENROUTER_API_KEY: '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',
      STRATEGIST_MODEL: 'nvidia/nemotron-3-ultra-550b-a55b:free'
    },
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
  }]
}

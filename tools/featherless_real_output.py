import json
import os
from pathlib import Path

import httpx


def load_env(path: Path) -> dict[str, str]:
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding='utf-8'))


def normalize_model(model: str) -> str:
    if model.startswith('featherless/'):
        return model[len('featherless/'):]
    return model


def main():
    cwd = Path.cwd()
    project_env = cwd / '.env'
    local_env = load_env(project_env)
    os.environ.update(local_env)

    user_env_path = Path.home() / '.nanobot' / 'config.json'
    config = load_config(user_env_path) if user_env_path.exists() else {}

    models = []
    if config:
        models = config.get('agents', {}).get('defaults', {}).get('models', [])
        if not models:
            default_model = config.get('agents', {}).get('defaults', {}).get('model')
            if default_model:
                models = [default_model]

    if not models:
        models = [os.getenv('FEATHERLESS_MODEL', 'featherless/Qwen/Qwen2.5-Coder-32B-Instruct')]

    api_base = os.getenv('FEATHERLESS_API_BASE', 'https://api.featherless.ai/v1')
    api_key = os.getenv('FEATHERLESS_API_KEY') or (
        config.get('providers', {}).get('featherless', {}).get('apiKey') if config else None
    )

    if not api_key:
        raise ValueError('FEATHERLESS_API_KEY is not set in .env or config')

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    for model in models:
        bare_model = normalize_model(model)
        print(f"\n=== Testing model: {model} (normalized: {bare_model}) ===")

        payload = {
            'model': bare_model,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello from nanobot test. Please respond concisely.'},
            ],
            'temperature': 0.0,
            'max_tokens': 60,
        }

        try:
            r = httpx.post(f'{api_base}/chat/completions', headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            choice = data.get('choices', [{}])[0]
            text = choice.get('message', {}).get('content') or choice.get('text')
            print('Reply:', text)
            print('finish_reason:', choice.get('finish_reason'))
            print('usage:', data.get('usage'))

        except httpx.HTTPStatusError as e:
            print('HTTPStatusError:', e.response.status_code, e.response.text)
        except Exception as e:
            print('ERROR', type(e).__name__, e)


if __name__ == '__main__':
    main()

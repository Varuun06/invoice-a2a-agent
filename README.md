# A2A invoice agent (GA3 Q2)

    python -m pip install -r requirements.txt
    python test_local.py            # protocol/lifecycle tests with a stubbed model

Run (env vars; no secrets in the code):

    PUBLIC_BASE_URL=https://<your-host>/a2a/   # exact URL you submit, trailing slash included
    LLM_API_KEY=...                            # any OpenAI-compatible provider
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai   # default (Gemini free tier)
    LLM_MODEL=gemini-3.5-flash
    python -m uvicorn app:app --host 0.0.0.0 --port 8000     # keep ONE worker (in-process locks)

Expose it over public HTTPS with no redirects, e.g. `cloudflared tunnel --url http://localhost:8000`
(then set PUBLIC_BASE_URL to `https://<random>.trycloudflare.com/a2a/` and restart before submitting).

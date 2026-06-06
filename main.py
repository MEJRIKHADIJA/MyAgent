from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent import Agent, MODEL
from logger import get_recent_logs, log_error


agent = Agent()

app = FastAPI(
    title="MyAgent",
    description="AI agent with calculator, DuckDuckGo search, Groq LLM, and fallback routing.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    answer:          str
    route:           str
    tool_attempted:  str
    fallback_reason: str | None = None
    confidence:      str


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log_error(str(request.url), reason=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "route": "ERROR", "answer": None},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    log_error(str(request.url), reason=str(exc.detail))
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail), "route": "ERROR", "answer": None},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    log_error(str(request.url), reason=str(exc))
    return JSONResponse(
        status_code=422,
        content={"error": str(exc), "route": "ERROR", "answer": None},
    )


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest) -> QueryResponse:
    result = agent.answer(request.query)
    return QueryResponse(**result)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "tools":  ["calculator", "search (duckduckgo)"],
        "llm":    f"groq/{MODEL}",
    }


@app.get("/logs")
async def logs() -> list[dict[str, str]]:
    return get_recent_logs(20)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="g" x1="8" y1="8" x2="56" y2="56">
      <stop stop-color="#67e8f9"/>
      <stop offset=".5" stop-color="#a78bfa"/>
      <stop offset="1" stop-color="#f9a8d4"/>
    </linearGradient>
  </defs>
  <rect width="64" height="64" rx="18" fill="#090b18"/>
  <path d="M19 43V21h6l7 13 7-13h6v22h-6V32l-5 9h-4l-5-9v11z" fill="url(#g)"/>
</svg>
    """.strip()
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MyAgent</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #090b18;
      color: #f8fbff;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      overflow-x: hidden;
      padding: 28px;
      background:
        radial-gradient(circle at 18% 12%, rgba(115, 91, 255, .42), transparent 28%),
        radial-gradient(circle at 82% 22%, rgba(0, 214, 255, .35), transparent 26%),
        radial-gradient(circle at 50% 92%, rgba(255, 99, 177, .26), transparent 28%),
        linear-gradient(135deg, #090b18 0%, #10172f 48%, #111827 100%);
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);
      background-size: 44px 44px;
      mask-image: linear-gradient(to bottom, rgba(0,0,0,.7), transparent);
    }

    main {
      width: min(900px, 100%);
      position: relative;
      padding: 34px;
      border: 1px solid rgba(255,255,255,.18);
      border-radius: 30px;
      background: rgba(12, 18, 38, .72);
      box-shadow: 0 24px 90px rgba(0,0,0,.42);
      backdrop-filter: blur(18px);
    }

    .hero {
      display: grid;
      gap: 12px;
      margin-bottom: 28px;
    }

    .eyebrow {
      width: fit-content;
      padding: 8px 12px;
      border: 1px solid rgba(255,255,255,.15);
      border-radius: 999px;
      color: #bae6fd;
      background: rgba(14, 165, 233, .12);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: .06em;
      text-transform: uppercase;
    }

    h1 {
      margin: 0;
      max-width: 720px;
      font-size: clamp(40px, 8vw, 78px);
      line-height: .94;
      letter-spacing: -.07em;
    }

    .gradient-text {
      background: linear-gradient(90deg, #93c5fd, #c4b5fd, #f9a8d4);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .subtitle {
      margin: 0;
      max-width: 620px;
      color: #cbd5e1;
      font-size: 17px;
      line-height: 1.6;
    }

    .panel {
      padding: 18px;
      border: 1px solid rgba(255,255,255,.14);
      border-radius: 22px;
      background: rgba(255,255,255,.08);
    }

    form {
      display: flex;
      gap: 12px;
    }

    input {
      flex: 1;
      min-width: 0;
      padding: 17px 18px;
      border: 1px solid rgba(255,255,255,.16);
      border-radius: 16px;
      outline: none;
      background: rgba(2, 6, 23, .74);
      color: #fff;
      font-size: 16px;
      transition: border-color .2s ease, box-shadow .2s ease;
    }

    input:focus {
      border-color: rgba(147, 197, 253, .8);
      box-shadow: 0 0 0 4px rgba(59, 130, 246, .18);
    }

    button {
      border: 0;
      border-radius: 16px;
      cursor: pointer;
      font-weight: 800;
      transition: transform .16s ease, opacity .16s ease, box-shadow .16s ease;
    }

    #submit {
      padding: 0 24px;
      color: #08111f;
      background: linear-gradient(135deg, #67e8f9, #a78bfa, #f9a8d4);
      box-shadow: 0 14px 35px rgba(167, 139, 250, .28);
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: .7; cursor: wait; transform: none; }

    .suggestions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }

    .suggestions button {
      padding: 9px 12px;
      border: 1px solid rgba(255,255,255,.14);
      color: #dbeafe;
      background: rgba(255,255,255,.08);
    }

    #result {
      display: none;
      margin-top: 18px;
      padding: 18px;
      border: 1px solid rgba(255,255,255,.14);
      border-radius: 20px;
      background: rgba(2, 6, 23, .58);
      animation: rise .22s ease-out;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      margin-bottom: 14px;
      padding: 7px 11px;
      border-radius: 999px;
      color: #fff;
      font-size: 12px;
      font-weight: 900;
      letter-spacing: .05em;
    }

    .CALCULATOR, .SEARCH { background: linear-gradient(135deg, #16a34a, #22c55e); }
    .FALLBACK_FROM_CALC, .FALLBACK_FROM_SEARCH { background: linear-gradient(135deg, #d97706, #f59e0b); }
    .DIRECT_LLM { background: linear-gradient(135deg, #0284c7, #6366f1); }
    .ERROR { background: linear-gradient(135deg, #dc2626, #f43f5e); }

    pre {
      margin: 0;
      color: #f8fafc;
      white-space: pre-wrap;
      font: inherit;
      line-height: 1.65;
    }

    .meta {
      margin-top: 14px;
      color: #94a3b8;
      font-size: 13px;
    }

    .footer {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 18px;
      color: #94a3b8;
      font-size: 13px;
    }

    .dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      margin-right: 7px;
      border-radius: 999px;
      background: #22c55e;
      box-shadow: 0 0 16px #22c55e;
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 640px) {
      body { padding: 16px; }
      main { padding: 22px; border-radius: 24px; }
      form { flex-direction: column; }
      #submit { min-height: 52px; }
      .footer { flex-direction: column; }
    }
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div class="eyebrow">MyAgent</div>
    <h1>Ask <span class="gradient-text">MyAgent</span> anything.</h1>

  </section>

  <section class="panel">
    <form id="ask-form">
      <input id="query" placeholder="Try: What is sqrt(144) + 5?" autocomplete="off" required>
      <button id="submit" type="submit">Ask</button>
    </form>

    <div class="suggestions" aria-label="Example prompts">
      <button type="button" data-query="What is sqrt(144) + 5?">Calculate</button>
      <button type="button" data-query="What are the news for today?">Today’s news</button>
      <button type="button" data-query="Hello how are you?">Say hello</button>
    </div>

    <section id="result" aria-live="polite">
      <span id="route" class="badge"></span>
      <pre id="answer"></pre>
      <div class="meta" id="meta"></div>
    </section>
  </section>

  <div class="footer">
    <span><span class="dot"></span>API ready at /ask</span>
    <span>Health check: /health</span>
  </div>
</main>
<script>
  const form   = document.getElementById("ask-form");
  const button = document.getElementById("submit");
  const input  = document.getElementById("query");
  const result = document.getElementById("result");
  const routeEl  = document.getElementById("route");
  const answerEl = document.getElementById("answer");
  const metaEl   = document.getElementById("meta");

  document.querySelectorAll("[data-query]").forEach((example) => {
    example.addEventListener("click", () => {
      input.value = example.dataset.query;
      input.focus();
    });
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    button.disabled = true;
    button.textContent = "Thinking...";
    routeEl.className = "badge";
    routeEl.textContent = "WORKING";
    answerEl.textContent = "Asking the agent...";
    metaEl.textContent = "";
    result.style.display = "block";

    try {
      const res  = await fetch("/ask", {
        method:  "POST",
        headers: {"Content-Type": "application/json"},
        body:    JSON.stringify({ query: input.value })
      });
      const data = await res.json();
      const routeName = data.route || "ERROR";
      routeEl.textContent = routeName;
      routeEl.classList.add(routeName);
      answerEl.textContent = data.answer || data.error || "No answer returned.";
      metaEl.textContent = [
        data.tool_attempted  ? `tool: ${data.tool_attempted}`    : "",
        data.confidence      ? `confidence: ${data.confidence}`  : "",
        data.fallback_reason ? `fallback: ${data.fallback_reason}` : "",
      ].filter(Boolean).join("  ·  ");
    } catch (err) {
      routeEl.textContent = "ERROR";
      routeEl.classList.add("ERROR");
      answerEl.textContent = String(err);
    } finally {
      button.disabled = false;
      button.textContent = "Ask";
    }
  });
</script>
</body>
</html>
    """.strip())

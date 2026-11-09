import asyncio
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal, TypedDict
from urllib.parse import quote

import httpx
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from opentelemetry import trace
from opentelemetry.sdk.trace import Status, StatusCode
from phoenix.otel import register, using_session
from pydantic import BaseModel, Field
from twilio.base.exceptions import TwilioRestException
from twilio.request_validator import RequestValidator
from twilio.rest import Client

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

POSTGRES_CONNECTION_STRING = os.getenv(
    "POSTGRES_URL",
    "postgresql+psycopg://postgres:postgres@postgres:5432/enterprise_db",
)
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "llmops-assistant")
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY", "")
PHOENIX_COLLECTOR_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6007"
)
OLLAMA_HOST_URL = os.getenv(
    "OLLAMA_HOST_URL", "http://ollama:11434"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
MATRIX_HOMESERVER_URL = os.getenv(
    "MATRIX_HOMESERVER_URL", "http://synapse:8008"
).rstrip("/")
MATRIX_ACCESS_TOKEN = os.getenv("MATRIX_ACCESS_TOKEN", "")
MATRIX_USER_ID = os.getenv("MATRIX_USER_ID", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_API_KEY_SID = os.getenv("TWILIO_API_KEY_SID", "")
TWILIO_API_KEY_SECRET = os.getenv("TWILIO_API_KEY_SECRET", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")
TWILIO_WEBHOOK_URL = os.getenv("TWILIO_WEBHOOK_URL", "")

provider = register(
    project_name=SERVICE_NAME,
    auto_instrument=True,
    batch=True,
    endpoint=PHOENIX_COLLECTOR_ENDPOINT,
    protocol="http/protobuf",
    api_key=PHOENIX_API_KEY or None,
)
tracer = trace.get_tracer(SERVICE_NAME)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)


class DocumentMetadata(BaseModel):
    source: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class RAGDocument(BaseModel):
    content: str
    metadata: DocumentMetadata


class EnterpriseState(TypedDict):
    messages: Annotated[list, add_messages]
    channel: str
    user_id: str
    intent: Literal["rag_query", "direct_chat", "live_information"]
    temperature: float
    retrieved_docs: list[RAGDocument]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)


class WhatsAppSendRequest(BaseModel):
    phone: str = Field(pattern=r"^\+[1-9]\d{7,14}$")
    message: str = Field(min_length=1, max_length=1_600)


embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    base_url=OLLAMA_HOST_URL,
)
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="corporate_documents",
    connection=POSTGRES_CONNECTION_STRING,
    use_jsonb=True,
)
retriever = vectorstore.as_retriever()


def router_node(state: EnterpriseState) -> dict:
    query = state["messages"][-1].content.lower()
    if "reembolso" in query or "política" in query:
        return {"intent": "rag_query", "temperature": 0.0}
    if any(term in query for term in ("notícia de hoje", "notícias", "agora")):
        return {"intent": "live_information", "temperature": 0.0}
    return {"intent": "direct_chat", "temperature": 0.2}


def retrieve_node(state: EnterpriseState) -> dict:
    query = state["messages"][-1].content
    raw_docs = retriever.invoke(query)
    return {
        "retrieved_docs": [
            RAGDocument(
                content=doc.page_content,
                metadata=DocumentMetadata(
                    source=doc.metadata.get("source", "PostgreSQL/pgvector"),
                    confidence_score=0.0,
                ),
            )
            for doc in raw_docs
        ]
    }


def generate_node(state: EnterpriseState) -> dict:
    query = state["messages"][-1].content
    intent = state["intent"]

    if intent == "live_information":
        response = AIMessage(
            content=(
                "Este assistente roda localmente e não consulta a internet. "
                "Não consigo confirmar notícias ou informações em tempo real."
            )
        )
        return {"messages": [response]}

    if intent == "rag_query":
        context = "\n\n".join(doc.content for doc in state["retrieved_docs"])
        prompt = (
            "Responda em português usando somente o contexto abaixo. "
            "Se ele não contiver a resposta, diga que não encontrou a informação.\n\n"
            f"Contexto:\n{context or '(nenhum documento encontrado)'}\n\n"
            f"Pergunta: {query}"
        )
    else:
        prompt = query

    llm = ChatOllama(
        base_url=OLLAMA_HOST_URL,
        model=OLLAMA_MODEL,
        temperature=state["temperature"],
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


def route_decision(state: EnterpriseState) -> str:
    return "to_retrieve" if state["intent"] == "rag_query" else "to_generate"


workflow = StateGraph(EnterpriseState)
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_decision,
    {"to_retrieve": "retrieve", "to_generate": "generate"},
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
enterprise_app = workflow.compile()


def initial_state(message: str, channel: str, user_id: str) -> EnterpriseState:
    return {
        "messages": [HumanMessage(content=message)],
        "channel": channel,
        "user_id": user_id,
        "intent": "direct_chat",
        "temperature": 0.2,
        "retrieved_docs": [],
    }


def resolve_session_id(
    channel: str,
    user_id: str,
    session_id: str | None = None,
) -> str:
    if session_id:
        return session_id[:128]
    identity = f"{channel}:{user_id}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:32]


def run_agent(
    message: str,
    channel: str,
    user_id: str,
    session_id: str | None = None,
) -> str:
    session_id = resolve_session_id(channel, user_id, session_id)
    started_at = time.perf_counter()
    with tracer.start_as_current_span("assistant.request") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("session.id", session_id)
        span.set_attribute("app.channel", channel)
        span.set_attribute("app.model", OLLAMA_MODEL)
        span.set_attribute("app.input.length", len(message))
        span.set_attribute("input.value", message)
        try:
            with using_session(session_id):
                result = enterprise_app.invoke(
                    initial_state(message, channel, user_id),
                    config={"metadata": {"session_id": session_id}},
                )
            response = str(result["messages"][-1].content)
            span.set_attribute("app.intent", result.get("intent", "unknown"))
            span.set_attribute(
                "app.rag.documents", len(result.get("retrieved_docs", []))
            )
            span.set_attribute("app.output.length", len(response))
            span.set_attribute("output.value", response)
            span.set_attribute("app.success", True)

            usage = getattr(result["messages"][-1], "usage_metadata", None)
            if usage:
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                if input_tokens is not None:
                    span.set_attribute("llm.token_count.prompt", input_tokens)
                if output_tokens is not None:
                    span.set_attribute("llm.token_count.completion", output_tokens)
            return response
        except Exception as exc:
            span.set_attribute("app.success", False)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            span.set_attribute("app.duration_ms", elapsed_ms)
            span.set_attribute("llm.request.duration", elapsed_ms / 1000)


def missing_twilio_settings() -> list[str]:
    required_settings = {
        "TWILIO_ACCOUNT_SID": TWILIO_ACCOUNT_SID,
        "TWILIO_API_KEY_SID": TWILIO_API_KEY_SID,
        "TWILIO_API_KEY_SECRET": TWILIO_API_KEY_SECRET,
        "TWILIO_AUTH_TOKEN": TWILIO_AUTH_TOKEN,
        "TWILIO_WHATSAPP_FROM": TWILIO_WHATSAPP_FROM,
        "TWILIO_WEBHOOK_URL": TWILIO_WEBHOOK_URL,
    }
    return [name for name, value in required_settings.items() if not value]


def send_whatsapp_message(user_id: str, message: str) -> str:
    required_settings = {
        "TWILIO_ACCOUNT_SID": TWILIO_ACCOUNT_SID,
        "TWILIO_API_KEY_SID": TWILIO_API_KEY_SID,
        "TWILIO_API_KEY_SECRET": TWILIO_API_KEY_SECRET,
        "TWILIO_WHATSAPP_FROM": TWILIO_WHATSAPP_FROM,
    }
    missing = [name for name, value in required_settings.items() if not value]
    if missing:
        raise RuntimeError(
            "WhatsApp sending is not configured; missing: " + ", ".join(missing)
        )

    client = Client(
        TWILIO_API_KEY_SID,
        TWILIO_API_KEY_SECRET,
        TWILIO_ACCOUNT_SID,
    )
    recipient = user_id if user_id.startswith("whatsapp:") else f"whatsapp:{user_id}"
    sender = (
        TWILIO_WHATSAPP_FROM
        if TWILIO_WHATSAPP_FROM.startswith("whatsapp:")
        else f"whatsapp:{TWILIO_WHATSAPP_FROM}"
    )
    try:
        sent_message = client.messages.create(
            from_=sender,
            to=recipient,
            body=message,
        )
        return sent_message.sid
    except TwilioRestException as exc:
        logger.error(
            "Twilio WhatsApp send failed (status=%s, code=%s)",
            exc.status,
            exc.code,
        )
        raise


def process_whatsapp_message(user_id: str, text: str) -> None:
    session_id = resolve_session_id("whatsapp", user_id)
    answer = run_agent(text, "whatsapp", user_id, session_id)
    send_whatsapp_message(user_id, answer)


async def matrix_sync_loop(client: httpx.AsyncClient) -> None:
    since: str | None = None
    sync_url = f"{MATRIX_HOMESERVER_URL}/_matrix/client/v3/sync"

    while True:
        try:
            response = await client.get(
                sync_url,
                params={"timeout": 30_000, **({"since": since} if since else {})},
                headers={"Authorization": f"Bearer {MATRIX_ACCESS_TOKEN}"},
                timeout=35.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            logger.exception("Matrix sync request failed")
            await asyncio.sleep(5)
            continue
        sync_data = response.json()
        next_batch = sync_data["next_batch"]
        if since is None:
            since = next_batch
            continue
        since = next_batch

        joined_rooms = sync_data.get("rooms", {}).get("join", {})
        for room_id, room_data in joined_rooms.items():
            events = room_data.get("timeline", {}).get("events", [])
            for event in events:
                if (
                    event.get("type") != "m.room.message"
                    or event.get("sender") == MATRIX_USER_ID
                    or event.get("content", {}).get("msgtype") != "m.text"
                ):
                    continue

                message = event["content"].get("body", "").strip()
                if not message:
                    continue

                session_id = resolve_session_id("matrix", room_id)
                answer = await asyncio.to_thread(
                    run_agent, message, "matrix", room_id, session_id
                )
                room_path = quote(room_id, safe="")
                send_url = (
                    f"{MATRIX_HOMESERVER_URL}/_matrix/client/v3/rooms/"
                    f"{room_path}/send/m.room.message/{uuid.uuid4().hex}"
                )
                try:
                    send_response = await client.put(
                        send_url,
                        headers={"Authorization": f"Bearer {MATRIX_ACCESS_TOKEN}"},
                        json={"msgtype": "m.text", "body": answer},
                    )
                    send_response.raise_for_status()
                except httpx.HTTPError:
                    logger.exception("Could not send Matrix reply to room %s", room_id)


def report_matrix_task_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "Matrix bot stopped unexpectedly",
            exc_info=(type(error), error, error.__traceback__),
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    matrix_task: asyncio.Task | None = None
    if MATRIX_ACCESS_TOKEN and MATRIX_USER_ID:
        client = httpx.AsyncClient()
        matrix_task = asyncio.create_task(matrix_sync_loop(client))
        matrix_task.add_done_callback(report_matrix_task_failure)
        logger.info("Matrix bot sync started for %s", MATRIX_USER_ID)
    else:
        client = None
        logger.info(
            "Matrix bot is disabled; configure MATRIX_USER_ID and "
            "MATRIX_ACCESS_TOKEN to enable it."
        )

    try:
        yield
    finally:
        if matrix_task is not None:
            matrix_task.cancel()
            try:
                await matrix_task
            except asyncio.CancelledError:
                pass
        if client is not None:
            await client.aclose()
        provider.shutdown()


app = FastAPI(
    title="Open-Source Local AI Assistant",
    lifespan=lifespan,
)


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> Response:
    missing = missing_twilio_settings()
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                "WhatsApp integration is not configured; missing: "
                + ", ".join(missing)
            ),
        )
    if not TWILIO_WEBHOOK_URL.startswith("https://"):
        raise HTTPException(
            status_code=503,
            detail="TWILIO_WEBHOOK_URL must use HTTPS.",
        )

    form_data = dict(await request.form())
    signature = request.headers.get("X-Twilio-Signature", "")
    if not RequestValidator(TWILIO_AUTH_TOKEN).validate(
        TWILIO_WEBHOOK_URL, form_data, signature
    ):
        raise HTTPException(status_code=403, detail="Invalid webhook signature.")

    sender = form_data.get("From", "").strip()
    text = form_data.get("Body", "").strip()
    if sender and text:
        background_tasks.add_task(process_whatsapp_message, sender, text)

    return Response(content="<Response/>", media_type="application/xml")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    request: ChatRequest,
    session_id: Annotated[
        str | None,
        Header(alias="X-Session-ID", max_length=128),
    ] = None,
) -> dict[str, str]:
    session_id = session_id or str(uuid.uuid4())
    try:
        response = await asyncio.to_thread(
            run_agent, request.message, "http", "local", session_id
        )
    except httpx.ConnectError as exc:
        logger.exception("Could not connect to Ollama while processing chat")
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is unavailable. Install/start Ollama on the Mac and "
                "download the configured model."
            ),
        ) from exc
    return {"response": response, "session_id": session_id}


@app.post("/whatsapp/send")
async def whatsapp_send(request: WhatsAppSendRequest) -> dict[str, str]:
    try:
        message_sid = await asyncio.to_thread(
            send_whatsapp_message, request.phone, request.message
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TwilioRestException as exc:
        raise HTTPException(
            status_code=502,
            detail="A Twilio não conseguiu enviar a mensagem. Confira o remetente e o número.",
        ) from exc
    return {"status": "sent", "message_sid": message_sid}


@app.websocket("/ws/voice")
async def voice_websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    connection_session_id = str(uuid.uuid4())
    try:
        while True:
            data = await websocket.receive_json()
            user_text = data.get("text", "").strip()
            if not user_text:
                await websocket.send_json(
                    {"status": "error", "message": "O campo 'text' é obrigatório."}
                )
                continue

            user_id = str(data.get("user_id", "anonymous"))
            session_id = resolve_session_id(
                "websocket",
                user_id,
                str(data.get("session_id") or connection_session_id),
            )
            response = await asyncio.to_thread(
                run_agent,
                user_text,
                "websocket",
                user_id,
                session_id,
            )
            await websocket.send_json(
                {
                    "status": "success",
                    "response_text": response,
                    "session_id": session_id,
                }
            )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

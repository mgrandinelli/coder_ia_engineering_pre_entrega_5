"""Agente ReAct con LangGraph: nodo de modelo, nodo de herramientas, ciclo y memoria persistente."""

import logging
import operator
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from demo_llm import build_demo_llm
from tools import TOOLS

load_dotenv()

log = logging.getLogger("agente")

DB_PATH = Path(__file__).parent / "checkpoints.sqlite"
RECURSION_LIMIT = 10  # techo de pasos por pregunta: corta un posible bucle y acota el costo
MAX_MENSAJES = 20  # mensajes del historial que se envían al modelo en cada llamada
TEMPERATURE = 0
API_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
THREAD_EJEMPLO = "sesion-demo"
PREGUNTA_EJEMPLO = "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?"

SYSTEM_PROMPT = (
    "Sos un asistente de atención de una tienda en línea. Respondé sobre clientes y pedidos usando "
    "solo los datos que devuelven las herramientas; no inventes datos.\n"
    "Si una herramienta devuelve un error o le falta un dato, intentá de nuevo con otro argumento u otra "
    "herramienta, o pedile al usuario la aclaración que haga falta.\n"
    "Cuando ya tengas todos los datos, respondé en español, de forma breve."
)


class AgentState(MessagesState):
    """Estado del grafo. Hereda `messages` (con su reductor add_messages) de MessagesState."""

    # Contador acumulado con el reductor operator.add: cada nodo devuelve cuánto sumar.
    llamadas_a_herramientas: Annotated[int, operator.add]


def modo_demo() -> bool:
    """True si no hay clave para el proveedor elegido: se usa el modelo guionado."""
    return not os.getenv(API_KEY_ENV.get(os.getenv("LLM_PROVIDER") or "openai", ""))


def build_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER") or "openai"
    if modo_demo():
        log.warning("Sin clave para '%s': se usa el modelo guionado (modo demostración)", provider)
        return build_demo_llm()
    # Cada integración se importa solo si se usa: alcanza con tener instalada una de las dos.
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5",
                             temperature=TEMPERATURE, max_tokens=1024)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini", temperature=TEMPERATURE)


def recortar(mensajes: list[AnyMessage]) -> list[AnyMessage]:
    """Limita el historial que ve el modelo. El estado guardado conserva la conversación completa."""
    return trim_messages(
        mensajes,
        strategy="last",
        token_counter=len,  # cuenta mensajes, no tokens
        max_tokens=MAX_MENSAJES,
        start_on="human",  # nunca empieza por un resultado de herramienta suelto
        include_system=True,
    )


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None, llm: BaseChatModel | None = None
) -> CompiledStateGraph:
    llm = llm or build_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    async def agent(state: AgentState) -> dict:
        """Nodo del modelo: decide si responde o si llama a una herramienta."""
        historial = [SystemMessage(content=SYSTEM_PROMPT), *recortar(state["messages"])]
        respuesta: AIMessage = await llm_with_tools.ainvoke(historial)
        return {"messages": [respuesta], "llamadas_a_herramientas": len(respuesta.tool_calls)}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    # tools_condition mira el último mensaje: si trae tool_calls va a "tools"; si no, a END.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")  # el resultado vuelve al modelo: este es el ciclo ReAct
    return builder.compile(checkpointer=checkpointer)


def config_de(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


async def run_turn(graph: CompiledStateGraph, pregunta: str, thread_id: str) -> list[dict]:
    """Ejecuta una pregunta dentro de un hilo y devuelve los pasos del ciclo, en orden."""
    pasos: list[dict] = [{"tipo": "usuario", "contenido": pregunta}]
    entrada = {"messages": [HumanMessage(content=pregunta)]}
    try:
        async for actualizacion in graph.astream(entrada, config_de(thread_id), stream_mode="updates"):
            for nodo, cambios in actualizacion.items():
                for mensaje in cambios.get("messages", []):
                    if nodo == "tools":
                        pasos.append({"tipo": "resultado_herramienta", "herramienta": mensaje.name,
                                      "contenido": mensaje.content})
                    elif mensaje.tool_calls:
                        pasos.extend({"tipo": "llamada_herramienta", "herramienta": t["name"], "argumentos": t["args"]}
                                     for t in mensaje.tool_calls)
                    else:
                        pasos.append({"tipo": "respuesta", "contenido": mensaje.content})
    except GraphRecursionError:
        log.error("Se alcanzó el límite de %d pasos sin llegar a una respuesta", RECURSION_LIMIT)
        pasos.append({"tipo": "error", "contenido": f"Límite de {RECURSION_LIMIT} pasos alcanzado"})
    except Exception as exc:
        # Clave inválida, límite de tasa o red: se informa y la conversación guardada no se pierde.
        log.error("Falló la llamada al modelo (%s): %s", type(exc).__name__, exc)
        pasos.append({"tipo": "error", "contenido": f"{type(exc).__name__}: {exc}"})
    return pasos


async def ask(pregunta: str = PREGUNTA_EJEMPLO, thread_id: str = THREAD_EJEMPLO, db_path: Path = DB_PATH) -> str:
    """Hace una pregunta al agente con memoria persistente en SQLite y devuelve la respuesta final."""
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        pasos = await run_turn(build_graph(checkpointer), pregunta, thread_id)
    return pasos[-1]["contenido"]


def leer_historial(thread_id: str = THREAD_EJEMPLO, db_path: Path = DB_PATH) -> list[AnyMessage]:
    """Lee de SQLite los mensajes guardados de un hilo, con una conexión nueva (SqliteSaver síncrono)."""
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        guardado = checkpointer.get(config_de(thread_id))
    return guardado["channel_values"].get("messages", []) if guardado else []

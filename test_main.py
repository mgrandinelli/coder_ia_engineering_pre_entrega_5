"""Tests de la pre-entrega 5. No usan la red ni claves: el modelo se reemplaza por un doble guionado.

Ejecutar con: pytest -v
"""

import json
import logging
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import MessagesState

import agent
import main
from demo_llm import CONVERSACION_DEMO, build_demo_llm
from tools import TOOLS, buscar_cliente, buscar_pedidos, detalle_pedido

pytestmark = pytest.mark.asyncio


def llamada(herramienta: str, numero: int = 1, **argumentos) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": herramienta, "args": argumentos, "id": f"t{numero}"}])


class ModeloEspia(BaseChatModel):
    """Devuelve las respuestas preparadas, en orden, y guarda lo que recibió en cada llamada."""

    respuestas: list[Any]
    recibido: list = []
    herramientas: list = []

    @property
    def _llm_type(self) -> str:
        return "espia"

    def bind_tools(self, tools, **kwargs):
        self.herramientas = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.recibido.append(messages)
        numero = len(self.recibido)
        guion = self.respuestas[min(numero, len(self.respuestas)) - 1]
        # Mensaje nuevo en cada llamada, como un modelo real: sin id repetido ni tool_call_id repetido.
        llamadas = [{**t, "id": f"{t['id']}-{numero}"} for t in guion.tool_calls]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=guion.content, tool_calls=llamadas))])


@pytest.fixture
def db(tmp_path):
    return tmp_path / "checkpoints.sqlite"


def tipos(pasos: list[dict]) -> list[str]:
    return [p["tipo"] for p in pasos]


# --- Herramientas ---


async def test_buscar_pedidos_devuelve_el_resumen_del_cliente():
    resultado = await buscar_pedidos.ainvoke({"cliente_id": 102})

    assert (resultado["pedidos"], resultado["total"]) == (3, 14500)
    assert resultado["pedido_ids"] == ["P-1021", "P-1032", "P-1045"]
    assert resultado["ultimo_pedido_id"] == "P-1045"


async def test_las_herramientas_devuelven_el_error_como_dato_y_no_lanzan():
    assert "No existe un cliente con id 999" in (await buscar_pedidos.ainvoke({"cliente_id": 999}))["error"]
    assert "No existe el pedido" in (await detalle_pedido.ainvoke({"pedido_id": "P-0000"}))["error"]
    assert "error" in await buscar_cliente.ainvoke({"nombre": "Nadie"})


async def test_detalle_pedido_y_buscar_cliente():
    detalle = await detalle_pedido.ainvoke({"pedido_id": "P-1045"})
    clientes = await buscar_cliente.ainvoke({"nombre": "carla"})

    assert detalle["productos"] == ["Monitor de 24 pulgadas", "Cable HDMI"]
    assert clientes == {"clientes": [{"cliente_id": 107, "nombre": "Carla Ruiz"}]}


async def test_cada_herramienta_usa_tool_con_docstring_descriptivo_y_argumentos_tipados():
    assert [t.name for t in TOOLS] == ["buscar_pedidos", "detalle_pedido", "buscar_cliente"]
    for herramienta in TOOLS:
        assert len(herramienta.description) > 200
        assert "Usar cuando" in herramienta.description
    assert buscar_pedidos.args["cliente_id"]["type"] == "integer"
    assert detalle_pedido.args["pedido_id"]["type"] == "string"


# --- Arquitectura del grafo y estado ---


async def test_el_estado_hereda_de_messages_state():
    assert MessagesState in agent.AgentState.__orig_bases__
    assert {"messages", "llamadas_a_herramientas"} <= set(agent.AgentState.__annotations__)


async def test_el_grafo_tiene_nodo_de_modelo_nodo_de_herramientas_y_ciclo():
    grafo = agent.build_graph(llm=build_demo_llm()).get_graph()
    aristas = {(a.source, a.target, a.conditional) for a in grafo.edges}

    assert {"agent", "tools"} <= set(grafo.nodes)
    assert ("__start__", "agent", False) in aristas
    assert ("tools", "agent", False) in aristas  # el ciclo
    assert ("agent", "tools", True) in aristas and ("agent", "__end__", True) in aristas  # tools_condition


async def test_el_modelo_recibe_las_herramientas_con_bind_tools():
    modelo = ModeloEspia(respuestas=[AIMessage(content="Hola")])

    agent.build_graph(llm=modelo)

    assert modelo.herramientas == TOOLS


# --- Razonamiento multi-paso y ciclo de retorno ---


async def test_razonamiento_multi_paso_llama_dos_herramientas_antes_de_responder():
    grafo = agent.build_graph(llm=build_demo_llm())

    estado = await grafo.ainvoke({"messages": [HumanMessage(content=CONVERSACION_DEMO[0])]}, agent.config_de("t"))

    usadas = [m.name for m in estado["messages"] if isinstance(m, ToolMessage)]
    assert usadas == ["buscar_pedidos", "detalle_pedido"]
    assert estado["llamadas_a_herramientas"] == 2
    assert "$14.500" in estado["messages"][-1].content


async def test_el_resultado_de_la_herramienta_vuelve_al_modelo():
    modelo = ModeloEspia(respuestas=[llamada("buscar_pedidos", cliente_id=102), AIMessage(content="Listo")])

    await agent.build_graph(llm=modelo).ainvoke({"messages": [HumanMessage(content="¿Pedidos del 102?")]},
                                               agent.config_de("t"))

    segunda_llamada = modelo.recibido[1]
    assert isinstance(segunda_llamada[-1], ToolMessage)
    assert '"total": 14500' in segunda_llamada[-1].content


async def test_ante_un_error_de_la_herramienta_el_agente_pide_aclaracion_y_reintenta(db):
    async with AsyncSqliteSaver.from_conn_string(str(db)) as checkpointer:
        grafo = agent.build_graph(checkpointer)
        for pregunta in CONVERSACION_DEMO[:2]:
            await agent.run_turn(grafo, pregunta, "hilo")
        con_error = await agent.run_turn(grafo, CONVERSACION_DEMO[2], "hilo")
        reintento = await agent.run_turn(grafo, CONVERSACION_DEMO[3], "hilo")

    assert "No existe un cliente con id 999" in con_error[2]["contenido"]
    assert con_error[-1]["tipo"] == "respuesta" and "?" in con_error[-1]["contenido"]
    assert [p["herramienta"] for p in reintento if p["tipo"] == "llamada_herramienta"] == [
        "buscar_cliente", "buscar_pedidos"]


async def test_el_limite_de_recursion_corta_un_bucle_infinito(caplog):
    siempre_llama = ModeloEspia(respuestas=[llamada("buscar_pedidos", cliente_id=102)])

    with caplog.at_level(logging.ERROR, logger="agente"):
        pasos = await agent.run_turn(agent.build_graph(llm=siempre_llama), "¿Pedidos?", "t")

    assert agent.config_de("t")["recursion_limit"] == agent.RECURSION_LIMIT == 10
    assert pasos[-1] == {"tipo": "error", "contenido": "Límite de 10 pasos alcanzado"}
    assert "límite de 10 pasos" in caplog.text


async def test_un_fallo_de_la_api_no_interrumpe_el_programa():
    class ModeloRoto(ModeloEspia):
        def _generate(self, *args, **kwargs):
            raise ConnectionError("sin red")

    pasos = await agent.run_turn(agent.build_graph(llm=ModeloRoto(respuestas=[])), "Hola", "t")

    assert pasos[-1]["tipo"] == "error" and "ConnectionError" in pasos[-1]["contenido"]


# --- Persistencia con SqliteSaver ---


async def test_con_el_mismo_thread_id_el_agente_recuerda_la_sesion(db):
    modelo = ModeloEspia(respuestas=[AIMessage(content="Anotado."), AIMessage(content="Ana.")])
    async with AsyncSqliteSaver.from_conn_string(str(db)) as checkpointer:
        await agent.run_turn(agent.build_graph(checkpointer, modelo), "Me llamo Ana.", "hilo-1")
    # Conexión nueva a la misma base, como al reiniciar el programa.
    async with AsyncSqliteSaver.from_conn_string(str(db)) as checkpointer:
        await agent.run_turn(agent.build_graph(checkpointer, modelo), "¿Cómo me llamo?", "hilo-1")

    contenidos = [m.content for m in modelo.recibido[1]]
    assert "Me llamo Ana." in contenidos and "Anotado." in contenidos


async def test_otro_thread_id_empieza_sin_memoria(db):
    modelo = ModeloEspia(respuestas=[AIMessage(content="Anotado."), AIMessage(content="No lo sé.")])
    async with AsyncSqliteSaver.from_conn_string(str(db)) as checkpointer:
        grafo = agent.build_graph(checkpointer, modelo)
        await agent.run_turn(grafo, "Me llamo Ana.", "hilo-1")
        await agent.run_turn(grafo, "¿Cómo me llamo?", "hilo-2")

    assert "Me llamo Ana." not in [m.content for m in modelo.recibido[1]]


async def test_leer_historial_recupera_los_mensajes_con_sqlite_saver(db):
    async with AsyncSqliteSaver.from_conn_string(str(db)) as checkpointer:
        await agent.run_turn(agent.build_graph(checkpointer, build_demo_llm()), CONVERSACION_DEMO[0], "hilo")

    guardados = agent.leer_historial("hilo", db)

    assert [type(m).__name__ for m in guardados] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert agent.leer_historial("otro-hilo", db) == []


async def test_recortar_limita_el_historial_que_ve_el_modelo():
    largo = [m for i in range(30) for m in (HumanMessage(content=f"p{i}"), AIMessage(content=f"r{i}"))]

    recortado = agent.recortar(largo)

    assert len(recortado) <= agent.MAX_MENSAJES < len(largo)
    assert isinstance(recortado[0], HumanMessage)
    assert recortado[-1].content == "r29"


# --- Repositorio, credenciales y traza ---


async def test_sin_clave_se_usa_el_modo_demostracion(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert agent.modo_demo()

    monkeypatch.setenv("OPENAI_API_KEY", "clave-de-prueba")
    assert not agent.modo_demo()
    assert type(agent.build_llm()).__name__ == "ChatOpenAI"


async def test_las_claves_no_estan_en_el_repositorio():
    base = agent.DB_PATH.parent
    plantilla = (base / ".env.example").read_text()
    ignorados = (base / ".gitignore").read_text().split()

    assert "OPENAI_API_KEY=\n" in plantilla and "ANTHROPIC_API_KEY=\n" in plantilla
    assert ".env" in ignorados and "*.sqlite" in ignorados


async def test_ask_devuelve_la_respuesta_final(monkeypatch, db):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    respuesta = await agent.ask(CONVERSACION_DEMO[0], "hilo", db)

    assert "3 pedidos" in respuesta


async def test_main_genera_la_traza_en_json_y_en_log(monkeypatch, db, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    traza = await main.main(db_path=db, trazas=tmp_path)

    en_disco = json.loads((tmp_path / "traza_ejemplo.json").read_text())
    registro = (tmp_path / "traza_ejemplo.log").read_text()
    assert en_disco == traza
    assert traza["recursion_limit"] == 10 and traza["llamadas_a_herramientas"] == 6
    assert tipos(traza["turnos"][0]) == [
        "usuario", "llamada_herramienta", "resultado_herramienta",
        "llamada_herramienta", "resultado_herramienta", "respuesta"]
    assert "→ El agente decide usar la herramienta: buscar_pedidos(cliente_id=102)" in registro
    assert "Memoria del hilo" in capsys.readouterr().out

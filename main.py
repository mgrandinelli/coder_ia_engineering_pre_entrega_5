"""Pre-entrega 5 - Prueba del agente: razonamiento multi-paso, error de herramienta y memoria por thread_id.

Uso: python main.py
Genera trazas/traza_ejemplo.json y trazas/traza_ejemplo.log con el ciclo ReAct de la conversación.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import (DB_PATH, RECURSION_LIMIT, AgentState, ask, build_graph, leer_historial, modo_demo,
                   run_turn)
from demo_llm import CONVERSACION_DEMO
from tools import TOOLS

__all__ = ["AgentState", "TOOLS", "ask", "build_graph", "leer_historial", "main"]

TRAZAS = Path(__file__).parent / "trazas"


def formatear(pasos: list[dict]) -> list[str]:
    """Convierte los pasos de un turno en las líneas del log del ciclo ReAct."""
    lineas = []
    for paso in pasos:
        match paso["tipo"]:
            case "usuario":
                lineas.append(f'Usuario: "{paso["contenido"]}"')
            case "llamada_herramienta":
                argumentos = ", ".join(f"{k}={v!r}" for k, v in paso["argumentos"].items())
                lineas.append(f"→ El agente decide usar la herramienta: {paso['herramienta']}({argumentos})")
            case "resultado_herramienta":
                lineas.append(f"→ La herramienta devuelve: {paso['contenido']}")
            case "respuesta":
                lineas.append("→ El agente razona: ya tiene los datos → responde.")
                lineas.append(f'Respuesta: "{paso["contenido"]}"')
            case "error":
                lineas.append(f"→ Error: {paso['contenido']}")
    return lineas


async def main(db_path: Path = DB_PATH, trazas: Path = TRAZAS) -> dict:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")
    thread_id = f"sesion-{datetime.now():%Y%m%d-%H%M%S}"
    modelo = "guionado (modo demostración, sin API key)" if modo_demo() else (
        os.getenv("LLM_PROVIDER") or "openai")

    turnos = []
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = build_graph(checkpointer)
        for pregunta in CONVERSACION_DEMO:
            pasos = await run_turn(graph, pregunta, thread_id)
            turnos.append(pasos)
            print("\n".join(formatear(pasos)), end="\n\n")

    # Memoria: los mensajes se leen de SQLite con una conexión nueva, como tras reiniciar el programa.
    guardados = leer_historial(thread_id, db_path)
    llamadas = sum(p["tipo"] == "llamada_herramienta" for pasos in turnos for p in pasos)
    print(f"Memoria del hilo '{thread_id}': {len(guardados)} mensajes guardados en {Path(db_path).name}")
    print(f"Llamadas a herramientas en la conversación: {llamadas}")

    traza = {
        "thread_id": thread_id,
        "modelo": modelo,
        "recursion_limit": RECURSION_LIMIT,
        "mensajes_guardados": len(guardados),
        "llamadas_a_herramientas": llamadas,
        "turnos": turnos,
    }
    trazas.mkdir(exist_ok=True)
    (trazas / "traza_ejemplo.json").write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    (trazas / "traza_ejemplo.log").write_text(
        f"thread_id: {thread_id}\nmodelo: {modelo}\nrecursion_limit: {RECURSION_LIMIT}\n\n"
        + "\n\n".join("\n".join(formatear(pasos)) for pasos in turnos) + "\n", encoding="utf-8")
    return traza


if __name__ == "__main__":
    asyncio.run(main())

"""Modelo guionado para el modo demostración (sin API key).

No es parte del agente: reemplaza al LLM real cuando no hay credenciales, para poder ejecutar
el grafo, las herramientas y la persistencia de punta a punta. Devuelve, en orden, las
decisiones que tomaría el modelo en la conversación de ejemplo de main.py.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


def _llamada(herramienta: str, numero: int, **argumentos) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": herramienta, "args": argumentos, "id": f"demo_{numero}"}])


# Conversación de ejemplo: cada pregunta de CONVERSACION_DEMO consume los mensajes hasta la respuesta.
CONVERSACION_DEMO = [
    "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total? ¿Qué productos tenía el último?",
    "¿Y el anterior?",
    "¿Cuántos pedidos tiene el cliente 999?",
    "Me equivoqué de número: es Carla Ruiz.",
]

GUION = [
    # Pregunta 1: dos herramientas encadenadas (razonamiento multi-paso)
    _llamada("buscar_pedidos", 1, cliente_id=102),
    _llamada("detalle_pedido", 2, pedido_id="P-1045"),
    AIMessage(content="El cliente 102 (Ana Gómez) tuvo 3 pedidos por un total de $14.500. El último, P-1045 del "
                      "19/03/2026, incluía un monitor de 24 pulgadas y un cable HDMI, y está en camino."),
    # Pregunta 2: usa el contexto guardado (ya conoce los ids de los pedidos del cliente 102)
    _llamada("detalle_pedido", 3, pedido_id="P-1032"),
    AIMessage(content="El pedido anterior, P-1032 del 03/02/2026, incluía unos auriculares con micrófono por "
                      "$3.800 y ya fue entregado."),
    # Pregunta 3: la herramienta devuelve un error y el agente pide una aclaración
    _llamada("buscar_pedidos", 4, cliente_id=999),
    AIMessage(content="No existe un cliente con el id 999. ¿Cuál es el nombre del cliente, para buscarlo por nombre?"),
    # Pregunta 4: segundo intento por otro camino
    _llamada("buscar_cliente", 5, nombre="Carla Ruiz"),
    _llamada("buscar_pedidos", 6, cliente_id=107),
    AIMessage(content="Carla Ruiz es el cliente 107 y tiene 1 pedido por un total de $2.100."),
]


class ModeloGuionado(GenericFakeChatModel):
    """Devuelve los mensajes del guion en orden. `bind_tools` no cambia nada: el guion ya trae las llamadas."""

    def bind_tools(self, tools, **kwargs):
        return self


def build_demo_llm() -> ModeloGuionado:
    return ModeloGuionado(messages=iter(GUION))

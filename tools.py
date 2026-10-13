"""Herramientas del agente. Simulan consultas a la base de datos de pedidos de una tienda.

El modelo decide cuál usar leyendo solo el nombre, los argumentos y el docstring de cada una.
Ante un dato inexistente no lanzan una excepción: devuelven un mensaje de error en texto,
para que el agente pueda corregir el argumento, probar otra herramienta o pedir una aclaración.
"""

import asyncio

from langchain_core.tools import tool

CLIENTES: dict[int, str] = {102: "Ana Gómez", 103: "Luis Pereyra", 107: "Carla Ruiz"}

PEDIDOS: dict[str, dict] = {
    "P-1021": {"cliente_id": 102, "fecha": "2026-01-14", "importe": 4200, "estado": "entregado",
               "productos": ["Teclado mecánico", "Mouse inalámbrico"]},
    "P-1032": {"cliente_id": 102, "fecha": "2026-02-03", "importe": 3800, "estado": "entregado",
               "productos": ["Auriculares con micrófono"]},
    "P-1045": {"cliente_id": 102, "fecha": "2026-03-19", "importe": 6500, "estado": "en camino",
               "productos": ["Monitor de 24 pulgadas", "Cable HDMI"]},
    "P-1038": {"cliente_id": 103, "fecha": "2026-02-20", "importe": 1900, "estado": "entregado",
               "productos": ["Webcam HD"]},
    "P-1050": {"cliente_id": 107, "fecha": "2026-03-22", "importe": 2100, "estado": "en preparación",
               "productos": ["Disco SSD de 1 TB"]},
}

LATENCIA_S = 0.05  # demora simulada de la base de datos


@tool
async def buscar_pedidos(cliente_id: int) -> dict:
    """Devuelve el resumen de pedidos de un cliente a partir de su id numérico.

    Usar cuando se pregunta cuántos pedidos tiene un cliente, cuánto gastó en total
    o cuáles son sus pedidos. Requiere el id numérico del cliente (por ejemplo 102);
    si solo se conoce el nombre, obtener antes el id con `buscar_cliente`.

    Devuelve: nombre del cliente, `pedidos` (cantidad), `total` (suma de importes en pesos),
    `pedido_ids` (ordenados del más antiguo al más reciente) y `ultimo_pedido_id`.
    No incluye los productos de cada pedido: para eso usar `detalle_pedido`.
    Si el cliente no existe, devuelve un campo `error` con la explicación.
    """
    await asyncio.sleep(LATENCIA_S)
    if cliente_id not in CLIENTES:
        return {"error": f"No existe un cliente con id {cliente_id}. Verificar el id o buscarlo por nombre con buscar_cliente."}
    ids = sorted((p for p, d in PEDIDOS.items() if d["cliente_id"] == cliente_id), key=lambda p: PEDIDOS[p]["fecha"])
    return {
        "cliente_id": cliente_id,
        "nombre": CLIENTES[cliente_id],
        "pedidos": len(ids),
        "total": sum(PEDIDOS[p]["importe"] for p in ids),
        "pedido_ids": ids,
        "ultimo_pedido_id": ids[-1] if ids else None,
    }


@tool
async def detalle_pedido(pedido_id: str) -> dict:
    """Devuelve el detalle de un pedido a partir de su id con formato P-NNNN (por ejemplo P-1045).

    Usar cuando se pregunta qué productos tenía un pedido, su fecha, su importe o su estado
    de envío. El id del pedido se obtiene de `pedido_ids` o `ultimo_pedido_id` en el
    resultado de `buscar_pedidos`; no inventar ids.

    Devuelve: `cliente_id`, `fecha` (AAAA-MM-DD), `importe` en pesos, `estado` y `productos`.
    Si el pedido no existe, devuelve un campo `error` con la explicación.
    """
    await asyncio.sleep(LATENCIA_S)
    if pedido_id not in PEDIDOS:
        return {"error": f"No existe el pedido {pedido_id}. Obtener los ids válidos con buscar_pedidos."}
    return {"pedido_id": pedido_id, **PEDIDOS[pedido_id]}


@tool
async def buscar_cliente(nombre: str) -> dict:
    """Busca clientes por nombre o apellido y devuelve su id numérico.

    Usar cuando se menciona a un cliente por su nombre (por ejemplo "Carla Ruiz") y hace falta
    su id para llamar a `buscar_pedidos`. La búsqueda no distingue mayúsculas y acepta
    una parte del nombre.

    Devuelve `clientes`: una lista de {`cliente_id`, `nombre`}. Si hay más de un resultado,
    pedir al usuario que aclare cuál es. Si no hay ninguno, devuelve un campo `error`.
    """
    await asyncio.sleep(LATENCIA_S)
    encontrados = [{"cliente_id": i, "nombre": n} for i, n in CLIENTES.items() if nombre.strip().lower() in n.lower()]
    if not encontrados:
        return {"error": f"No hay clientes cuyo nombre contenga '{nombre}'. Pedir al usuario que lo verifique."}
    return {"clientes": encontrados}


TOOLS = [buscar_pedidos, detalle_pedido, buscar_cliente]

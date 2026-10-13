# Pre-entrega 5: Agente de razonamiento cíclico con memoria persistente

Curso AI Engineering (Coderhouse), módulo 5.

Un agente ReAct construido con LangGraph. Decide por sí mismo cuándo llamar a una herramienta, vuelve a razonar con el resultado y recuerda la conversación entre ejecuciones gracias a un checkpointer en SQLite.

```python
respuesta = await ask("¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?", thread_id="sesion-1")
```

## Cómo levantar el entorno

Requiere Python 3.12 o superior.

```bash
git clone https://github.com/mgrandinelli/coder_ia_engineering_pre_entrega_5.git
cd coder_ia_engineering_pre_entrega_5

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # y completar la clave del proveedor elegido
python main.py
```

`main.py` mantiene una conversación de cuatro preguntas con un mismo `thread_id`, imprime el ciclo ReAct de cada una y guarda la traza en `trazas/`.

## Variables de entorno

| Variable | Obligatoria | Qué es |
|---|---|---|
| `LLM_PROVIDER` | No (por defecto `openai`) | Proveedor del modelo: `openai` o `anthropic` |
| `OPENAI_API_KEY` | Con OpenAI | Clave de la API de OpenAI |
| `ANTHROPIC_API_KEY` | Con Anthropic | Clave de la API de Anthropic |
| `OPENAI_MODEL` | No (por defecto `gpt-4o-mini`) | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | No (por defecto `claude-haiku-4-5`) | Modelo de Anthropic |

Las claves se leen del archivo `.env`, que está en `.gitignore` junto con la base de checkpoints. Al repositorio solo se sube `.env.example`, sin valores.

## Archivos

| Archivo | Contenido |
|---|---|
| `agent.py` | Estado, `StateGraph`, nodo del modelo, `llm.bind_tools()`, persistencia y `recursion_limit` |
| `tools.py` | Tres herramientas con `@tool` que simulan consultas a una base de datos de pedidos |
| `demo_llm.py` | Modelo guionado para ejecutar sin clave (modo demostración) |
| `main.py` | Prueba de ejecución; genera la traza |
| `trazas/traza_ejemplo.json`, `trazas/traza_ejemplo.log` | Traza del ciclo ReAct de la prueba |
| `test_main.py` | 20 tests, sin red ni claves |

## Arquitectura y decisiones de diseño

El grafo tiene dos nodos y un ciclo:

- START → `agent`
- `agent` → `tools`, si el modelo pidió una herramienta (arista condicional `tools_condition`)
- `agent` → END, si el modelo respondió sin pedir herramientas
- `tools` → `agent` (el resultado vuelve al modelo: este es el ciclo ReAct)

```python
builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=checkpointer)
```

### Estado

```python
class AgentState(MessagesState):
    llamadas_a_herramientas: Annotated[int, operator.add]
```

`AgentState` hereda de `MessagesState`, que aporta el campo `messages` con su reductor `add_messages`: cada nodo devuelve solo los mensajes nuevos y LangGraph los agrega al historial. El campo `llamadas_a_herramientas` usa el reductor `operator.add` para acumular cuántas herramientas pidió el modelo.

### Herramientas (`tools.py`)

| Herramienta | Qué hace |
|---|---|
| `buscar_pedidos(cliente_id: int)` | Resumen de pedidos de un cliente: cantidad, total e ids |
| `detalle_pedido(pedido_id: str)` | Fecha, importe, estado y productos de un pedido |
| `buscar_cliente(nombre: str)` | Busca el id de un cliente a partir de su nombre |

Cada una está definida con `@tool`, tiene los argumentos tipados y un docstring que indica cuándo usarla, qué devuelve y cómo se relaciona con las otras. Son corrutinas y simulan la demora de una base de datos con `asyncio.sleep`.

Ante un dato inexistente, las herramientas no lanzan una excepción: devuelven un campo `error` con la explicación. Así el error llega al modelo como una observación más, y el modelo puede corregir el argumento, probar otra herramienta o pedir una aclaración.

### Autonomía

El modelo se vincula a las herramientas con `llm.bind_tools(TOOLS)`. No hay rutas manuales en el grafo: la única decisión es `tools_condition`, que mira si la última respuesta del modelo trae llamadas a herramientas. Qué herramienta usar y con qué argumentos lo decide el modelo.

### Persistencia

El checkpointer guarda el estado del grafo en `checkpoints.sqlite` después de cada nodo. Cada conversación se identifica con un `thread_id`:

```python
config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
```

- Con el mismo `thread_id`, el agente retoma la conversación donde quedó, incluso después de cerrar el programa.
- Con otro `thread_id`, empieza sin memoria.

Se usan las dos variantes del checkpointer de SQLite. El agente corre en un entorno asíncrono, y para eso LangGraph provee `AsyncSqliteSaver`, la versión asíncrona de `SqliteSaver`. `SqliteSaver` se usa en `leer_historial()`, que abre una conexión nueva y lee los mensajes guardados de un hilo.

### Límites

- **`recursion_limit = 10`.** Cada pregunta puede recorrer como máximo 10 pasos del grafo. Si el modelo entra en un bucle de llamadas, la ejecución se corta, se informa en el log y el programa sigue.
- **Historial recortado.** El estado guarda la conversación completa, pero al modelo se le envían como máximo los últimos 20 mensajes (`trim_messages`), empezando siempre por un mensaje del usuario.

### Modelo y `temperature`

`ChatOpenAI` y `ChatAnthropic` se crean con `temperature=0`. El modelo por defecto de Anthropic es `claude-haiku-4-5` porque acepta ese parámetro; los modelos Claude más recientes (por ejemplo `claude-opus-5-5`) ya no lo usan y lo rechazan con un error 400.

### Modo demostración

Sin clave del proveedor, `build_llm()` devuelve un modelo guionado (`demo_llm.py`) en lugar del modelo real. Ese modelo emite, en orden, las decisiones de la conversación de ejemplo. El grafo, las herramientas, la arista condicional y la persistencia son los reales y se ejecutan de verdad; lo único simulado son las decisiones del modelo.

## Prueba de ejecución y traza

`main.py` hace cuatro preguntas en un mismo hilo:

| Pregunta | Qué demuestra |
|---|---|
| ¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total? ¿Qué productos tenía el último? | Razonamiento multi-paso: dos herramientas antes de responder |
| ¿Y el anterior? | Memoria: usa los ids de pedidos que ya conoce del turno previo |
| ¿Cuántos pedidos tiene el cliente 999? | Ciclo de retorno: la herramienta devuelve un error y el agente pide una aclaración |
| Me equivoqué de número: es Carla Ruiz. | Segundo intento por otro camino: busca por nombre y vuelve a consultar |

En total, el agente llama a herramientas 6 veces y el hilo queda con 20 mensajes guardados en SQLite.

La traza completa está en `trazas/traza_ejemplo.json` (estructurada) y en `trazas/traza_ejemplo.log`. Fue generada en modo demostración, con el modelo guionado; el campo `modelo` de la traza lo indica. Con una clave configurada, `python main.py` la vuelve a generar con el modelo real. Contenido del log:

```
thread_id: sesion-20261013-144318
modelo: guionado (modo demostración, sin API key)
recursion_limit: 10

Usuario: "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total? ¿Qué productos tenía el último?"
→ El agente decide usar la herramienta: buscar_pedidos(cliente_id=102)
→ La herramienta devuelve: {"cliente_id": 102, "nombre": "Ana Gómez", "pedidos": 3, "total": 14500, "pedido_ids": ["P-1021", "P-1032", "P-1045"], "ultimo_pedido_id": "P-1045"}
→ El agente decide usar la herramienta: detalle_pedido(pedido_id='P-1045')
→ La herramienta devuelve: {"pedido_id": "P-1045", "cliente_id": 102, "fecha": "2026-03-19", "importe": 6500, "estado": "en camino", "productos": ["Monitor de 24 pulgadas", "Cable HDMI"]}
→ El agente razona: ya tiene los datos → responde.
Respuesta: "El cliente 102 (Ana Gómez) tuvo 3 pedidos por un total de $14.500. El último, P-1045 del 19/03/2026, incluía un monitor de 24 pulgadas y un cable HDMI, y está en camino."

Usuario: "¿Y el anterior?"
→ El agente decide usar la herramienta: detalle_pedido(pedido_id='P-1032')
→ La herramienta devuelve: {"pedido_id": "P-1032", "cliente_id": 102, "fecha": "2026-02-03", "importe": 3800, "estado": "entregado", "productos": ["Auriculares con micrófono"]}
→ El agente razona: ya tiene los datos → responde.
Respuesta: "El pedido anterior, P-1032 del 03/02/2026, incluía unos auriculares con micrófono por $3.800 y ya fue entregado."

Usuario: "¿Cuántos pedidos tiene el cliente 999?"
→ El agente decide usar la herramienta: buscar_pedidos(cliente_id=999)
→ La herramienta devuelve: {"error": "No existe un cliente con id 999. Verificar el id o buscarlo por nombre con buscar_cliente."}
→ El agente razona: ya tiene los datos → responde.
Respuesta: "No existe un cliente con el id 999. ¿Cuál es el nombre del cliente, para buscarlo por nombre?"

Usuario: "Me equivoqué de número: es Carla Ruiz."
→ El agente decide usar la herramienta: buscar_cliente(nombre='Carla Ruiz')
→ La herramienta devuelve: {"clientes": [{"cliente_id": 107, "nombre": "Carla Ruiz"}]}
→ El agente decide usar la herramienta: buscar_pedidos(cliente_id=107)
→ La herramienta devuelve: {"cliente_id": 107, "nombre": "Carla Ruiz", "pedidos": 1, "total": 2100, "pedido_ids": ["P-1050"], "ultimo_pedido_id": "P-1050"}
→ El agente razona: ya tiene los datos → responde.
Respuesta: "Carla Ruiz es el cliente 107 y tiene 1 pedido por un total de $2.100."
```

## Tests

`test_main.py` tiene 20 tests. No usan la red ni claves: el modelo se reemplaza por un doble guionado y la base de checkpoints se crea en una carpeta temporal.

```bash
pytest -v
```

Cubren los cuatro criterios de la rúbrica:

- **Arquitectura del grafo y estado:** el estado hereda de `MessagesState`; el grafo tiene los nodos `agent` y `tools`, la arista condicional y el ciclo; el modelo recibe las herramientas con `bind_tools`.
- **Herramientas y razonamiento multi-paso:** cada herramienta responde bien y devuelve los errores como dato; el agente llama dos herramientas antes de responder; el resultado vuelve al modelo; ante un error pide una aclaración y reintenta; el `recursion_limit` corta un bucle.
- **Persistencia:** con el mismo `thread_id` y una conexión nueva, el modelo recibe la conversación anterior; con otro `thread_id` no; `leer_historial` recupera los mensajes con `SqliteSaver`.
- **Repositorio y credenciales:** las claves no están en el repositorio; `main.py` genera la traza en JSON y en log.

## Reporte de tests

Salida de `pytest -v`:

```
============================= test session starts ==============================
collecting ... collected 20 items

test_main.py::test_buscar_pedidos_devuelve_el_resumen_del_cliente PASSED [  5%]
test_main.py::test_las_herramientas_devuelven_el_error_como_dato_y_no_lanzan PASSED [ 10%]
test_main.py::test_detalle_pedido_y_buscar_cliente PASSED                [ 15%]
test_main.py::test_cada_herramienta_usa_tool_con_docstring_descriptivo_y_argumentos_tipados PASSED [ 20%]
test_main.py::test_el_estado_hereda_de_messages_state PASSED             [ 25%]
test_main.py::test_el_grafo_tiene_nodo_de_modelo_nodo_de_herramientas_y_ciclo PASSED [ 30%]
test_main.py::test_el_modelo_recibe_las_herramientas_con_bind_tools PASSED [ 35%]
test_main.py::test_razonamiento_multi_paso_llama_dos_herramientas_antes_de_responder PASSED [ 40%]
test_main.py::test_el_resultado_de_la_herramienta_vuelve_al_modelo PASSED [ 45%]
test_main.py::test_ante_un_error_de_la_herramienta_el_agente_pide_aclaracion_y_reintenta PASSED [ 50%]
test_main.py::test_el_limite_de_recursion_corta_un_bucle_infinito PASSED [ 55%]
test_main.py::test_un_fallo_de_la_api_no_interrumpe_el_programa PASSED   [ 60%]
test_main.py::test_con_el_mismo_thread_id_el_agente_recuerda_la_sesion PASSED [ 65%]
test_main.py::test_otro_thread_id_empieza_sin_memoria PASSED             [ 70%]
test_main.py::test_leer_historial_recupera_los_mensajes_con_sqlite_saver PASSED [ 75%]
test_main.py::test_recortar_limita_el_historial_que_ve_el_modelo PASSED  [ 80%]
test_main.py::test_sin_clave_se_usa_el_modo_demostracion PASSED          [ 85%]
test_main.py::test_las_claves_no_estan_en_el_repositorio PASSED          [ 90%]
test_main.py::test_ask_devuelve_la_respuesta_final PASSED                [ 95%]
test_main.py::test_main_genera_la_traza_en_json_y_en_log PASSED          [100%]

============================== 20 passed in 3.33s ==============================
```

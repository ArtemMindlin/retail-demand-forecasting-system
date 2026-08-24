# Capa web: dashboard Django y API operacional

Este documento describe `src/retail_forecasting/api/`: el cuadro de mandos y la
superficie JSON del sistema. Para el pipeline de modelado, ver
`docs/system_design.md`.

## Principio rector

La capa web **no calcula nada del dominio**. Renderiza y orquesta llamadas a
`forecasting/`, `inventory/`, `simulation/` y `eda/`. Si una vista necesita un
numero nuevo, el numero se calcula en `services/` o en el pipeline, nunca en la
plantilla ni en la vista.

Esto es lo que hace testeable el sistema: los servicios son funciones puras
sobre DataFrames, sin dependencia de Django, y las vistas quedan en 20-40 lineas.

## Arquitectura

```text
navegador
  |
  |  HTML completo (navegacion) o fragmento (htmx)
  v
views/           <- finas: leen query params, llaman a services, renderizan
  |
  v
services/        <- pandas puro, SIN Django. Testeable en aislamiento
  |
  v
MLflow           <- indice de que corridas existen (mlflow.db)
  |
  v
mlruns/<id>/artifacts/  <- artefactos en disco (CSV, Parquet, JSON, PNG)
```

`ArtifactStore` descubre las corridas por MLflow y las lee como ficheros del
directorio de artefactos que este le devuelve. Son ficheros normales porque el
pipeline los escribio ahi: `open_run_directory` abre la corrida y le entrega ese
directorio, y para un almacen local escribir dentro *es* registrar. Una sola copia
y ningun paso de subida.

El modo de fallo cambio con ello: un almacen inalcanzable ya no cuesta el registro,
cuesta la corrida, asi que el `try/except` que lo envolvia se ha ido. Tragarselo
reportaria exito sobre una corrida sin ficheros.

Descubrir por el indice elimina ademas una clase de fallo en lugar de defenderse de
ella: un nombre de corrida tiene que ser una clave que MLflow ya conoce, asi que no
hay ruta que recorrer ni nombre que sanear.

El plano OPS escribe en su propio experimento, `retail_forecasting_ops`, y no como
una corrida mas: sus costes no son costes de walk-forward y `docs/runs.md` dice
explicitamente que no se citen al lado. Compartir experimento los pondria bajo un
mismo `search_runs`.

### La ubicacion de artefactos, y por que esta relativa

MLflow hornea en la fila del experimento donde viven sus artefactos, y las filas de
este almacen la tienen RELATIVA (`mlruns`). Hace falta que lo este: una ruta absoluta
graba el checkout que creo el experimento, y el contenedor que monta el mismo
`mlflow.db` en `/app` buscaria los artefactos donde no estan.

**No lo consigue el codigo.** MLflow absolutiza contra el directorio de trabajo
cualquier ubicacion que se le pase, y no ofrece forma de guardar una relativa:
`mlruns`, `./mlruns` y `file:mlruns` vuelven las tres absolutas (medido). Las filas
estan relativas porque se reescribieron a mano, y eso aguanta: **una corrida hereda
la ubicacion guardada de su experimento tal cual**, asi que las corridas nuevas de
los tres experimentos ya salen relativas.

Lo que no cubre: un experimento NUEVO nace absoluto otra vez. Si se añade uno, hay
que reescribirlo igual, sobre `experiments.artifact_location` y `runs.artifact_uri`.

El precio de todo esto es que el almacen queda atado al directorio de trabajo:
lanzar el pipeline desde un subdirectorio no encuentra sus artefactos. Los comandos
documentados y el contenedor arrancan los dos desde la raiz.

`active_run.log` vive en `var/`, fuera del almacen, porque se escribe mientras la corrida va:
una corrida de MLflow es el registro cerrado de algo que ya acabo, no un sitio para
estado que cambia. Por eso `champion_registry.json` se mudo a `models_dir`, que es
estado mutable que sobrevive a las corridas que lo actualizan.

No hay base de datos de aplicacion: la unica pieza de estado propia de la web es
la sesion del operador, que viaja en una cookie firmada.

### Modulos

| Modulo | Responsabilidad |
| --- | --- |
| `settings.py` | Configuracion Django. Sin `DATABASES`, sesiones en cookie firmada |
| `urls.py` | Mapa de rutas. El `name` de cada tab coincide con `context.MODES` |
| `context.py` | Context processors: navegacion de dos niveles y parametros what-if |
| `middleware.py` | `LoginRequiredMiddleware`: exige sesion salvo lista publica |
| `store.py` | Unica costura entre Django y `services/`: construye el `ArtifactStore` |
| `charts.py` | Primitivas SVG compartidas (sparkline, histograma, grafico de forecast) |
| `eda_charts.py` | Los ocho renderizadores SVG del EDA y los del plano de analisis |
| `academic.py` | Contenido editorial de los modulos matematicos del dashboard |
| `templatetags/dashboard.py` | Set de iconos inline y filtros de formato |

### Servicios

| Servicio | Que resuelve |
| --- | --- |
| `services/runs.py` | `ArtifactStore`: descubrimiento de runs, cache de predicciones, validacion anti-traversal |
| `services/forecast.py` | Conformal empirico, Newsvendor, PSI por SKU, tabla de SKUs, drift, alertas |
| `services/ops.py` | Lectura e indexado semanal del backtest de origen rodante (rejilla no solapada; excluye semanas parciales) |
| `services/eda.py` | Catalogo de figuras y datos listos para graficar desde los CSV del EDA |
| `services/experiments.py` | Imputacion latente, ranking de calidad, frente de Pareto, coste justo |
| `services/pipeline.py` | Ejecucion en background del pipeline, con lock y rate limit |

## Vistas

Dos planos, como en el resto del sistema: operacion e investigacion.

| Ruta | Vista | Contenido |
| --- | --- | --- |
| `/ops/` | Backtest OPS | Reproduccion semana a semana del backtest de origen rodante; compara cadencias de reentreno con intervalo bootstrap |
| `/dashboard/` | Dashboard | KPIs, grafico demanda real vs predicha con banda conformal, modulos matematicos |
| `/skus/` | Analisis SKU | Tabla por SKU con busqueda, filtro por estado, orden y ajustes manuales de pedido |
| `/drift/` | Monitor de drift | PSI por feature con histogramas referencia vs actual |
| `/eda/` | EDA | Las figuras del modulo de analisis exploratorio, redibujadas como SVG |
| `/latent/` | Demanda latente | Veredicto de calidad de reconstruccion y comparacion por estrategia |
| `/pareto/` | Pareto tuning | Frente multiobjetivo, sensibilidad al ratio de costes, backtest de coste justo |

Cada pestana es una URL real: la navegacion es historial del navegador, no
estado de componente.

### Estados vacios

Cuando falta un artefacto, la vista lo dice explicitamente y nombra el comando
que lo genera. **No se fabrican datos de relleno.** Un run sin
`drift_report.json` muestra "sin informe de drift", no un panel vacio que se
confunda con "cero deriva".

## Hojas de estilo

Cinco ficheros, repartidos **por rol** y cargados en ese orden, que es la cascada:

| Fichero | Que contiene |
| --- | --- |
| `tokens.css` | El unico sitio donde se elige un color, una fuente o un alfa: 11 canales `--rgb-*` y la escalera de 12 alfas. Ninguna otra hoja introduce un valor de color crudo |
| `base.css` | Reset, tipografia, fondo del canvas, `.glass`, `.tnum`, `.label-mono`, scrollbar. Nada con nombre de pieza de UI |
| `layout.css` | Rejilla de la app, columna del sidebar, canvas, utilidades de rejilla. Donde van las cosas, nunca como se ven |
| `components.css` | Piezas reutilizables: tarjetas, KPI, cromo de graficos, leyendas, modales, cajones, sliders, botones, consola |
| `views.css` | Lo especifico de una vista: tabla SKU, drift, OPS, EDA, plano de investigacion, login, referencia de API |

Antes eran dos, `app.css` (heredado del `index.html` monolitico) y `components.css`
(era Django), **repartidos por tema**: 15 familias de componentes tenian reglas en los
dos ficheros, asi que cambiar una tarjeta KPI obligaba a editar dos sitios y confiar en
la cascada. Regla de oro del reparto nuevo: una regla vive en `views.css` solo mientras
la usa exactamente una vista; cuando la necesita una segunda, se muda a
`components.css` (asi paso `ops-chart-*` a `card-*`).

Para comprobar un refactor de estilos: `make render-snapshots SNAPSHOT=antes`, cambiar,
`make render-snapshots SNAPSHOT=despues` y `diff -r`. Un cambio solo-CSS debe dejar los
12 snapshots identicos byte a byte. Para la cascada, que el HTML no ve, se compara el
estilo computado de cada elemento en el navegador entre las dos versiones de la hoja.

## Graficos renderizados en servidor

Todos los graficos son SVG generado en Python. No hay libreria de graficos en el
cliente.

- La geometria se calcula en `charts.py` / `eda_charts.py` y se envia como
  marcado.
- El escalado responsive es `viewBox` + `width="100%"`, no un `ResizeObserver`.
- Los trazos llevan `vector-effect="non-scaling-stroke"` donde el `viewBox` se
  estira horizontalmente, para que el grosor de linea no se deforme.
- Los colores son variables CSS (`var(--c-conf)`, `var(--c-inv)`…), asi que la
  paleta vive en la hoja de estilos y no duplicada en Python.
- **Una sola proporcion para los graficos que ocupan una `.chart-card` entera**:
  `charts.HERO_WIDTH/HERO_HEIGHT` (900x230) y su equivalente a 760 de ancho,
  `eda_charts.HERO_HEIGHT`. Como el SVG se escala por el ancho, la proporcion *es*
  la altura renderizada: con el tope de 1200px de `.chart-card` todos caen en
  ~307px, en vez de que cada vista elija su propia altura. `eda_charts.HEIGHT`
  (300) se queda para los mosaicos multicolumna del EDA, que nunca son de ancho
  completo. El scatter de Pareto es mas cuadrado a proposito (un scatter lo
  necesita) y se iguala capando `.pareto-grid` al mismo 1200px.
- El lenguaje visual tambien es unico: violeta discontinuo = prediccion, franja
  gris-azul = intervalo conformal, punto = valor realizado. Los ejes usan
  `hero_y_domain()`, que da etiquetas redondas sin estirar el dominio hasta el
  primer tick (eso desperdiciaba media altura del panel).
- El hover es agnostico del grafico: la vista envuelve el SVG en `.chart-canvas`
  con `data-chart-points` / `data-chart-width` / `data-chart-height`, y el
  manejador rellena cada `[data-field="x"]` del tooltip con la propiedad `x` del
  punto, ya formateada en Python. Anadir tooltip a un grafico nuevo no toca el JS.

El unico JavaScript propio son ~140 lineas: el manejador de puntero que mueve la
cruz y el tooltip (`static/js/chart.js`, compartido por el grafico del dashboard y
el del plano OPS) y el inicializador de KaTeX
(`static/js/latex.js`).

## Interactividad

**htmx** para todo lo que necesita datos nuevos del servidor, **Alpine** para lo
puramente visual (modales, lightbox, valores en vivo de los sliders).

Los parametros de escenario viven en la query string, no en estado de cliente:

```text
/skus/?service_level=88&shortage_cost=25&holding_cost=4&capacity=12000&sort=q_star&dir=asc
```

Consecuencia deseada: cualquier escenario es una URL compartible y recargable, y
el boton "atras" funciona.

El patron de intercambio es siempre el mismo: el control hace `hx-get` sobre la
vista actual con `hx-target="#view-content"` y `hx-select="#view-content"`, de
modo que el servidor recalcula y solo se reemplaza la region de contenido.

### Estado que si es del cliente

Los ajustes manuales de cantidad en la tabla de SKU viven en `localStorage`.
Nunca se han persistido en servidor y siguen sin hacerlo: son un cuaderno
privado del operador, no una decision del sistema.

## Autenticacion

Cuenta unica de operador, con credenciales de entorno (`AUTH_USERNAME`,
`AUTH_PASSWORD`). Comparacion en tiempo constante con `hmac.compare_digest`.

- Sesion en cookie firmada (`django.contrib.sessions.backends.signed_cookies`).
  Rotar `DJANGO_SECRET_KEY` cierra la sesion de todos.
- `AUTH_PASSWORD` vacio **deshabilita el acceso**. No se acepta contrasena vacia.
- `LoginRequiredMiddleware` protege todo salvo `login` y `health`. Distingue el
  modo de fallo: redireccion para navegacion, `401` JSON para `/api/*`, y
  cabecera `HX-Redirect` para peticiones htmx (que no pueden seguir un `302`
  hacia el interior de un fragmento).

## Superficie JSON

La mayoria de endpoints antiguos existian solo para alimentar al cliente y han
desaparecido en favor de HTML renderizado. Se conserva el subconjunto que es una
interfaz maquina-a-maquina real, documentado en `/api/`:

| Metodo | Ruta | Auth |
| --- | --- | --- |
| `GET` | `/health` | publico |
| `POST` | `/api/forecast` | sesion |
| `GET` | `/api/skus` | sesion |
| `GET` | `/api/download/predictions` | sesion |
| `GET` | `/api/download/costs` | sesion |
| `POST` | `/predict_orders` | sesion |

Los dos endpoints `POST` (`/api/forecast` y `/predict_orders`) estan exentos de
CSRF: son destinos maquina autenticados por cookie de sesion, no de formulario.
El dashboard nunca los usa; para sus propias interacciones tiene las vistas de
fragmento.

## Ejecucion del pipeline desde el dashboard

`services/pipeline.py` lanza `python -m retail_forecasting.run` como subproceso y
vuelca la salida a `var/active_run.log`. Un lock permite una sola ejecucion
simultanea; un rate limit por IP (3 cada 10 minutos) evita el machaque.

La consola hace polling con `hx-trigger="every 1s"` mientras el estado es
`running`, y deja de pedir sola cuando termina, porque una respuesta de run
finalizado se renderiza sin ese atributo.

Al arrancar una ejecucion se invalida la cache del `ArtifactStore`, porque las
predicciones estan a punto de cambiar.

## Trampas conocidas

Cuatro errores que ya se han cometido una vez y conviene no repetir:

1. **Los comentarios `{# #}` de Django no admiten varias lineas.** Un comentario
   multilinea se renderiza como texto visible en la pagina. Usa
   `{% comment %}…{% endcomment %}`.

2. **`LANGUAGE_CODE = "es-es"` localiza los numeros en plantilla.** `95.0` sale
   como `95,0`, lo que rompe expresiones de Alpine y genera CSS invalido
   (`width: 45,3%`). Para cualquier numero que acabe en JavaScript, CSS o un
   atributo SVG usa el sufijo `u` de `floatformat` (`|floatformat:"3u"`) o el
   filtro `|unlocalize`. Ojo: `|unlocalize` **despues** de `floatformat` no hace
   nada, porque `floatformat` ya devuelve una cadena localizada.

3. **Los atributos htmx se heredan por el arbol DOM.** Un elemento con `hx-get`
   dentro de un boton que declara `hx-target` adoptara ese target. Si no es lo
   que quieres, pon `hx-target="this"` explicito.

4. **No uses `hx-swap-oob` en un fragmento que ademas es la respuesta directa a
   su propia peticion.** htmx lo intercambia dos veces y arrasa elementos
   vecinos. Si el fragmento se sirve en ambos contextos, condiciona el atributo
   (`{% if oob %}`) y activalo solo cuando viaja dentro de otra respuesta.

## Estaticos y dependencias externas

`static/vendor/` contiene htmx y Alpine servidos desde el propio dominio (96 KB
en total). Quedan dos dependencias de CDN heredadas: **Google Fonts** para las
tres familias tipograficas y **KaTeX** para renderizar las formulas de los
modulos matematicos.

Con `DEBUG=false` los estaticos se sirven con `ManifestStaticFilesStorage`, lo
que exige ejecutar `collectstatic` (el `Dockerfile` lo hace en tiempo de build).

## Despliegue

```text
Caddy (TLS)  ->  uvicorn retail_forecasting.api.asgi:application  ->  mlflow.db + mlruns/
```

Variables obligatorias en produccion: `DJANGO_SECRET_KEY`,
`DJANGO_ALLOWED_HOSTS` y `AUTH_PASSWORD`. Ver `.env.example`.

`SECURE_PROXY_SSL_HEADER` esta configurado porque el TLS lo termina el proxy
inverso: sin el, `request.is_secure()` siempre seria falso y las cookies seguras
no se respetarian.

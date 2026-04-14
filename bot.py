import os

# 1. Función para limpiar el texto (Preprocesado)
def preprocesar_input(texto):
    texto = texto.lower() # Todo a minúsculas
    # Corregimos palabras mal escritas
    ERRORES = {"internat": "internet", "ruter": "router"}
    for mal, bien in ERRORES.items():
        texto = texto.replace(mal, bien)
    return texto

# 2. El "Cerebro" que busca en los documentos (Simulación de RAG)
def recuperar_chunks(pregunta):
    # Aquí el robot lee el archivo faq.txt
    with open("documentos/faq.txt", "r", encoding="utf-8") as f:
        contenido = f.read()
    
    # Si la pregunta tiene palabras que están en el archivo, le da el trozo de texto
    if "router" in pregunta or "wifi" in pregunta or "contraseña" in pregunta:
        return contenido
    return "" # Si no encuentra nada, devuelve vacío

# 3. La lógica de decisión (¿Sabe o no sabe?)
def es_respuesta_valida(respuesta, contexto):
    if contexto == "": # Si no encontramos papeles en el archivador
        return False
    if "no sé" in respuesta.lower() or "lo siento" in respuesta.lower():
        return False
    return True

# 4. El Gran Final: El Pipeline
def manejar_pregunta(pregunta_usuario):
    print(f"\n--- Procesando: {pregunta_usuario} ---")
    
    # A. Limpiar
    limpio = preprocesar_input(pregunta_usuario)
    
    # B. Buscar información
    contexto = recuperar_chunks(limpio)
    
    # C. Generar respuesta (Aquí simulamos al LLM para que no gastes dinero aún)
    if "router" in limpio:
        respuesta = "Para arreglar el router debes esperar 30 segundos y reenchufar."
    elif "wifi" in limpio or "contraseña" in limpio:
        respuesta = "La clave del wifi es 12345678."
    else:
        respuesta = "No tengo información sobre eso."

    # D. Validar y decidir si escalamos a un humano
    if not es_respuesta_valida(respuesta, contexto):
        return "🆘 No estoy seguro de la respuesta. Te paso con un humano..."
    
    return "🤖 Bot dice: " + respuesta

# --- PROBEMOS EL SISTEMA ---
print(manejar_pregunta("¿Como arreglo el ruter?")) # Error ortográfico a propósito
print(manejar_pregunta("Quiero pedir una pizza"))   # Algo que no sabe hacer
recommendation_system.py.
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1) Datos de ejemplo: usuario, pelicula, calificacion
# ------------------------------------------------------------
ratings = pd.DataFrame({
    "usuario": ["U1", "U1", "U1", "U2", "U2", "U2", "U3", "U3", "U3", "U4", "U4", "U4"],
    "pelicula": ["P1", "P2", "P3", "P1", "P2", "P4", "P2", "P3", "P4", "P1", "P3", "P4"],
    "rating": [5, 4, 3, 4, 5, 2, 2, 5, 4, 5, 1, 3]
})

print("Datos originales:")
print(ratings)
print()

# ------------------------------------------------------------
# 2) Crear matriz usuario x pelicula
#    Cada fila = usuario
#    Cada columna = pelicula
#    El valor = rating
# ------------------------------------------------------------
matriz = ratings.pivot_table(
    index="usuario",
    columns="pelicula",
    values="rating",
    fill_value=0
)

print("Matriz usuario-pelicula:")
print(matriz)
print()

# ------------------------------------------------------------
# 3) Calcular similitud entre usuarios con cosine_similarity
#    Cosine similarity mide qué tan parecidos son dos usuarios
#    según sus ratings.
# ------------------------------------------------------------
similitudes = cosine_similarity(matriz)

# Lo convertimos a DataFrame para facilitar lectura y acceso
sim_df = pd.DataFrame(
    similitudes,
    index=matriz.index,
    columns=matriz.index
)

print("Matriz de similitud entre usuarios:")
print(sim_df)
print()

# ------------------------------------------------------------
# 4) Función para recomendar películas
# ------------------------------------------------------------
def recomendar(usuario_objetivo, top_n=3):
    """
    Recomendación simple:
    - Ignora las películas que ya ha visto el usuario
    - Busca usuarios similares
    - Suma la puntuación que esos usuarios dieron a cada película
    - La recomienda según la similitud ponderada
    """
    # 4.1) Si el usuario no existe en la matriz, avisamos
    if usuario_objetivo not in matriz.index:
        raise ValueError(f"El usuario '{usuario_objetivo}' no existe en los datos.")

    # 4.2) Obtener usuarios similares (sin incluirse a sí mismo)
    usuarios_similares = sim_df[usuario_objetivo].drop(usuario_objetivo)

    # Ordenamos de mayor a menor similitud
    usuarios_similares = usuarios_similares.sort_values(ascending=False)

    # 4.3) Generar lista de recomendaciones
    recomendaciones = []

    # Recorremos todas las películas
    for pelicula in matriz.columns:
        # Si el usuario ya vio esa película, la ignoramos
        if matriz.loc[usuario_objetivo, pelicula] != 0:
            continue

        # 4.4) Calculamos una puntuación ponderada por similitud
        #     score = suma(similitud_usuario * rating_del_otro_usuario)
        #     dividido por suma de similitudes absolutas
        #     Esto evita que una película con pocos votos similares se sobrevalorice.
        puntuaciones = []

        for otro_usuario, similitud in usuarios_similares.items():
            rating = matriz.loc[otro_usuario, pelicula]
            if rating > 0:
                puntuaciones.append(similitud * rating)

        # Si no hay usuarios que hayan visto la película, la ignoramos
        if not puntuaciones:
            continue

        # Ponderamos por similitud
        score = sum(puntuaciones) / (usuarios_similares.abs().sum() + 1e-9)

        recomendaciones.append((pelicula, score))

    # 4.5) Ordenamos por mayor score
    recomendaciones = sorted(recomendaciones, key=lambda x: x[1], reverse=True)

    # 4.6) Devolvemos las mejores N recomendaciones
    return recomendaciones[:top_n]

# ------------------------------------------------------------
# 5) Probar la recomendación para un usuario
# ------------------------------------------------------------
usuario = "U3"
recomendaciones = recomendar(usuario, top_n=3)

print(f"Recomendaciones para {usuario}:")
for pelicula, score in recomendaciones:
    print(f"- {pelicula}: score estimado = {score:.4f}")

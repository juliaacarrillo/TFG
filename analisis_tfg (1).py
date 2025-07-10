#!/usr/bin/env python
# coding: utf-8

# ## INDICE:
# * [Analisis temporal](#Analisis-temporal)
# * [Clustering](#Clustering)
# * [Clustering incluyendo tiempos de decision](#Clustering-tiempos-decisión)
# * [Gráficos variados](#Radar)
# * [Regresión](#Regresion)

# ## Analisis temporal <a class="anchor" id="Analisis-temporal"></a>

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt

#LECTURA DE DATOS

datos = 'pgg_dynamic_resource (accessed 2024-04-18).xlsx'
#guardo todos los datos del excel en un DataFrame 
df = pd.read_excel(datos, engine='openpyxl')

df_resources= df[['player.id_in_group','group.id_in_subsession','group.n_resources','subsession.round_number','session.code']]


# China/España

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt

#LECTURA DE DATOS
#guardo todos los datos del excel en un DataFrame 
#df = pd.read_csv("experimental_data.tsv", sep="\t")

#df_resources_old= df[['player.id_in_group','group.id_in_subsession','group.n_resources','subsession.round_number','session.code','country']]
#df_resources_old = df_resources_old[df_resources_old['country'] == "Spain"]

#df_resources_old.head()


# In[3]:


import matplotlib.pyplot as plt
import pandas as pd # Importamos pandas


# Paso previo: Generar session_group_n_resources a partir de `df_resources`
session_group_n_resources = {}

for session_code in df_resources['session.code'].unique():
    session_data = df_resources[df_resources['session.code'] == session_code]
    group_n_resources = {}
    for group in sorted(session_data['group.id_in_subsession'].unique()):
        group_resources = []
        for ronda in sorted(session_data['subsession.round_number'].unique()):
            filtro = (session_data['group.id_in_subsession'] == group) & \
                     (session_data['subsession.round_number'] == ronda)
            resources_ronda = session_data[filtro]['group.n_resources'].drop_duplicates()
            if not resources_ronda.empty:
                group_resources.append(resources_ronda.iloc[0])
        group_n_resources[group] = group_resources
    session_group_n_resources[session_code] = group_n_resources

# Paso 1: Crear un identificador único para cada grupo en cada sesión y recolectar los recursos por grupo único
group_resources_across_rounds = {}
for session_code, group_data in session_group_n_resources.items():
    for group_id, resources_list in group_data.items():
        unique_group_id = f"{session_code}_grupo_{group_id}"
        if unique_group_id not in group_resources_across_rounds:
            group_resources_across_rounds[unique_group_id] = []
        group_resources_across_rounds[unique_group_id].extend(resources_list)

# ----------- INICIO DE LA MODIFICACIÓN PARA CREAR EL DATAFRAME -----------
# Paso 2: Crear el DataFrame a partir de group_resources_across_rounds

# Preparamos los datos para el DataFrame
data_for_df = []
for unique_group_id, resources in group_resources_across_rounds.items():
    for i, resource_value in enumerate(resources):
        round_number = i + 1 # Las rondas empiezan en 1
        data_for_df.append({
            'unique_group_id': unique_group_id,
            'round_number': round_number,
            'resources': resource_value
        })

# Creamos el DataFrame
df_datos_grupos = pd.DataFrame(data_for_df)

# ----------- FIN DE LA MODIFICACIÓN PARA CREAR EL DATAFRAME -----------

# Paso 3: Generar el gráfico (esto sigue igual que tu código original)
plt.figure(figsize=(10, 8))

# Ordenar los IDs únicos: primero por el nombre de la sesión (alfabético)
# y luego por el número de grupo (numérico)
sorted_group_ids = sorted(
    group_resources_across_rounds.keys(),
    key=lambda x: (x.split("_grupo_")[0], int(x.split("_grupo_")[1]))
)

# Iterar en orden y graficar
for unique_group_id in sorted_group_ids:
    resources = group_resources_across_rounds[unique_group_id]
    rounds = list(range(1, len(resources) + 1))
    plt.plot(rounds, resources, label=unique_group_id, marker='o', alpha=0.7)
plt.xlabel('Ronda',fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Recurso',fontsize=22)
plt.ylim(0, 400)
plt.yticks(range(0, 401, 100))
# Añadir la leyenda a un lado
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=20)
plt.tight_layout() # Ajusta el layout para que todo encaje bien
plt.show()


# In[4]:


df_datos_grupos


# Esto para analizar los datos de China/España que tengo el df_resources_old

# In[5]:


import matplotlib.pyplot as plt

# Paso previo: Generar session_group_n_resources a partir de `df_resources`
# Este código asume que tienes un DataFrame `df_resources` con las columnas 'session.code', 'group.id_in_subsession', 'subsession.round_number', y 'group.n_resources'.

# Crear un diccionario para almacenar los recursos por sesión y grupo
session_group_n_resources = {}

# Agrupar los datos por sesión, grupo y ronda
for session_code in df_resources['session.code'].unique():  # Iteramos sobre todas las sesiones
    session_data = df_resources[df_resources['session.code'] == session_code]  # Filtrar los datos de la sesión actual

    # Crear un diccionario para almacenar los recursos por grupo dentro de esta sesión
    group_n_resources = {}

    for group in sorted(session_data['group.id_in_subsession'].unique()):  # Iteramos sobre los grupos de la sesión
        group_resources = []
        for ronda in sorted(session_data['subsession.round_number'].unique()):  # Iteramos sobre las rondas en orden
            # Filtrar los datos para el grupo y la ronda actual
            filtro = (session_data['group.id_in_subsession'] == group) & (session_data['subsession.round_number'] == ronda)
            resources_ronda = session_data[filtro]['group.n_resources'].drop_duplicates()

            if not resources_ronda.empty:
                group_resources.append(resources_ronda.iloc[0])  # Tomamos el primer valor único
        
        # Añadimos los recursos del grupo a la sesión actual
        group_n_resources[group] = group_resources
    
    # Guardamos el diccionario de grupos y sus recursos en la sesión
    session_group_n_resources[session_code] = group_n_resources

# Paso 1: Crear un identificador único para cada grupo en cada sesión y recolectar los recursos por grupo único (combinación de sesión y grupo)
group_resources_across_rounds = {}

# Iteramos sobre cada sesión y sus recursos por grupo
for session_code, group_n_resources in session_group_n_resources.items():
    for group, resources in group_n_resources.items():
        unique_group_id = f"{session_code}_grupo_{group}"

        if unique_group_id not in group_resources_across_rounds:
            group_resources_across_rounds[unique_group_id] = []
        
        group_resources_across_rounds[unique_group_id].extend(resources)

# Paso 2: Generar el gráfico

# Creamos una figura para el gráfico
plt.figure(figsize=(10, 8))

# Iteramos sobre los grupos y graficamos sus recursos
for unique_group_id, resources in group_resources_across_rounds.items():
    rounds = list(range(1, len(resources) + 1))  # Eje X

    plt.plot(rounds, resources, label=unique_group_id, marker='o')

# Personalizamos el gráfico
# Personalización
plt.title('Recursos por ronda', fontsize=22)
plt.xlabel('Rondas', fontsize=21)
plt.ylabel('Recursos', fontsize=21)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylim(0, 400)
plt.yticks(range(0, 401, 100))
plt.legend(fontsize=13)
plt.show()


# In[6]:


# Creamos una figura para el gráfico
plt.figure(figsize=(10, 8))

# Para controlar si ya hemos graficado una categoría
depleted_plotted = False
overexploiting_plotted = False
optimal_plotted = False

# Lista para almacenar los grupos
overexploiting_groups = []
depleted_groups = []
optimal_groups = []

# Iteramos sobre los grupos y graficamos sus recursos
for unique_group_id, resources in group_resources_across_rounds.items():
    rounds = list(range(1, len(resources) + 1))
    if any(r < 100 for r in resources):  # 1. ¿Agotado?
        color = 'red'
        marker = 's'
        label = 'Agotamiento' if not depleted_plotted else None
        depleted_plotted = True
        depleted_groups.append(unique_group_id)
        overexploiting_groups.append(unique_group_id)
    elif any(r < 134 for r in resources):  # 2. Si no es Agotado, ¿Sobreexplotado?
                                          # (esto implica que algún r < 136)
        color = 'grey'
        marker = '^'
        label = 'Sobreexplotación' if not overexploiting_plotted else None
        overexploiting_plotted = True
        overexploiting_groups.append(unique_group_id)
    else:  # 3. Si no es Agotado NI Sobreexplotado, entonces DEBE SER ÓPTIMO
           # (esto implica que todos los r >= 136)
        color = 'blue'
        marker = 'o'
        # La condición resources[-1] > 151 puede ser una característica especial
        # DENTRO de los Óptimos, pero la categoría principal es Óptimo.
        # Si quieres una etiqueta diferente para esos, podrías añadir un if aquí dentro.
        # Por ejemplo:
        # if resources[-1] > 151:
        #     label = 'Óptimo (Final > 151)' if not optimal_plus_plotted else None
        # else:
        #     label = 'Óptimo' if not optimal_plotted else None
        # Pero para mantenerlo simple y según tu definición base:
        label = 'Óptimo' if not optimal_plotted else None
        optimal_plotted = True
        optimal_groups.append(unique_group_id)

   # Crear los índices de los puntos que quieres marcar: cada 5 rondas desde 1 a 50
    mark_indices = [i for i, r in enumerate(rounds) if (r - 1) % 5 == 0 or r == 50]

    # Usar markevery con la lista de índices
    plt.plot(rounds, resources, label=label, color=color, marker=marker, markersize=8, markevery=mark_indices)

  

# Línea horizontal discontinua
plt.axhline(y=151, color='blue', alpha=0.6,linestyle='--')
plt.axhline(y=100, color='red', alpha=0.6,linestyle='--')

# Personalización
#plt.title('Recursos por ronda', fontsize=22)
plt.xlabel('Ronda', fontsize=21)
plt.ylabel('Recurso', fontsize=21)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylim(0, 400)
plt.grid(False)
plt.yticks(range(0, 401, 100))
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize=21)
plt.tight_layout()
plt.show()
plt.savefig("evolucion.pdf", bbox_inches='tight')

# Imprimimos los grupos sobreexplotados
print("Grupos sobreexplotados (Overexploiting):", overexploiting_groups)
print("Grupos sobreexplotados (Depleted):", depleted_groups)
print("Grupos sobreexplotados (Optimal):", optimal_groups)


# In[7]:


sustained_groups = list(set(overexploiting_groups) - set(depleted_groups))
len(sustained_groups)


# Para ovservar lo que sucede en los grupos sobreexplotados, vamos a realizar un análisis estadístico de la evolución con un modelo de serie temporal autoregresivo (AR) aplicado a series temporales de los recursos de los grupos: 
# $R_t=c_o+c_1(t-t_o)+(1+c_2)R_{t-1}
# +c_3(R_{t-1}
# -R_{t-2}
# )$
# 

# PRUEBA DE REGRESIÓN 

# In[8]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# rondas de la 20 a la 50
df_datos_grupos_filtered = df_datos_grupos[
    (df_datos_grupos['round_number'] >= 20) &
    (df_datos_grupos['round_number'] <= 50)
].copy()

df_sustained = df_datos_grupos_filtered[df_datos_grupos_filtered['unique_group_id'].isin(sustained_groups)].copy()
results = {}

# Iterate over each unique group in the filtered data
for group_id in df_sustained['unique_group_id'].unique():
    group_data = df_sustained[df_sustained['unique_group_id'] == group_id].sort_values(by='round_number').copy()

    # Calculate t_0 for the current group
    t_0 = group_data['round_number'].min()

    # Create lagged variables and (t - t_0)
    group_data['resources_lag1'] = group_data['resources'].shift(1)
    group_data['resources_lag2'] = group_data['resources'].shift(2)
    group_data['t_minus_t0'] = group_data['round_number'] - t_0

    # Drop rows with NaN values introduced by shifting
    group_data = group_data.dropna()

    if len(group_data) > 2: # Need at least 3 data points for R_t, R_t-1, R_t-2
        # This transformation is needed to correctly interpret c2 directly from regression coefficients
        group_data['dependent_var_transformed'] = group_data['resources'] - group_data['resources_lag1']
        group_data['R_t_minus_R_t_minus_1'] = group_data['resources_lag1'] - group_data['resources_lag2']

        # Define the formula for OLS.
        try:
            model_formula = 'dependent_var_transformed ~ t_minus_t0 + resources_lag1 + R_t_minus_R_t_minus_1'
            model = ols(model_formula, data=group_data).fit()

            # Extract coefficients
            c0_est = model.params['Intercept']
            c1_est = model.params['t_minus_t0']
            c2_est = model.params['resources_lag1']
            c3_est = model.params['R_t_minus_R_t_minus_1']

            # Extract p-values
            p_val_c0 = model.pvalues['Intercept']
            p_val_c1 = model.pvalues['t_minus_t0']
            p_val_c2 = model.pvalues['resources_lag1']
            p_val_c3 = model.pvalues['R_t_minus_R_t_minus_1']

            # Function to get significance stars
            def get_stars(p_value):
                if p_value < 0.001:
                    return '***'
                elif p_value < 0.01:
                    return '**'
                elif p_value < 0.05:
                    return '*'
                else:
                    return ''

            stars_c0 = get_stars(p_val_c0)
            stars_c1 = get_stars(p_val_c1)
            stars_c2 = get_stars(p_val_c2)
            stars_c3 = get_stars(p_val_c3)

            results[group_id] = {
                'c0': c0_est,
                'c0_stars': stars_c0,
                'c1': c1_est,
                'c1_stars': stars_c1,
                'c2': c2_est,
                'c2_stars': stars_c2,
                'c3': c3_est,
                'c3_stars': stars_c3,
                'summary': model.summary().as_text(), # Store full summary for detailed analysis
                'recovering_resource': c1_est > 0, # Check the condition for recovery
                'R2': model.rsquared,
                'R2_adj': model.rsquared_adj
            }
        except Exception as e:
            results[group_id] = {'error': str(e)}
            print(f"Error fitting model for group {group_id}: {e}")
    else:
        results[group_id] = {'error': 'Not enough data points for regression after lagging.'}
        print(f"Skipping group {group_id}: Not enough data points after lagging.")

# Print results for each group
for group_id, res in results.items():
    print(f"\n--- Group {group_id} ---")
    if 'error' in res:
        print(f"Error: {res['error']}")
    else:
        print(f"Estimated c0: {res['c0']:.4f}{res['c0_stars']}")
        print(f"Estimated c1 (trend): {res['c1']:.4f}{res['c1_stars']}")
        print(f"Estimated c2: {res['c2']:.4f}{res['c2_stars']}")
        print(f"Estimated c3: {res['c3']:.4f}{res['c3_stars']}")
        print(f"Is resource recovering (c1 > 0)? {res['recovering_resource']}")
        # print("\nModel Summary:")
        # print(res['summary']) # Uncomment to see the full summary for each group


# In[9]:


from tabulate import tabulate

# Lista para recolectar resultados
result_rows = []

for group_id, res in results.items():
    if 'error' not in res:
        resource_status = 'depleting' if res['c1'] < 0 and res['c1_stars'] else 'sustained'

        # Extraer R² y R² ajustado desde el diccionario directamente
        r_squared = res.get('R2', np.nan)
        r_squared_adj = res.get('R2_adj', np.nan)

        result_rows.append({
            'Group ID': group_id,
            'c₀': f"{res['c0']:.4f}{res['c0_stars']}",
            'c₁ (trend)': f"{res['c1']:.4f}{res['c1_stars']}",
            'c₂': f"{res['c2']:.4f}{res['c2_stars']}",
            'c₃': f"{res['c3']:.4f}{res['c3_stars']}",
            'R²': round(r_squared, 4),
            'R²_adj': round(r_squared_adj, 4),
            'Resource': resource_status
        })

# Crear y mostrar tabla
results_df = pd.DataFrame(result_rows).sort_values(by='Group ID')
print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False))

# Opcional: exportar
# results_df.to_csv("modelo_resultados.csv", index=False)
# results_df.to_excel("modelo_resultados.xlsx", index=False)


# In[10]:


import pandas as pd
import numpy as np

# --- Preparar los datos como antes ---
result_rows = []
for group_id, res in results.items():
    if 'error' not in res:
        resource_status = 'Depleting' if res['c1'] < 0 and res['c1_stars'] else 'Sustained'
        r_squared = res.get('R2', np.nan)
        r_squared_adj = res.get('R2_adj', np.nan)

        result_rows.append({
            'Group': group_id,  # 👈 Cambiado aquí
            '$c_0$': f"{res['c0']:.3f}{res['c0_stars']}",
            '$c_1$': f"{res['c1']:.3f}{res['c1_stars']}",
            '$c_2$': f"{res['c2']:.3f}{res['c2_stars']}",
            '$c_3$': f"{res['c3']:.3f}{res['c3_stars']}",
            '$R^2$': f"{r_squared:.2f}",
            '$R^2_{adj}$': f"{r_squared_adj:.2f}",
            'Resource': resource_status
        })

df = pd.DataFrame(result_rows).sort_values(by='Group')

# --- Generar LaTeX ---
latex = df.to_latex(index=False,
                    caption="Time-series analysis of the virtual forest's state to determine the presence of significant trends.",
                    label="tab:virtual_forest",
                    column_format='lccccccc',
                    escape=False)

# --- Agregar encabezados agrupados manualmente ---
header = (
    "\\begin{table}[htbp]\n\\centering\n"
    "\\caption{Time-series analysis of the virtual forest's state to determine the presence of significant trends.}\n"
    "\\label{tab:virtual_forest}\n"
    "\\begin{tabular}{lccccccc}\n"
    "\\toprule\n"
    " & \\multicolumn{4}{c}{\\textbf{Parameters}} & \\multicolumn{2}{c}{\\textbf{Goodness of fit}} & \\\\\n"
    "\\cmidrule(lr){2-5} \\cmidrule(lr){6-7}\n"
    "\\textbf{Group} & $c_0$ & $c_1$ & $c_2$ & $c_3$ & $R^2$ & $R^2_{adj}$ & \\textbf{Resource} \\\\\n"
    "\\midrule\n"
)

footer = "\\bottomrule\n\\end{tabular}\n\\end{table}"

# Extraer solo el cuerpo de la tabla (sin header/footer generados por pandas)
body = "\n".join(latex.splitlines()[5:-3])

# Unir todo
full_latex = header + body + "\n" + footer

# Imprimir para copiar/pegar en Overleaf
print(full_latex)



# Repito el ajuste pero para que ahora me devuelva los datos en una tabla, donde cada parámetro aparece tambien con sus significancia estadística mediante asteríscos: 
# $* \rightarrow 5\% \quad ** \rightarrow 1\% \quad *** \rightarrow 0.1\%$

# In[11]:


from tabulate import tabulate  # pip install tabulate si no lo tienes

# Lista para recolectar resultados
result_rows = []

for group_id, res in results.items():
    if 'error' not in res:
        # Clasificar el grupo como depleting o sustained
        resource_status = 'depleting' if res['c1'] < 0 and res['c1_stars'] else 'sustained'

        # Extraer R-squared y R-squared adjusted desde el texto del summary
        try:
            summary_text = res['summary']
            r_squared = float(summary_text.split('R-squared:')[1].split('\n')[0].strip())
            r_squared_adj = float(summary_text.split('Adj. R-squared:')[1].split('\n')[0].strip())
        except:
            r_squared = np.nan
            r_squared_adj = np.nan

        result_rows.append({
            'Group ID': group_id,
            'c₀': f"{res['c0']:.4f}{res['c0_stars']}",
            'c₁ (trend)': f"{res['c1']:.4f}{res['c1_stars']}",
            'c₂': f"{res['c2']:.4f}{res['c2_stars']}",
            'c₃': f"{res['c3']:.4f}{res['c3_stars']}",
            'R²': round(r_squared, 4),
            'R²_adj': round(r_squared_adj, 4),
            'Resource': resource_status
        })

# Crear DataFrame y ordenar
results_df = pd.DataFrame(result_rows).sort_values(by='Group ID')

# Mostrar tabla bonita en consola
print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False))

# Exportar si lo deseas
# results_df.to_csv("modelo_resultados.csv", index=False)
# results_df.to_excel("modelo_resultados.xlsx", index=False)


# In[12]:


# Crear listas de grupos por categoría
depleting_groups = results_df[results_df['Resource'] == 'depleting']['Group ID'].tolist()
sustained_groups = results_df[results_df['Resource'] == 'sustained']['Group ID'].tolist()

# Mostrar los arrays (opcional)
print(f"\nDepleting Groups ({len(depleting_groups)}): {depleting_groups}")
print(f"Sustained Groups ({len(sustained_groups)}): {sustained_groups}")


# In[13]:


import matplotlib.pyplot as plt

# Crear figura
plt.figure(figsize=(10, 8))

# Banderas para mostrar cada categoría solo una vez en la leyenda
sostenido_plotted = False
agotandose_plotted = False

for unique_group_id, resources in group_resources_across_rounds.items():
    # Asegurar que hay al menos 50 rondas
    if len(resources) < 50:
        continue

    # Seleccionar rondas 20 a 50 (índices 19 a 49)
    rounds_20_50 = list(range(20, 51))
    resources_20_50 = resources[19:50]

    # Ver si el grupo está en alguna categoría
    if unique_group_id in sustained_groups:
        color = '#25a18e'
        marker = '^'
        label = 'Sostenido' if not sostenido_plotted else None
        sostenido_plotted = True
    elif unique_group_id in depleting_groups:
        color = '#d62728'  # Rojo menos intenso
        marker = 'o'
        label = 'Agotándose' if not agotandose_plotted else None
        agotandose_plotted = True
    else:
        continue  # No pertenece a ninguna categoría relevante

    # Marcadores cada 5 rondas (dentro del rango 20–50)
    mark_indices = [i for i, r in enumerate(rounds_20_50) if (r - 1) % 5 == 0 or r == 50]

    # Graficar
    plt.plot(rounds_20_50, resources_20_50, label=label,
             color=color, marker=marker, markersize=13, markevery=mark_indices, linewidth=2.5)


# Personalización general
#plt.title('Recursos por ronda (20–50)', fontsize=22)
plt.xlabel('Ronda', fontsize=21)
plt.ylabel('Recurso', fontsize=21)
plt.grid(False)
plt.tick_params(axis='both', labelsize=20)
plt.ylim(100, 200)
plt.yticks(range(100, 201, 25))  # Puedes ajustar el salto (25, 20, etc.)

# Leyenda clara con solo las dos categorías
plt.legend(fontsize=21,loc='upper left')
plt.tight_layout()

# Mostrar y guardar
plt.show()
# plt.savefig("evolucion_filtrada.pdf", bbox_inches='tight')  # Descomenta para guardar


# <h3> Eliminaciónde los bots </h3>

# In[14]:


import pandas as pd 
df1 = pd.read_csv("ganadores/data_1.csv")
df2 = pd.read_csv("ganadores/data_2.csv")
df3 = pd.read_csv("ganadores/data_3.csv")
#unimos para tener los id que nos interesan en uno solo 
df_participants = pd.concat([df1[["ID"]],df2[["ID"]],df3[["ID"]]], ignore_index=True)
#print(df1["ID"],df2["ID"],df3["ID"])


# Una vez quitados los bots, nos quedan 12,21 y 22 participantes, es decir: 55 jugadores

# ## Clustering <a class="anchor" id="Clustering"></a>

# Ahora  me da igual el grupo y el id, solo quiero distinguir los datos. Al menos por ahora

# In[15]:


import pandas as pd
datos = 'pgg_dynamic_resource (accessed 2024-04-18).xlsx'
#guardo todos los datos del excel en un DataFrame 
df = pd.read_excel(datos, engine='openpyxl')
df_profit_all = df[['participant.id_in_session','participant.code', 'group.n_resources', 'group.id_in_subsession','player.harvesting_time', 'player.payoff', 'subsession.round_number', 'session.code']]
# Creamos la nueva columna 'session_grupo_id' combinando las columnas deseadas
df_profit_all['session_grupo_id'] = df_profit_all['session.code'].astype(str) + "_grupo_" + df_profit_all['group.id_in_subsession'].astype(str)
df_profit = df_profit_all[df_profit_all["participant.code"].isin(df_participants["ID"])]
conteo=df_profit.groupby('session.code')['participant.code'].nunique()
df_profit.head()


# In[16]:


pip install --user scikit-learn


# Ahora se necesitan las 4 características sobre las que aplicamos el algoritmo k-means: el esfuerzo acumulado de la primera mitad, el esfuerzo acumulado de la segunda mitad, las recompensas totales de la primera mitad y las recompensas totales de la segunda mitad. Así, cada jugador tendrá esos 4 valores asignados. Lo primero que tenemos que hacer es organizar los datos para ello.

# In[17]:


#separamos las mitades de las sesiones con la columna de las rondas
df_first_half = df_profit[df_profit['subsession.round_number'] <= 25]
df_second_half = df_profit[df_profit['subsession.round_number'] > 25]


# In[18]:


#para sacar las características, hay que usar groupby que agrupa por sesión y jugador (se incluye el id)
df_first_half_grouped = df_first_half.groupby(['participant.id_in_session','session.code','session_grupo_id']).agg(
    total_harvesting_first_half=('player.harvesting_time', 'sum'), #el agg es para agregar la suma que se especifica con 'sum'
    total_payoff_first_half=('player.payoff', 'sum'),
).reset_index() #con eso se añade a un nuevo data frame y se actualizan los indices 

#segunda mitad
df_second_half_grouped = df_second_half.groupby(['participant.id_in_session','session.code']).agg(
    total_harvesting_second_half=('player.harvesting_time', 'sum'),
    total_payoff_second_half=('player.payoff', 'sum'),
).reset_index()

#se junta todo con .merge según las columnas que los distinguen (id y sesión)
df_kmeans = pd.merge(df_first_half_grouped, df_second_half_grouped, on=['participant.id_in_session','session.code'],how='outer')

# Mostrar resultados finales
df_kmeans.head()


# El número total de jugadores es 55 con que tiene buena pinta.

# <h3> Número de clusters </h3>
# 
# Se utiliza el método Average Silhouette Width, evaluando cómo de bien se agrupan los puntos. Evalúa los posibles y el máximo del gráfico será el óptimo. Al final es hacer el ajuste para los diferentes valores y calcular el ancho de la silueta promedio.

# In[19]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # librería para el kmeans
from sklearn.metrics import silhouette_score  # para el método 

# Nuestras características son las 4 mencionadas
X = df_kmeans[['total_harvesting_first_half', 'total_payoff_first_half',
               'total_harvesting_second_half', 'total_payoff_second_half']]
# Guardaremos en una lista los valores de la Y
Y = []
# Bucle en k para probar los números de clusters
for k in range(2, 11): 
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)  # Hará el fit con los 4 parámetros 
    score = silhouette_score(X, kmeans.labels_)
    Y.append(score)

# Agregar el punto (1,0) a las listas de datos
X_values = [1] + list(range(2, 11))
Y_values = [0] + Y

# Graficar la puntuación de la silueta
plt.figure(figsize=(8, 6))
plt.plot(X_values, Y_values, marker='o', linestyle='-', color = '#25a18e')
plt.xlabel('Número de clusters (k)',fontsize=21)
plt.title('Método de Silueta',fontsize=22)
plt.ylabel('Ancho de silueta promedio',fontsize=21)
plt.xticks(X_values,fontsize=20)  # Mostrar todos los valores de k en el eje x
#plt.grid(True)
plt.yticks(fontsize=20)

# Mostrar el gráfico
plt.show()

# Mostrar los valores de la silueta para cada k (incluyendo el punto (1,0))
print("Silhouette scores for each k:")
for k, score in zip(X_values, Y_values):
    print(f'k={k}: {score}')


# El número de clusters óptimo es: 
# $$ N_{clusters}=4$$

# Además, tenemos un máximo local en 6 y otro en 8. En caso de estar realizando comparativas con otros datos, puede ser interesante usar estos otros valores en caso de ser máximo en ese otro conjunto de datos.

# El concepto del valor de silueta combina los siguientes elementos:
# 
# 1. **Cohesión**: mide qué tan cerca están los puntos dentro de su propio clúster, calculado como la *distancia promedio* entre un punto y los demás puntos de su clúster.  
# 2. **Separación**: mide qué tan lejos están los puntos de otros clústeres, calculado como la *distancia promedio* entre un punto y los puntos del clúster más cercano al que no pertenece.
# 
# De este modo, la fórmula del valor de silueta se define como:
# 
# $
# s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
# $
# 
# Donde:  
# - $a(i)$: es la distancia promedio entre el punto \(i\) y los demás puntos de su propio clúster.  
# - $b(i)$: es la distancia promedio entre el punto \(i\) y los puntos del clúster más cercano al que \(i\) no pertenece.
# 

# Podemos añadir la elipse de cada cluster (más fancy) mediante $sklearn.mixture.GaussianMixture$ que calcula la elipse que representa cada distribución gaussiana de cada cluster.

# <h4> Alternativa con elbow mode

# In[20]:


pip install kneed


# In[21]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # librería para el kmeans
from sklearn.metrics import silhouette_score  # para el método 

# Guardaremos en una lista los valores de la inercia 
inertia = []

# Bucle en k para probar los números de clusters
for k in range(2, 11): 
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)  # Hará el fit con los 4 parámetros 
    inertia.append(kmeans.inertia_)  # Guardamos la inercia


# In[22]:


from kneed import KneeLocator

# Encontrar el "codo" automáticamente con la función KneeLocator
kl = KneeLocator(range(2, 11), inertia, curve="convex", direction="decreasing")

# Valor óptimo de k
print(f"El número óptimo de clusters es: {kl.elbow}")

# Graficar con el codo resaltado
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertia, marker='o', linestyle='-', color = '#25a18e', label="Inercia")
plt.axvline(kl.elbow, color='r', linestyle='--', label=f"Codo en k={kl.elbow}")
plt.title('Método del Codo',fontsize=22)
plt.xlabel('Número de clusters (k)',fontsize=21)
plt.ylabel('WCSS',fontsize=21)
plt.xticks(range(2, 11),fontsize=20)
plt.yticks(fontsize=20)
#plt.legend()
plt.grid(False)
plt.show()


# Vamos a tomar 4, que es el siguiente máximo en el método de la silueta. Además así coincide con el número tomado en España en el experimento.

# <h3> Clustering por kmeans

# Usando la función KMeans se hacen las iteraciones necesarias hasta que converja 

# In[23]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#nuestras características son las 4 mencionadas
X = df_kmeans[['total_harvesting_first_half', 'total_payoff_first_half',
               'total_harvesting_second_half', 'total_payoff_second_half']]

#ahora se hace lo mismo que antes pero solo para el valor de número de cluster óptimo
kmeans = KMeans(n_clusters=4, random_state=42)
#añadimos una columna con el cluster al q pertenece cada fila (creo)
df_kmeans['Cluster'] = kmeans.fit_predict(X) # Hará el fit con los 4 parámetros y al ser _predict devuelve los centroides 
df_sesion = df_kmeans[df_kmeans['session.code'] == '4jl850y2']
df_sesion


# <h3> Diferentes agrupaciones 

# Primera mitad para el profit y para el effort

# In[24]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# A visualizar
x_feature = 'total_harvesting_first_half'
y_feature = 'total_payoff_first_half'

# Colores personalizados para cada clúster (ajusta según tus clústeres)
#VERDE
#AZUL 
#ROJO
#MORADO
colores_personalizados = ['#8cb369','#25a18e','#e76f51','#540d6e']  # Agrega más colores si tienes más clústeres

# Función para dibujar una elipse basada en la covarianza de los puntos de un cluster
def draw_ellipse(position, covariance, ax, color):
    v, w = np.linalg.eigh(covariance)  # Autovalores y autovectores
    angle = np.degrees(np.arctan2(w[0, 1], w[0, 0]))  # Ángulo de rotación
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # Escalado de los autovalores
    ellipse = Ellipse(position, width=v[0], height=v[1], angle=angle, 
                      edgecolor=color, facecolor=color, lw=2, alpha=0.3,zorder=3)
    ax.add_patch(ellipse)

# Crear la figura
plt.figure(figsize=(14, 10))
ax = plt.gca()

# Dibujar los puntos de dispersión con colores personalizados según el clúster
for i in df_kmeans['Cluster'].unique():  # Iterar sobre cada clúster
    plt.scatter(df_kmeans[df_kmeans['Cluster'] == i][x_feature], 
                df_kmeans[df_kmeans['Cluster'] == i][y_feature], 
                color=colores_personalizados[i],  # Asignar color basado en el índice del clúster
                label=f'Cluster {i}', 
                s=100, alpha=1)

# Obtener los clusters únicos y sus colores correspondientes
clusters = np.sort(df_kmeans['Cluster'].unique())

# Dibujar las elipses con los colores personalizados
for i, cluster in enumerate(clusters):
    cluster_data = df_kmeans[df_kmeans['Cluster'] == cluster]
    
    # Calcular matriz de covarianza y media del cluster
    if len(cluster_data) > 1:  # Asegurar que hay suficientes puntos
        covariance = np.cov(cluster_data[[x_feature, y_feature]].T)
        mean_position = cluster_data[[x_feature, y_feature]].mean().values
        draw_ellipse(mean_position, covariance, ax, color=colores_personalizados[i])


plt.xticks(fontsize=25)
plt.xlabel(r"Esfuerzo (1$^{\mathrm{a}}$ mitad)", fontsize=28)
plt.ylabel(r"Beneficio (1$^{\mathrm{a}}$ mitad)", fontsize=28)
plt.yticks(fontsize=25)
plt.xlim(0,200)
plt.ylim(0,120)
#plt.title('Primera mitad con elipses de dispersión', fontsize=18)
plt.savefig("cluster1mitad.png", dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()


# Visto esto se podría clasificar un poco como se hizo con moderado, timido y agresivo y el grupo morado está un poco por determinar, podrían ser flipping?

# Ahora vamos a ver la segunda mitad en ambos casos también.

# In[25]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# A visualizar
x_feature = 'total_harvesting_second_half'
y_feature = 'total_payoff_second_half'

# Crear la figura
plt.figure(figsize=(14, 10))
ax = plt.gca()

# Dibujar los puntos de dispersión con colores personalizados según el clúster
for i in df_kmeans['Cluster'].unique():  # Iterar sobre cada clúster
    plt.scatter(df_kmeans[df_kmeans['Cluster'] == i][x_feature], 
                df_kmeans[df_kmeans['Cluster'] == i][y_feature], 
                color=colores_personalizados[i],  # Asignar color basado en el índice del clúster
                label=f'Cluster {i}', 
                s=100, alpha=1)

# Obtener los clusters únicos y sus colores correspondientes
clusters = np.sort(df_kmeans['Cluster'].unique())

# Dibujar las elipses con los colores personalizados
for i, cluster in enumerate(clusters):
    cluster_data = df_kmeans[df_kmeans['Cluster'] == cluster]
    
    # Calcular matriz de covarianza y media del cluster
    if len(cluster_data) > 1:  # Asegurar que hay suficientes puntos
        covariance = np.cov(cluster_data[[x_feature, y_feature]].T)
        mean_position = cluster_data[[x_feature, y_feature]].mean().values
        draw_ellipse(mean_position, covariance, ax, color=colores_personalizados[i])

# Configuración del gráfico
plt.xlabel(r"Esfuerzo (2$^{\mathrm{a}}$mitad)", fontsize=28)
plt.ylabel(r"Beneficio (2$^{\mathrm{a}}$mitad)", fontsize=28)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlim(0,200)
plt.ylim(0,120)
#plt.title('Segunda mitad con elipses de dispersión', fontsize=18)
plt.savefig("cluster2mitad.png", dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()



# Vamos a comparar ahora el effort en la segunda ronda con la primera para ver la comparación de esfuerzos entre ambas mitades

# In[26]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# A visualizar
x_feature = 'total_harvesting_first_half'
y_feature = 'total_harvesting_second_half'

# Crear la figura
plt.figure(figsize=(14, 10))
ax = plt.gca()

# Dibujar los puntos de dispersión con colores personalizados según el clúster
for i in df_kmeans['Cluster'].unique():  # Iterar sobre cada clúster
    plt.scatter(df_kmeans[df_kmeans['Cluster'] == i][x_feature], 
                df_kmeans[df_kmeans['Cluster'] == i][y_feature], 
                color=colores_personalizados[i],  # Asignar color basado en el índice del clúster
                label=f'Cluster {i}', 
                s=100, alpha=1)

# Obtener los clusters únicos y sus colores correspondientes
clusters = np.sort(df_kmeans['Cluster'].unique())

# Dibujar las elipses con los colores personalizados
for i, cluster in enumerate(clusters):
    cluster_data = df_kmeans[df_kmeans['Cluster'] == cluster]
    
    # Calcular matriz de covarianza y media del cluster
    if len(cluster_data) > 1:  # Asegurar que hay suficientes puntos
        covariance = np.cov(cluster_data[[x_feature, y_feature]].T)
        mean_position = cluster_data[[x_feature, y_feature]].mean().values
        draw_ellipse(mean_position, covariance, ax, color=colores_personalizados[i])

# Configuración del gráfico
plt.xlabel(r"Esfuerzo (1$^{\mathrm{a}}$ mitad)", fontsize=28)
plt.ylabel(r"Esfuerzo (2$^{\mathrm{a}}$  mitad)", fontsize=28)
plt.xticks(fontsize=25)
plt.xlim(0,200)
plt.ylim(0,200)
plt.yticks(fontsize=25)
#plt.title('Esfuerzo primera/segunda mitad', fontsize=18)

x_values = [25, 175]
plt.plot(x_values, x_values, linestyle='--', color='black', label='Diagonal (pendiente 1)')
plt.xlim(0, 200)
plt.ylim(0, 200)
plt.savefig("clusteresfuerzo.png", dpi=300, bbox_inches='tight')


#plt.grid(True)
plt.show()

# Calcular puntos por encima y por debajo de la diagonal
above_line = (df_kmeans[y_feature] > df_kmeans[x_feature]).sum()
below_line = (df_kmeans[y_feature] < df_kmeans[x_feature]).sum()

# Mostrar los resultados de los puntos por encima y por debajo
print(f"Puntos por encima de la diagonal: {above_line}")
print(f"Puntos por debajo o sobre la diagonal: {below_line}")


# <h3> ¿Dentro de cada cluster, cómo son los jugadores?

# In[27]:


import pandas as pd
# 1. Crear un diccionario para mapear cada ID de grupo a su tipo
group_type_map = {}
group_lists = {
    'Óptimo': optimal_groups,
    'Agotado': depleted_groups,
    'Agotamiento': depleting_groups,
    'Sostenido': sustained_groups
}

for group_type, group_list in group_lists.items():
    for group_id in group_list:
        group_type_map[group_id] = group_type

# 2. Añadir una columna 'group_type' al DataFrame
df_kmeans['group_type'] = df_kmeans['session_grupo_id'].map(group_type_map)

# 3. Agrupar por cluster y tipo de grupo, y contar
# Filtramos primero para quitar filas donde 'group_type' es NaN (grupos no clasificados)
# Luego agrupamos, contamos (.size()), y pivotamos (.unstack())
# fill_value=0 pone 0 donde no haya combinaciones
counts_df = df_kmeans.dropna(subset=['group_type']) \
                     .groupby(['Cluster', 'group_type']) \
                     .size() \
                     .unstack(fill_value=0)

# 4. Asegurar que todas las columnas de tipos de grupo estén presentes
# Esto es útil si algún tipo de grupo no tiene ningún miembro en ningún cluster
desired_columns = ['Óptimo', 'Sostenido', 'Agotamiento', 'Agotado']
counts_df = counts_df.reindex(columns=desired_columns, fill_value=0)
counts_df_transposed = counts_df.T
npartic_group = counts_df_transposed.sum(axis=1) # gente por grupo (suma de fila)
counts_df_norm = counts_df_transposed.div(npartic_group,axis=0) # divido cada fila por esa suma
counts_df_norm


# In[28]:


pip install plotly


# ## Gráficos variados <a class="anchor" id="Radar"></a>

# In[29]:


import plotly.io as pio #para poder ejecutar la imagen con fig
pio.renderers.default = "notebook"


# Participantes/Cluster

# In[30]:


# Calcular el conteo de participantes por clúster
cluster_counts = df_kmeans['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Num_Participantes']

# Crear una lista para almacenar el número de participantes por clúster y el normalizado para después
num_participantes_por_cluster = [0] * (cluster_counts['Cluster'].max() + 1)
num_participantes_por_cluster_normalizado = [0] * (cluster_counts['Cluster'].max() + 1)

# Llenar la lista con el número de participantes de cada clúster
for index, row in cluster_counts.iterrows():
    cluster = row['Cluster']
    num_participantes = row['Num_Participantes']
    num_participantes_por_cluster[cluster] = num_participantes
# el número total de participantes para normalizar 
n_partic = sum(num_participantes_por_cluster)

# participantes/cluster normalizado 
num_participantes_por_cluster_normalizado = num_participantes_por_cluster/n_partic


# <h3> Gráfico tipo barra apilada </h3>

# Fracción de participantes en cada categoria

# In[31]:


import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Datos de ejemplo - AHORA UNA ÚNICA LISTA
etiquetas_subcategorias = ['Moderados', 'Timidos', 'Agresivos', 'Invertidos']
etiqueta_eje_x = 'Zaragoza'
colores_personalizados = ['#8cb369','#25a18e','#e76f51','#540d6e'] # Asegúrate de tener suficientes colores

# La posición de la única barra en el eje x
x = [0]
ancho = 0.1  # Ancho de la única barra

# Variable para rastrear la altura acumulada
altura_acumulada = np.array([0.0] * len(x))

fig, ax = plt.subplots()

for i, valor_subcategoria in enumerate(num_participantes_por_cluster_normalizado):
    ax.bar(x, valor_subcategoria, bottom=altura_acumulada, color=colores_personalizados[i], label=etiquetas_subcategorias[i])
    altura_acumulada += valor_subcategoria

España = [0.42,0.08,0.25,0.25]
China = [0.65,0.1,0.25,0]

# Añadir etiquetas y título
ax.set_ylabel('Fracción',fontsize=16)
ax.set_xticks(x)
ax.tick_params(axis='y', labelsize=16)
ax.set_xticklabels([etiqueta_eje_x],fontsize=16)
ax.legend(fontsize=16)

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# In[32]:


# Combinar todo
datos_ciudades = [
    num_participantes_por_cluster_normalizado,
    España,
    China
]

n_ciudades = len(datos_ciudades)
n_categorias = len(etiquetas_subcategorias)
x = np.arange(n_ciudades)  # [0, 1, 2]
etiquetas_ciudades = ['Estudiantes', 'España', 'China']

# Convertimos los datos a formato columna por categoría
datos_array = np.array(datos_ciudades).T  # Transpuesta: shape = (4, 3)

# --- Ajustes de figura y ancho ---
ancho_barra = 0.6
fig, ax = plt.subplots(figsize=(10, 7))  # Figura con layout automático

altura_acumulada = np.zeros(n_ciudades)

# Graficar cada subcategoría como una capa apilada
for i in range(n_categorias):
    ax.bar(
        x,
        datos_array[i],
        width=ancho_barra,
        bottom=altura_acumulada,
        color=colores_personalizados[i],
        label=etiquetas_subcategorias[i]
    )
    altura_acumulada += datos_array[i]

# Etiquetas y formato
ax.set_ylabel('Fracción', fontsize=21)
ax.set_yticks(np.arange(0, 1.01, 0.25))  # Desde 0 hasta 1, con pasos de 0.25
ax.set_xticks(x)
ax.set_xticklabels(etiquetas_ciudades, fontsize=21,rotation=45)
ax.tick_params(axis='y', labelsize=20)

# Ocultar los bordes superior y derecho
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Asegurarse de que los ejes izquierdo e inferior estén en negro
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Leyenda opcional
# ax.legend(fontsize=20, loc='upper right')

plt.ylim(0, 1.05)  # Límite superior para margen visual
plt.show()


# In[33]:


import matplotlib.pyplot as plt
import numpy as np

# Datos
etiquetas_subcategorias = ['Moderados', 'Timidos', 'Agresivos', 'Invertidos']
etiqueta_eje_x = 'Estudiantes'
colores_personalizados = ['#8cb369', '#25a18e', '#e76f51', '#540d6e']
num_participantes_por_cluster_normalizado = np.array([0.3, 0.2, 0.4, 0.1])

x = [0]
ancho = 0.2  # Más estrecho

altura_acumulada = np.array([0.0])

fig, ax = plt.subplots()

for i, valor_subcategoria in enumerate(num_participantes_por_cluster_normalizado):
    ax.bar(x, valor_subcategoria, width=ancho, bottom=altura_acumulada,
           color=colores_personalizados[i], label=etiquetas_subcategorias[i])
    altura_acumulada += valor_subcategoria

# Ajustes de ejes
ax.set_xlim(-0.6, 0.6)  # 🔧 Limita el espacio horizontal para que se note el ancho
ax.set_ylabel('Fracción', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels([etiqueta_eje_x], fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=16)

#plt.tight_layout()
plt.show()


# El casi 50% de los jugadores es moderado, un 15% tímido, un 25% agresivo y un 5% flipado (aprox). 

# In[34]:


# 1. Etiquetas para las barras (Tipos de Grupo)
etiquetas_barras = counts_df_norm.index.tolist()

# 2. Etiquetas para los segmentos (Clusters)
mapa_cluster_nombre = {
    0: 'Moderados',
    1: 'Tímidos',
    2: 'Agresivos',
    3: 'Invertidos'
}
clusters_numericos = counts_df_norm.columns.tolist()
etiquetas_segmentos_nombres = [mapa_cluster_nombre[c] for c in clusters_numericos]

# 3. Colores
if len(colores_personalizados) < len(clusters_numericos):
    print("Advertencia: No hay suficientes colores definidos para todos los clusters.")

# 4. Posiciones de las barras en el eje X
x_pos = np.arange(len(etiquetas_barras))
ancho_barra = 0.8

# Crear figura y ejes con layout automático
fig_norm, ax_norm = plt.subplots(figsize=(10, 7), constrained_layout=True)
altura_acumulada_norm = np.zeros(len(etiquetas_barras))

# Dibujar las barras apiladas
for i, cluster_num in enumerate(clusters_numericos):
    valores_norm = counts_df_norm.loc[:, cluster_num].values
    ax_norm.bar(
        x_pos, valores_norm, ancho_barra, bottom=altura_acumulada_norm,
        color=colores_personalizados[i % len(colores_personalizados)],
        label=mapa_cluster_nombre[cluster_num]
    )
    altura_acumulada_norm += valores_norm

# --- Ejes y etiquetas ---
#ax_norm.set_ylabel('Fracción', fontsize=20)

# Eje X
ax_norm.set_xticks(x_pos)
ax_norm.set_xticklabels([label.capitalize() for label in etiquetas_barras],
                        fontsize=20, rotation=45, ha="right")
ax_norm.tick_params(axis='x', labelsize=20)

# Eje Y
y_ticks_pos = np.linspace(0, 1, 6)
ax_norm.set_yticks(y_ticks_pos)
ax_norm.set_yticks(np.arange(0, 1.01, 0.25))  # Desde 0 hasta 1, con pasos de 0.25
ax_norm.tick_params(axis='y', labelsize=20)

# Quitar bordes superior y derecho
ax_norm.spines['top'].set_visible(False)
ax_norm.spines['right'].set_visible(False)

# Asegurar bordes izquierdo e inferior en negro
ax_norm.spines['left'].set_color('black')
ax_norm.spines['bottom'].set_color('black')

# --- Leyenda fuera de la imagen ---
# loc='center left': Sets the anchor point of the legend itself to its center left.
# bbox_to_anchor=(1.02, 0.5): Places that anchor point at 1.02 (slightly to the right of the axes right edge)
#                              and 0.5 (vertically centered).
#                               These are *axes coordinates* (0,0 bottom-left of axes; 1,1 top-right of axes).
# borderaxespad=0: Removes padding between the axes and the legend box.
# You might need to adjust figsize to make space for the legend.
ax_norm.legend( loc='center left',
               bbox_to_anchor=(1.02, 0.5), fontsize=20)

# Límite superior Y
plt.ylim(0, 1.05)

# Mostrar
plt.show()


# Dentro de los que alcanzan el óptimo, el 50% pertenece al grupo de los moderados y el otro 50% a los tímidos, cosa que es de esperar debido a la forma de juego. Los que se cargan el bosque antes de la última ronda, contienen de todos los grupos excepto de los tímidos. Los flipados están presentes en todos aquellos, al cambiar el modo de jeugo en la segunda mitad, su presencia en los 3 desenlaces concuerda, siendo mayoritaria en los agotados, al aumentar al final del juego el esfuerzo. En comparación con lo que sucedía en España, en este caso si que tenemos grupos en agotamiento además de que los depleted no tienen tantos flipados, al ser su cambio inverso al de los de Zaragoza. Los moderados están más presentes en los agotados también, aunque esto también puede ser por la mayor existencia de jugadores de este grupo. En este caso, ningún jugador agresivo alcanza el óptimo ...

# <h4> Función para el gráfico tipo radar

# Dinámica del recurso y del profit acomulado 

# In[35]:


import matplotlib.pyplot as plt
import numpy as np

# Definición de constantes
NP = 6.0
M = 400.0
TAU = 1.0 / 14.0
G = 0.0504
C = 2.0
P_PARAM = 1.0
TS = 0.395

def dRdt(R):
    """Calcula la ecuación diferencial dR/dt."""
    if R == 0:
        return G * (M - R)
    return G * (M - R) - (NP * TS) / (TAU * M / R)

def computeP(R):
    """Calcula P(t) basándose en R(t)."""
    return TS * ((P_PARAM * R) / (TAU * M) - C)

def main():
    R = M
    t = 0.0
    P = 0.0
    dt = 0.01
    T_max = 50.0

    t_values = []
    R_values = []
    P_values = []

    while t <= T_max:
        t_values.append(t)
        R_values.append(R)
        P_values.append(P)
        
        P = P + computeP(R) * dt
        R = R + dt * dRdt(R)
        t += dt

    # Crear figura con 2 subplots horizontales
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # Gráfico R(t)
    color_r = '#8cb369'
    ax1.plot(t_values, R_values, color=color_r, linewidth=3, label='R(t)')
    ax1.set_xlabel('Ronda', fontsize=25)
    ax1.set_ylabel('Recurso', fontsize=25)
    ax1.tick_params(axis='both', labelsize=23)
    ax1.grid(False)
    ax1.set_yticks(np.arange(0, 401, 100))
    # Líneas horizontales adicionales
    ax1.axhline(y=151, color='blue', linestyle='--', alpha=0.3, linewidth=2)
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.3, linewidth=2)
    #ax1.legend(fontsize=20)

    # Gráfico P(t)
    color_p = '#25a18e'
    ax2.plot(t_values, P_values, color=color_p, linewidth=3, label='P(t)')
    ax2.set_xlabel('Ronda', fontsize=25)
    ax2.set_ylabel('Beneficio acumulado', fontsize=25)
    ax2.tick_params(axis='both', labelsize=23)
    ax2.grid(False)
    ax2.set_yticks(np.arange(0, 76, 25))
    # Líneas verticales adicionales (como en tu código original)
    ax2.axvline(x=25, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    #ax2.legend(fontsize=20)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


# In[36]:


import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros del modelo ---
g = 0.0504
M = 400.0
tau = 1.0 / 14.0
p = 1.0
c = 2.0
N = 6.0
Rc = 100.0

# --- Parámetros de simulación ---
T_END = 50.0
DT = 0.1
INITIAL_R = 400.0


# --- Funciones del modelo ---

# R en equilibrio para un valor dado de T
def calculate_R(T):
    numerator = g * M * tau * M
    denominator = g * tau * M + N * T
    return numerator / denominator

# Función objetivo (ganancia instantánea)
def objective_function(T):
    R = calculate_R(T)
    return (p * R * T) / (tau * M) - c * T

# Derivada de la función objetivo
def gradient_of_objective(T):
    B = g * tau * M
    A = p * g * M
    return (A * B) / ((B + N * T) ** 2) - c

# Máximo T permitido por la restricción R >= Rc
def get_max_T_given_Rc():
    numerator = g * M * tau * M
    B = g * tau * M
    return (numerator / Rc - B) / N

# Descenso de gradiente para encontrar T óptimo
def gradient_descent(initial_T, learning_rate, max_iterations, tolerance):
    T = initial_T
    T_max = get_max_T_given_Rc()

    for _ in range(max_iterations):
        grad = gradient_of_objective(T)
        T += learning_rate * grad

        T = max(0.0, min(T, T_max))  # Limita T entre 0 y T_max

        if abs(grad) < tolerance:
            break

    return T

# Simulación del sistema y acumulación de beneficios
def simulate_cumulative_profit(T_star, R0, t_end, dt):
    times = np.arange(0, t_end + dt, dt)
    R_values = []
    profit_values = []
    cumulative_profit = 0.0
    R = R0

    for t in times:
        dR = g * (M - R) - (R * N * T_star) / (tau * M)
        R += dR * dt

        profit = (p * R * T_star) / (tau * M) - c * T_star
        cumulative_profit += profit * dt

        R_values.append(R)
        profit_values.append(cumulative_profit)

    return times, R_values, profit_values

# --- Ejecución principal ---
initial_T = 1.0
learning_rate = 0.01
max_iterations = 1000
tolerance = 1e-6

T_star = gradient_descent(initial_T, learning_rate, max_iterations, tolerance)
R_star = calculate_R(T_star)
P_star = objective_function(T_star)
T_max_constraint = get_max_T_given_Rc()

times, R_values, profit_values = simulate_cumulative_profit(T_star, INITIAL_R, T_END, DT)

# --- Resultados ---
print(f"Restricción: R >= Rc = {Rc}")
print(f"T_max permitido por restricción: {T_max_constraint:.6f}")
print(r"$T^*_{\text{modelo}}$: {T_star:.6f}")
print(f"R* en equilibrio: {R_star:.6f}")
print(f"Ganancia instantánea P(T*, R*): {P_star:.6f}")
print(f"Ganancia acumulada: {profit_values[-1]:.6f}")
# --- Gráfica de la función objetivo P(T) ---
T_values = np.linspace(0.0, T_max_constraint, 300)
P_values = [objective_function(T) for T in T_values]

plt.figure(figsize=(10, 6))
plt.plot(T_values, P_values, color='purple')
plt.axvline(T_star, linestyle='--', color='orange', label = fr"$T^*_{{\mathrm{{modelo}}}}$= {T_star:.4f}")
plt.xlabel("Esfuerzo ($T_i$)",fontsize=21)
plt.ylabel("Beneficio",fontsize=21)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
plt.legend(fontsize=19)
plt.show()



# In[37]:


import plotly.express as px
import plotly.io as pio
from PIL import Image


def hex_to_rgba(hex_color, alpha):
    """Convierte un color HEX a formato RGBA con opacidad (alpha)"""
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f'rgba({r}, {g}, {b}, {alpha})'

def grafica_radar(cluster, color, titulo, num_participantes_por_cluster, opacidad=0.6):
    MSY_effort = 2.765
    MSY_profit_1 = 57.4
    MSY_profit_2 = 33.4

    df_moderate = df_kmeans[df_kmeans['Cluster'] == cluster].copy()
    ejes = [
        'total_harvesting_first_half',
        'total_harvesting_second_half',
        'total_payoff_first_half',
        'total_payoff_second_half'
    ]

    # Normalización
    df_moderate['total_harvesting_first_half'] /= (MSY_effort * 25)
    df_moderate['total_harvesting_second_half'] /= (MSY_effort * 25)
    df_moderate['total_payoff_first_half'] /= MSY_profit_1
    df_moderate['total_payoff_second_half'] /= MSY_profit_2

    valores_r = df_moderate[ejes].sum().values
    valores_r_escala = valores_r / num_participantes_por_cluster

    df = pd.DataFrame(dict(
        r=valores_r_escala,
        theta=["E1", "E2", "P1", "P2"]
    ))

    # Color con opacidad ajustable
    fillcolor = hex_to_rgba(color, opacidad)

    # Gráfico principal con zona coloreada y línea negra suave
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(
        fill='toself',
        fillcolor=fillcolor,
        line_color='rgba(0, 0, 0, 0.4)',
        line_width=3
    )

    # Estética del fondo y ejes
    fig.update_layout(
        title=dict(text=f"<b>{titulo}</b>", font=dict(size=24)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        polar=dict(
            bgcolor='white',
            radialaxis=dict(
                visible=True,
                range=[0, 1.9],
                gridcolor='black',
                griddash='solid',
                linecolor='black',
                dtick=0.4,  # Mostrar ticks cada 0.4
                tickfont=dict(color='black',size=22)
            ),
            angularaxis=dict(
                gridcolor='black',
                linecolor='black',
                tickfont=dict(color='black',size=22)
            )
        )
    )


    fig.show()


# In[38]:


# Ejemplo de cómo llamar a la función para cada clúster
colores_personalizados = ['#8cb369','#25a18e','#e76f51','#540d6e'] # Asegúrate de tener suficientes colores
titulos = ["MODERADOS","TIMIDOS","AGRESIVOS","INVERTIDOS"]
clusters_unicos = sorted(df_kmeans['Cluster'].unique())

for i, cluster in enumerate(clusters_unicos):
    color = colores_personalizados[i % len(colores_personalizados)] # Asigna colores rotando la lista
    titulo = titulos[i]
    num_participantes_cluster = num_participantes_por_cluster[cluster] # Obtén el número de participantes para el cluster actual
    grafica_radar(cluster, color, titulo, num_participantes_cluster) # Pasa el valor específico


# In[39]:


pip install -U kaleido


# In[40]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from PIL import Image

# Asegúrate de tener esto instalado:
# pip install kaleido pillow

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f'rgba({r}, {g}, {b}, {alpha})'

def grafica_radar(cluster, color, titulo, num_participantes_por_cluster, opacidad=0.6):
    MSY_effort = 2.765 #esfuerzo óptimo en cada una de las rondas 
    MSY_profit_1 = 57.4
    MSY_profit_2 = 33.4

    df_moderate = df_kmeans[df_kmeans['Cluster'] == cluster].copy()
    ejes = [
        'total_harvesting_first_half',
        'total_harvesting_second_half',
        'total_payoff_first_half',
        'total_payoff_second_half'
    ]

    df_moderate['total_harvesting_first_half'] /= (MSY_effort * 25) # normalizado al esfuerzo óptimo en cada mitad
    df_moderate['total_harvesting_second_half'] /= (MSY_effort * 25)
    df_moderate['total_payoff_first_half'] /= MSY_profit_1 # beneficio correspondiente 
    df_moderate['total_payoff_second_half'] /= MSY_profit_2

    valores_r = df_moderate[ejes].sum().values
    valores_r_escala = valores_r / num_participantes_por_cluster

    df = pd.DataFrame(dict(
        r=valores_r_escala,
        theta=["E1", "E2", "P1", "P2"]
    ))

    fillcolor = hex_to_rgba(color, opacidad)

    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(
        fill='toself',
        fillcolor=fillcolor,
        line_color='rgba(0, 0, 0, 0.4)',
        line_width=3
    )

    fig.update_layout(
        title=dict(text=f"<b>{titulo}</b>", font=dict(size=26, color='black',family='serif')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        polar=dict(
            bgcolor='white',
            radialaxis=dict(
                visible=True,
                range=[0, 1.9],
                gridcolor='black',
                griddash='solid',
                linecolor='black',
                dtick=0.4,
                tickfont=dict(color='black', size=22)
            ),
            angularaxis=dict(
                gridcolor='black',
                linecolor='black',
                tickfont=dict(color='black', size=24)
            )
        )
    )

    # 💾 Guardar imagen del radar
    nombre_archivo = f"radar_cluster_{cluster}.png"
    pio.write_image(fig, nombre_archivo, width=500, height=500)
    return nombre_archivo

# --- Ejecutar para cada clúster ---
colores_personalizados = ['#8cb369','#25a18e','#e76f51','#540d6e']
titulos = ["MODERADOS","TIMIDOS","AGRESIVOS","INVERTIDOS"]
clusters_unicos = sorted(df_kmeans['Cluster'].unique())

nombres_imagenes = []

for i, cluster in enumerate(clusters_unicos):
    color = colores_personalizados[i % len(colores_personalizados)]
    titulo = titulos[i]
    num_participantes_cluster = num_participantes_por_cluster[cluster]
    nombre_imagen = grafica_radar(cluster, color, titulo, num_participantes_cluster)
    nombres_imagenes.append(nombre_imagen)

# --- Combinar imágenes horizontalmente ---
imagenes = [Image.open(nombre) for nombre in nombres_imagenes]
anchos, altos = zip(*(i.size for i in imagenes))
total_ancho = sum(anchos)
alto_max = max(altos)

imagen_combinada = Image.new('RGB', (total_ancho, alto_max), (255, 255, 255))
x_offset = 0
for im in imagenes:
    imagen_combinada.paste(im, (x_offset, 0))
    x_offset += im.size[0]

imagen_combinada.save("radar_combinado.png")
print("Imagen combinada guardada como radar_combinado.png")


# In[41]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from PIL import Image

# Asegúrate de tener esto instalado:
# pip install kaleido pillow

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f'rgba({r}, {g}, {b}, {alpha})'

def grafica_radar(cluster, color, titulo, num_participantes_por_cluster, opacidad=0.6):
    MSY_effort = 2.765 #esfuerzo óptimo en cada una de las rondas 
    MSY_profit_1 = 57.4
    MSY_profit_2 = 33.4

    df_moderate = df_kmeans[df_kmeans['Cluster'] == cluster].copy()
    ejes = [
        'total_harvesting_first_half',
        'total_harvesting_second_half',
        'total_payoff_first_half',
        'total_payoff_second_half'
    ]

    df_moderate['total_harvesting_first_half'] /= (MSY_effort * 25) # normalizado al esfuerzo óptimo en cada mitad
    df_moderate['total_harvesting_second_half'] /= (MSY_effort * 25)
    df_moderate['total_payoff_first_half'] /= MSY_profit_1 # beneficio correspondiente 
    df_moderate['total_payoff_second_half'] /= MSY_profit_2

    valores_r = df_moderate[ejes].sum().values
    valores_r_escala = valores_r / num_participantes_por_cluster

    df = pd.DataFrame(dict(
        r=valores_r_escala,
        theta=["E1", "E2", "P1", "P2"]
    ))

    fillcolor = hex_to_rgba(color, opacidad)

    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(
        fill='toself',
        fillcolor=fillcolor,
        line_color='rgba(0, 0, 0, 0.4)',
        line_width=3
    )

    fig.update_layout(
        title=dict(text=f"<b>{titulo}</b>", font=dict(size=24, color='black',family='serif')),
        plot_bgcolor='white',
        paper_bgcolor='white',
        polar=dict(
            bgcolor='white',
            radialaxis=dict(
                visible=True,
                range=[0, 1.9],
                gridcolor='black',
                griddash='solid',
                linecolor='black',
                dtick=0.4,
                tickfont=dict(color='black', size=22)
            ),
            angularaxis=dict(
                gridcolor='black',
                linecolor='black',
                tickfont=dict(color='black', size=22)
            )
        )
    )

    # 💾 Guardar imagen del radar
    nombre_archivo = f"radar_cluster_{cluster}.png"
    pio.write_image(fig, nombre_archivo, width=500, height=500)
    return nombre_archivo

# --- Asumiendo que df_kmeans y num_participantes_por_cluster ya están definidos ---
# Ejemplo de cómo podrían estar definidos si no lo están:
# data = {
#     'Cluster': np.random.choice([0, 1, 2, 3], size=100),
#     'total_harvesting_first_half': np.random.rand(100) * 100,
#     'total_harvesting_second_half': np.random.rand(100) * 100,
#     'total_payoff_first_half': np.random.rand(100) * 100,
#     'total_payoff_second_half': np.random.rand(100) * 100,
# }
# df_kmeans = pd.DataFrame(data)
# num_participantes_por_cluster = df_kmeans['Cluster'].value_counts().to_dict()

# --- Ejecutar para cada clúster ---
colores_personalizados = ['#8cb369','#25a18e','#e76f51','#540d6e']
titulos = ["MODERADOS","TIMIDOS","AGRESIVOS","INVERTIDOS"]
clusters_unicos = sorted(df_kmeans['Cluster'].unique())

nombres_imagenes = []

for i, cluster in enumerate(clusters_unicos):
    color = colores_personalizados[i % len(colores_personalizados)]
    titulo = titulos[i]
    num_participantes_cluster = num_participantes_por_cluster[cluster]
    nombre_imagen = grafica_radar(cluster, color, titulo, num_participantes_cluster)
    nombres_imagenes.append(nombre_imagen)

# --- Combinar imágenes en dos filas ---
imagenes = [Image.open(nombre) for nombre in nombres_imagenes]

# Dividir las imágenes en dos filas
fila_superior_imagenes = imagenes[0:2]
fila_inferior_imagenes = imagenes[2:4] # Asumiendo que siempre hay 4 clusters

# Combinar imágenes para la fila superior
anchos_fila_superior, altos_fila_superior = zip(*(i.size for i in fila_superior_imagenes))
total_ancho_fila_superior = sum(anchos_fila_superior)
alto_max_fila_superior = max(altos_fila_superior)

imagen_fila_superior = Image.new('RGB', (total_ancho_fila_superior, alto_max_fila_superior), (255, 255, 255))
x_offset = 0
for im in fila_superior_imagenes:
    imagen_fila_superior.paste(im, (x_offset, 0))
    x_offset += im.size[0]

# Combinar imágenes para la fila inferior
anchos_fila_inferior, altos_fila_inferior = zip(*(i.size for i in fila_inferior_imagenes))
total_ancho_fila_inferior = sum(anchos_fila_inferior)
alto_max_fila_inferior = max(altos_fila_inferior)

imagen_fila_inferior = Image.new('RGB', (total_ancho_fila_inferior, alto_max_fila_inferior), (255, 255, 255))
x_offset = 0
for im in fila_inferior_imagenes:
    imagen_fila_inferior.paste(im, (x_offset, 0))
    x_offset += im.size[0]

# Combinar las dos filas verticalmente
# Asegúrate de que ambas filas tengan el mismo ancho para una alineación perfecta
ancho_final = max(imagen_fila_superior.width, imagen_fila_inferior.width)
alto_final = imagen_fila_superior.height + imagen_fila_inferior.height

radar_combinado = Image.new('RGB', (ancho_final, alto_final), (255, 255, 255))

# Centrar las imágenes si sus anchos son diferentes
x_offset_superior = (ancho_final - imagen_fila_superior.width) // 2
x_offset_inferior = (ancho_final - imagen_fila_inferior.width) // 2

radar_combinado.paste(imagen_fila_superior, (x_offset_superior, 0))
radar_combinado.paste(imagen_fila_inferior, (x_offset_inferior, imagen_fila_superior.height))

radar_combinado.save("radar_combinado.png")
print("Imagen combinada guardada como radar_combinado.png")


# ## Clustering incluyendo tiempos de decisión <a class="anchor" id="Clustering-tiempos-decisión"></a>

# <h3> Extracción de datos del tiempo de respuesta </h3>

# Tenemos en un archivo csv los datos del tiempo de respuesta, vamos a ver lo que hay dentro primero

# In[42]:


import pandas as pd

#cargamos el archivo
df = pd.read_csv('otree_pagecompletion_202501301423.csv')

# Mostrar las primeras filas del DataFrame para comprobar
#print(df.head())  # Muestra las primeras 5 filas
#print(df.to_string())  # Muestra todo el contenido bien formateado
df.head()


# In[43]:


df_selection = df[['page_name','seconds_on_page','participant_id','session_id']]
#opción pro df_selection = df[df['page_name'] == 'Decision'][['page_name', 'seconds_on_page', 'participant_id', 'session_id']]


# Nos interesan solo los tiempos de decision

# In[44]:


df_decisiontime= df_selection[df_selection['page_name'] =='Decision']
df_decisiontime.head()


# Comprobación: 

# In[45]:


#vamos a ver dentro de cada sesión cuántos participantes hay porque tiene pinta que faltan
conteo_por_sesion = df_decisiontime.groupby('session_id')['participant_id'].nunique()

print(conteo_por_sesion)#contamos cuántos diferentes con .nunique()
conteo_por_sesion = df_decisiontime.groupby('session_id')['participant_id'].nunique()

print(conteo_por_sesion)


# Dentro de los tiempos de decisión, tenemos que crear, para cada participante, una característica nueva. Vamos a ver el promedio, el máximo y el mínimo de tiempo de decisión

# In[46]:


df_time_all = df_decisiontime.groupby('participant_id')['seconds_on_page'].agg(
    avg_time='mean',
    min_time='min',
    max_time='max'
).reset_index()


# <h4> Filtrado de los bots

# In[47]:


# Ver cuántos cumplen la condición
print(f"Participantes con tiempo_min_decision >= 27: {(df_time_all['min_time'] >= 27).sum()}")

# Ver cuántos valores NaN hay en la columna
print(f"Valores NaN en tiempo_min_decision: {df_time_all['min_time'].isna().sum()}")

# Ver el total de filas antes y después
print(f"Total antes de filtrar: {len(df_time_all)}")

# Filtrar los bots
df_time = df_time_all[df_time_all['min_time'] < 27].reset_index(drop=True) #para actualizar los índices

print(f"Total después de filtrar: {len(df_time)}")
#df_time


# Ya hemos eliminado los bots del juego.

# <h4> Histograma de tiempos de decisión

# Por curiosidad, vamos a ver cómo se distribuyen con un histograma

# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt


# Crear una figura con 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 fila, 3 columnas

# Lista de columnas a graficar
columnas = ['min_time', 'avg_time', 'max_time']
titulos = ['Tiempo Mínimo', 'Tiempo Medio', 'Tiempo Máximo']
colores = ['royalblue', 'green', 'red']

# --- Límites deseados para los ejes ---
# Modifica estos valores según tus necesidades
limite_x_min = 0
limite_x_max = 32
limite_y_min = 0
limite_y_max = 0.45

# Iterar sobre las columnas para graficar cada histograma
for i, col in enumerate(columnas):
    sns.histplot(df_time[col], bins=20, kde=True, color=colores[i], stat="density", ax=axes[i])
    axes[i].set_xlabel('t (s)', fontsize=23)
    axes[i].set_title(f'{titulos[i]}', fontsize=25)

    # --- INICIO DE LA MODIFICACIÓN ---
    # Usamos una condición para poner el ylabel solo en el primer subplot (i=0)
    if i == 0:
        axes[i].set_ylabel('Densidad', fontsize=23)
    # --- FIN DE LA MODIFICACIÓN ---
    else:
        axes[i].set_ylabel('') # ¡Esta es la línea clave!
    # --- FIN DE LA CORRECCIÓN ---


    # Comando para fijar los límites de los ejes
    axes[i].set_xlim(limite_x_min, limite_x_max)
    axes[i].set_ylim(limite_y_min, limite_y_max)

    # Comando para cambiar tamaño de números de ejes
    axes[i].tick_params(axis='both', labelsize=23)
    axes[i].grid(False)


# Ajustar el layout
plt.tight_layout()
plt.show()


# El promedio pareece distinguir una gran parte en deliberados y otra en moderados. Viendo el tiempo mínimo si que tenemos que el rango de tiempo se abre pero sigue habiendo un 15% que no baja de los 30segundos, luego los que en promedio tomaban la deicisón de forma reflexiva parece ser que lo hacen durante todo el juego. En la distribución de tiempo máximo vemos que hay un pequeño porcentaje que toma la decisión de forma agresiva siempre. 

# <h4> Evolución temporal de cada jugador

# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Asegurar que los datos estén en el orden correcto
df_decisiontime = df_decisiontime.sort_values(by=['session_id', 'participant_id']).copy()

# Crear una columna de ronda basada en la aparición de cada 'participant_id'
df_decisiontime['round'] = df_decisiontime.groupby(['session_id', 'participant_id']).cumcount() + 1

# Graficar por sesión
sessions = df_decisiontime['session_id'].unique()
for session in sessions:
    df_session = df_decisiontime[df_decisiontime['session_id'] == session]
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_session, x='round', y='seconds_on_page', hue='participant_id', marker='o', palette='tab10')
    
    # Personalizar el gráfico
    plt.xlabel("Ronda")
    plt.ylabel("Tiempo de decisión (segundos)")
    plt.title(f"Evolución del tiempo de decisión por jugador - Sesión {session}")
    plt.legend(title="Jugador", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Mostrar el gráfico
    plt.show()


# Aquí quedaron pillados los bots!

# <h2> Clustering incluyendo tiempos de decisión promedio </h2>

# Ahora hay que añadir los tiempos promedios de decisión de los participantes al Data Frame que teníamos inicialmente para hacer el kmeans. 

# In[50]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#nuestras características son las 4 mencionadas+la nueva 
df_kmeans_tiempos = df_kmeans.copy()  #crear una copia para no modificar el original
df_kmeans_tiempos['avg_time'] = df_time['avg_time'].values
df_kmeans_tiempos['min_time'] = df_time['min_time'].values
df_kmeans_tiempos['max_time'] = df_time['max_time'].values
df_kmeans_tiempos.head()


# In[51]:


#vamos a ver dentro de cada sesión cuántos participantes hay porque tiene pinta que faltan
conteo_por_sesion = df_kmeans_tiempos.groupby('session.code')['participant.id_in_session'].nunique()

print(conteo_por_sesion)#contamos cuántos diferentes con .nunique()
conteo_por_sesion = df_kmeans_tiempos.groupby('session.code')['participant.id_in_session'].nunique()

print(conteo_por_sesion)


# Vale, está bien.

# <h4> Número de clusters

# In[52]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # librería para el kmeans
from sklearn.metrics import silhouette_score  # para el método 

# Nuestras características son las 4 mencionadas + el tiempo promedio 
X = df_kmeans_tiempos[['total_harvesting_first_half', 'total_payoff_first_half',
               'total_harvesting_second_half', 'total_payoff_second_half','avg_time']]
# Guardaremos en una lista los valores de la Y
Y = []
# Bucle en k para probar los números de clusters
for k in range(2, 11): 
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)  # Hará el fit con los 4 parámetros 
    score = silhouette_score(X, kmeans.labels_)
    Y.append(score)

# Agregar el punto (1,0) a las listas de datos
X_values = [1] + list(range(2, 11))
Y_values = [0] + Y

# Graficar la puntuación de la silueta
plt.figure(figsize=(8, 6))
plt.plot(X_values, Y_values, marker='o', linestyle='-', color = '#25a18e')
plt.xlabel('Número de clusters (k)',fontsize=21)
plt.title('Método de Silueta',fontsize=22)
plt.ylabel('Ancho de silueta promedio',fontsize=21)
plt.xticks(X_values,fontsize=20)  # Mostrar todos los valores de k en el eje x
#plt.grid(True)
plt.yticks(fontsize=20)


# Mostrar el gráfico
plt.show()

# Mostrar los valores de la silueta para cada k (incluyendo el punto (1,0))
print("Silhouette scores for each k:")
for k, score in zip(X_values, Y_values):
    print(f'k={k}: {score}')


# Parece que entre 2 y 4, como dos es poco, vamos a coger 4. He probado a hacer el clustering tomando media,min y maximo por separado y se obtiene la misma forma.
# $$N_{cluster}=4$$
# A ver qué pasa con el otro método: 
# <h4>Elbow mode

# In[53]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # librería para el kmeans
from sklearn.metrics import silhouette_score  # para el método 

# Guardaremos en una lista los valores de la inercia 
inertia = []

# Bucle en k para probar los números de clusters
for k in range(2, 11): 
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)  # Hará el fit con los 4 parámetros 
    inertia.append(kmeans.inertia_)  # Guardamos la inercia


# In[54]:


from kneed import KneeLocator

# Encontrar el "codo" automáticamente con la función KneeLocator
kl = KneeLocator(range(2, 11), inertia, curve="convex", direction="decreasing")

# Valor óptimo de k
print(f"El número óptimo de clusters es: {kl.elbow}")

# Graficar con el codo resaltado
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertia, marker='o', linestyle='-', color = '#25a18e', label="Inercia")
plt.axvline(kl.elbow, color='r', linestyle='--', label=f"Codo en k={kl.elbow}")
plt.title('Método del Codo',fontsize=22)
plt.xlabel('Número de clusters (k)',fontsize=21)
plt.ylabel('WCSS',fontsize=21)
plt.xticks(range(2, 11),fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
#plt.legend()
plt.show()


# In[55]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#ahora se hace lo mismo que antes pero solo para el valor de número de cluster óptimo
kmeans = KMeans(n_clusters=4, random_state=42)
#añadimos una columna con el cluster al q pertenece cada fila (creo)
df_kmeans_tiempos['Cluster_t'] = kmeans.fit_predict(X) # Hará el fit con los 4 parámetros y al ser _predict devuelve los centroides 
df_kmeans_tiempos

# Obtener los centroides
centroides = kmeans.cluster_centers_
print(centroides)
#df_kmeans_tiempos


# Que no cambia ni uno

# In[56]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# A visualizar
x_feature = 'total_harvesting_first_half'
y_feature = 'total_payoff_first_half'

# Crear la figura
plt.figure(figsize=(10, 6))
ax = plt.gca()

colores_personalizados =  ['#8cb369','#25a18e','#e76f51','#540d6e']  # Agrega más colores si tienes más clústeres

# Dibujar los puntos de dispersión con colores personalizados según el clúster
for i in df_kmeans_tiempos['Cluster'].unique():  # Iterar sobre cada clúster
    plt.scatter(df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == i][x_feature], 
                df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == i][y_feature], 
                color=colores_personalizados[i],  # Asignar color basado en el índice del clúster
                label=f'Cluster {i}', 
                s=70, alpha=0.9)

# Obtener los clusters únicos y sus colores correspondientes
clusters = np.sort(df_kmeans_tiempos['Cluster'].unique())

# Dibujar las elipses con los colores personalizados
for i, cluster in enumerate(clusters):
    cluster_data = df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == cluster]
    
    # Calcular matriz de covarianza y media del cluster
    if len(cluster_data) > 1:  # Asegurar que hay suficientes puntos
        covariance = np.cov(cluster_data[[x_feature, y_feature]].T)
        mean_position = cluster_data[[x_feature, y_feature]].mean().values
        draw_ellipse(mean_position, covariance, ax, color=colores_personalizados[i])

# Configuración del gráfico
plt.xlabel(r"Effort (1$^{\mathrm{st}}$ half)", fontsize=16)
plt.ylabel(r"Profit (1$^{\mathrm{st}}$  half)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Primera mitad con elipses de dispersión (tiempos)', fontsize=18)
plt.xlim(0,200)
plt.ylim(0,120)

# Mostrar el gráfico
plt.show()


# In[57]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# A visualizar
x_feature = 'total_harvesting_second_half'
y_feature = 'total_payoff_second_half'

# Crear la figura
plt.figure(figsize=(10, 6))
ax = plt.gca()

colores_personalizados =  ['#8cb369','#25a18e','#e76f51','#540d6e']   # Agrega más colores si tienes más clústeres

# Dibujar los puntos de dispersión con colores personalizados según el clúster
for i in df_kmeans_tiempos['Cluster'].unique():  # Iterar sobre cada clúster
    plt.scatter(df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == i][x_feature], 
                df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == i][y_feature], 
                color=colores_personalizados[i],  # Asignar color basado en el índice del clúster
                label=f'Cluster {i}', 
                s=70, alpha=0.9)

# Obtener los clusters únicos y sus colores correspondientes
clusters = np.sort(df_kmeans_tiempos['Cluster'].unique())

# Dibujar las elipses con los colores personalizados
for i, cluster in enumerate(clusters):
    cluster_data = df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == cluster]
    
    # Calcular matriz de covarianza y media del cluster
    if len(cluster_data) > 1:  # Asegurar que hay suficientes puntos
        covariance = np.cov(cluster_data[[x_feature, y_feature]].T)
        mean_position = cluster_data[[x_feature, y_feature]].mean().values
        draw_ellipse(mean_position, covariance, ax, color=colores_personalizados[i])

# Configuración del gráfico
plt.xlabel(r"Effort (2$^{\mathrm{nd}}$ half)", fontsize=16)
plt.ylabel(r"Profit (2$^{\mathrm{nd}}$  half)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Primera mitad con elipses de dispersión', fontsize=18)
plt.xlim(0,200)
plt.ylim(0,120)

# Mostrar el gráfico
plt.show()


# In[58]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# A visualizar
x_feature = 'total_harvesting_first_half'
y_feature = 'total_harvesting_second_half'

# Crear la figura
plt.figure(figsize=(10, 6))
ax = plt.gca()

colores_personalizados =  ['#8cb369','#25a18e','#e76f51','#540d6e']   # Agrega más colores si tienes más clústeres

# Dibujar los puntos de dispersión con colores personalizados según el clúster
for i in df_kmeans_tiempos['Cluster'].unique():  # Iterar sobre cada clúster
    plt.scatter(df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == i][x_feature], 
                df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == i][y_feature], 
                color=colores_personalizados[i],  # Asignar color basado en el índice del clúster
                label=f'Cluster {i}', 
                s=70, alpha=0.9)

# Obtener los clusters únicos y sus colores correspondientes
clusters = np.sort(df_kmeans_tiempos['Cluster'].unique())

# Dibujar las elipses con los colores personalizados
for i, cluster in enumerate(clusters):
    cluster_data = df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == cluster]
    
    # Calcular matriz de covarianza y media del cluster
    if len(cluster_data) > 1:  # Asegurar que hay suficientes puntos
        covariance = np.cov(cluster_data[[x_feature, y_feature]].T)
        mean_position = cluster_data[[x_feature, y_feature]].mean().values
        draw_ellipse(mean_position, covariance, ax, color=colores_personalizados[i])

# Configuración del gráfico
plt.xlabel(r"Effort (1$^{\mathrm{st}}$ half)", fontsize=16)
plt.ylabel(r"Effort (2$^{\mathrm{nd}}$  half)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Primera mitad con elipses de dispersión', fontsize=18)

# Añadimos la diagonal (línea de pendiente 1) en línea discontinua negra
x_values = [df_kmeans['total_harvesting_first_half'].min(), df_kmeans['total_harvesting_first_half'].max()]
plt.plot(x_values, x_values, linestyle='--', color='black', label='Diagonal (pendiente 1)')
plt.xlim(0,200)
plt.ylim(0,200)

# Calcular puntos por encima y por debajo de la diagonal
above_line = (df_kmeans[y_feature] > df_kmeans[x_feature]).sum()
below_line = (df_kmeans[y_feature] < df_kmeans[x_feature]).sum()

# Mostrar los resultados de los puntos por encima y por debajo
print(f"Puntos por encima de la diagonal: {above_line}")
print(f"Puntos por debajo o sobre la diagonal: {below_line}")
# Mostrar el gráfico
plt.show()


# Literalmente mismas cordenadas de los centroides

# In[59]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# A visualizar
x_feature = 'avg_time'
y_feature = 'total_harvesting_second_half'

# Crear la figura
plt.figure(figsize=(10, 6))
ax = plt.gca()

colores_personalizados =  ['#8cb369','#25a18e','#e76f51','#540d6e']   # Agrega más colores si tienes más clústeres

# Dibujar los puntos de dispersión con colores personalizados según el clúster
for i in df_kmeans_tiempos['Cluster'].unique():  # Iterar sobre cada clúster
    plt.scatter(df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == i][x_feature], 
                df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == i][y_feature], 
                color=colores_personalizados[i],  # Asignar color basado en el índice del clúster
                #label=f'Cluster {i}', 
                s=70, alpha=0.9)

# Obtener los clusters únicos y sus colores correspondientes
clusters = np.sort(df_kmeans_tiempos['Cluster'].unique())

# Dibujar las elipses con los colores personalizados
for i, cluster in enumerate(clusters):
    cluster_data = df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == cluster]
    
    # Calcular matriz de covarianza y media del cluster
    if len(cluster_data) > 1:  # Asegurar que hay suficientes puntos
        covariance = np.cov(cluster_data[[x_feature, y_feature]].T)
        mean_position = cluster_data[[x_feature, y_feature]].mean().values
        draw_ellipse(mean_position, covariance, ax, color=colores_personalizados[i])

# Configuración del gráfico
plt.xlabel(r"Tiempo de decisión promedio (s)", fontsize=21)
plt.ylabel(r"Esfuerzo (2$^{\mathrm{a}}$ mitad)", fontsize=21)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0,40)
plt.ylim(0,200)
plt.grid(False)
# Mostrar el gráfico
from scipy.stats import linregress

# Calcular regresión lineal para todos los puntos (sin importar el clúster)
x = df_kmeans_tiempos[x_feature]
y = df_kmeans_tiempos[y_feature]
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Dibujar la recta de regresión
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, '--', color='black', linewidth=2, label=f'Recta de regresión (r = {r_value:.2f})')

# Mostrar leyenda con los clusters y la recta
plt.legend(fontsize=18)
plt.savefig("clustertiemposeffort2.png", dpi=300, bbox_inches='tight')
plt.show()


# In[60]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# A visualizar
x_feature = 'avg_time'
y_feature = 'total_harvesting_first_half'

# Crear la figura
plt.figure(figsize=(10, 6))
ax = plt.gca()

colores_personalizados =  ['#8cb369','#25a18e','#e76f51','#540d6e']   # Agrega más colores si tienes más clústeres

# Dibujar los puntos de dispersión con colores personalizados según el clúster
for i in df_kmeans_tiempos['Cluster'].unique():  # Iterar sobre cada clúster
    plt.scatter(df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == i][x_feature], 
                df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == i][y_feature], 
                color=colores_personalizados[i],  # Asignar color basado en el índice del clúster
                #label=f'Cluster {i}', 
                s=70, alpha=0.9)

# Obtener los clusters únicos y sus colores correspondientes
clusters = np.sort(df_kmeans_tiempos['Cluster'].unique())

# Dibujar las elipses con los colores personalizados
for i, cluster in enumerate(clusters):
    cluster_data = df_kmeans_tiempos[df_kmeans_tiempos['Cluster'] == cluster]
    
    # Calcular matriz de covarianza y media del cluster
    if len(cluster_data) > 1:  # Asegurar que hay suficientes puntos
        covariance = np.cov(cluster_data[[x_feature, y_feature]].T)
        mean_position = cluster_data[[x_feature, y_feature]].mean().values
        draw_ellipse(mean_position, covariance, ax, color=colores_personalizados[i])

# Configuración del gráfico
plt.xlabel(r"Tiempo de decisión promedio (s)", fontsize=21)
plt.ylabel(r"Esfuerzo (1$^{\mathrm{a}}$ mitad)", fontsize=21)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0,40)
plt.grid(False)
plt.ylim(0,200)
from scipy.stats import linregress

# Calcular regresión lineal para todos los puntos (sin importar el clúster)
x = df_kmeans_tiempos[x_feature]
y = df_kmeans_tiempos[y_feature]
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Dibujar la recta de regresión
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, '--', color='black', linewidth=2, label=f'Recta de regresión (r = {r_value:.2f})')

# Mostrar leyenda con los clusters y la recta
plt.legend(fontsize=18)
plt.savefig("clustertiemposeffort1.png", dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()


# No afecta practicamente pues salen todos en el mismo valor de x, no distinguen los tiempos. Vemos además aquí como el grupo que salta de un comportamiento a otro aumenta mucho su esfuerzo en la segunda mitad del juego.

# <h2>Tiempos de decisión distinguiendo cada mitad del juego </h2>

# Como no se ve gran cosa con el tiempo promedio, vamos a probar separándolo por mitades del juego, para así poder distinguir mejor los tipos de jugadores.

# In[61]:


df_decisiontime= df_decisiontime.copy() #con copia que sino da problemas
#asignamos un índice de 1 a 50 para cada participante y sesión para así tener un conteo de las rondas
df_decisiontime.loc[:,'round']=df_decisiontime.groupby(['participant_id','session_id']).cumcount()+1 #empezamos en 1
#el loc porque sino tambien daba problemas 


# In[62]:


#dividimos en mitades el juego con apply y una función lambda
df_decisiontime.loc[:,'game_half']=df_decisiontime['round'].apply(lambda x:1 if x<=25 else 2)


# In[63]:


#calculamos ahora sí las cositas
df_halftime = df_decisiontime.groupby(['participant_id', 'session_id', 'game_half'])['seconds_on_page'].agg(
    mean_time=('mean'),
    max_time=('max'),
    min_time=('min')
).reset_index()

#ahora dividimos por mitades 
df_half_t_all=df_halftime.pivot_table(index=['participant_id','session_id'],
                                    columns='game_half',
                                    values=['mean_time','max_time','min_time'],
                                    aggfunc='first').reset_index() #para tomar el primer valor q encuentra
#renombramos
df_half_t_all.columns= ['participant_id', 'session_id', 
                            'mean_time_1', 'mean_time_2', 
                            'max_time_1', 'max_time_2', 
                            'min_time_1', 'min_time_2']
df_half_t_all


# Filtrado de bots

# In[64]:


df_half_t = df_half_t_all [df_half_t_all['min_time_1']<27].reset_index(drop=True)


# <h3> Clustering con tiempos de decisión por mitades </h3>

# Vamos a usar 4 clusters para poder comparar con el otro estudio. 

# In[70]:


#nuestras características son las 4 mencionadas+la nueva 
df_kmeans_tiempos_mitad = df_kmeans.copy()  #crear una copia para no modificar el original
df_kmeans_tiempos_mitad['mean_1'] = df_half_t['mean_time_1'].values
df_kmeans_tiempos_mitad['mean_2'] = df_half_t['mean_time_2'].values
df_kmeans_tiempos_mitad['max_1'] = df_half_t['max_time_1'].values
df_kmeans_tiempos_mitad['max_2'] = df_half_t['max_time_2'].values
df_kmeans_tiempos_mitad['min_1'] = df_half_t['min_time_1'].values
df_kmeans_tiempos_mitad['min_2'] = df_half_t['min_time_2'].values


# In[71]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#nuestras características son las 4 mencionadas más el tiempo de decisión promedio 
X = df_kmeans_tiempos_mitad[['total_harvesting_first_half', 'total_payoff_first_half',
               'total_harvesting_second_half', 'total_payoff_second_half','mean_1','mean_2']]

#ahora se hace lo mismo que antes pero solo para el valor de número de cluster óptimo
kmeans = KMeans(n_clusters=4, random_state=42)
#añadimos una columna con el cluster al q pertenece cada fila (creo)
df_kmeans_tiempos_mitad['Cluster'] = kmeans.fit_predict(X) # Hará el fit con los 4 parámetros y al ser _predict devuelve los centroides


# In[72]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# A visualizar
x_feature = 'mean_2'
y_feature = 'total_harvesting_first_half'

# Crear la figura
plt.figure(figsize=(10, 6))
ax = plt.gca()

colores_personalizados =  ['#8cb369','#25a18e','#e76f51','#540d6e']   # Agrega más colores si tienes más clústeres

# Dibujar los puntos de dispersión con colores personalizados según el clúster
for i in df_kmeans_tiempos['Cluster'].unique():  # Iterar sobre cada clúster
    plt.scatter(df_kmeans_tiempos_mitad[df_kmeans_tiempos_mitad['Cluster'] == i][x_feature], 
                df_kmeans_tiempos_mitad[df_kmeans_tiempos_mitad['Cluster'] == i][y_feature], 
                color=colores_personalizados[i],  # Asignar color basado en el índice del clúster
                #label=f'Cluster {i}', 
                s=70, alpha=0.9)

# Obtener los clusters únicos y sus colores correspondientes
clusters = np.sort(df_kmeans_tiempos_mitad['Cluster'].unique())

# Dibujar las elipses con los colores personalizados
for i, cluster in enumerate(clusters):
    cluster_data = df_kmeans_tiempos_mitad[df_kmeans_tiempos_mitad['Cluster'] == cluster]
    
    # Calcular matriz de covarianza y media del cluster
    if len(cluster_data) > 1:  # Asegurar que hay suficientes puntos
        covariance = np.cov(cluster_data[[x_feature, y_feature]].T)
        mean_position = cluster_data[[x_feature, y_feature]].mean().values
        draw_ellipse(mean_position, covariance, ax, color=colores_personalizados[i])

# Configuración del gráfico
plt.xlabel(r"Tiempo de decisión promedio (2$^{\mathrm{a}}$ mitad)(s)", fontsize=21)
plt.ylabel(r"Esfuerzo (1$^{\mathrm{a}}$ mitad)", fontsize=21)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Primera mitad con elipses de dispersión', fontsize=18)
plt.xlim(0,40)
plt.ylim(0,200)
plt.grid(False)

from scipy.stats import linregress

# Calcular regresión lineal para todos los puntos (sin importar el clúster)
x = df_kmeans_tiempos_mitad[x_feature]
y = df_kmeans_tiempos_mitad[y_feature]
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Dibujar la recta de regresión
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, '--', color='black', linewidth=2, label=f'Recta de regresión (r = {r_value:.2f})')

# Mostrar leyenda con los clusters y la recta
plt.legend(fontsize=18)
plt.savefig("clustertiemposeffort1_2mitad.png", dpi=300, bbox_inches='tight')


# Mostrar el gráfico
plt.show()


# In[73]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# A visualizar
x_feature = 'mean_2'
y_feature = 'total_harvesting_second_half'

# Crear la figura
plt.figure(figsize=(10, 6))
ax = plt.gca()

colores_personalizados =  ['#8cb369','#25a18e','#e76f51','#540d6e']   # Agrega más colores si tienes más clústeres

# Dibujar los puntos de dispersión con colores personalizados según el clúster
for i in df_kmeans_tiempos['Cluster'].unique():  # Iterar sobre cada clúster
    plt.scatter(df_kmeans_tiempos_mitad[df_kmeans_tiempos_mitad['Cluster'] == i][x_feature], 
                df_kmeans_tiempos_mitad[df_kmeans_tiempos_mitad['Cluster'] == i][y_feature], 
                color=colores_personalizados[i],  # Asignar color basado en el índice del clúster
                #label=f'Cluster {i}', 
                s=70, alpha=0.9)

# Obtener los clusters únicos y sus colores correspondientes
clusters = np.sort(df_kmeans_tiempos_mitad['Cluster'].unique())

# Dibujar las elipses con los colores personalizados
for i, cluster in enumerate(clusters):
    cluster_data = df_kmeans_tiempos_mitad[df_kmeans_tiempos_mitad['Cluster'] == cluster]
    
    # Calcular matriz de covarianza y media del cluster
    if len(cluster_data) > 1:  # Asegurar que hay suficientes puntos
        covariance = np.cov(cluster_data[[x_feature, y_feature]].T)
        mean_position = cluster_data[[x_feature, y_feature]].mean().values
        draw_ellipse(mean_position, covariance, ax, color=colores_personalizados[i])

# Configuración del gráfico
plt.xlabel(r"Tiempo de decisión promedio (2$^{\mathrm{a}}$ mitad)(s)", fontsize=21)
plt.ylabel(r"Esfuerzo (2$^{\mathrm{a}}$ mitad)", fontsize=21)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Segunda mitad con elipses de dispersión', fontsize=18)
plt.xlim(0,40)
plt.ylim(0,200)
plt.grid(False)

from scipy.stats import linregress

# Calcular regresión lineal para todos los puntos (sin importar el clúster)
x = df_kmeans_tiempos_mitad[x_feature]
y = df_kmeans_tiempos_mitad[y_feature]
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Dibujar la recta de regresión
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, '--', color='black', linewidth=2, label=f'Recta de regresión (r = {r_value:.2f})')

# Mostrar leyenda con los clusters y la recta
plt.legend(fontsize=18)
plt.savefig("clustertiemposeffort2_2mitad.png", dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()


# A los morados los separa

# Parece que los tiempos de decisión no afectan al clustering.

# ## Regresión <a class="anchor" id="Regresion"></a>

# Un linear mixed model es un modelo estadístico que se centra en incorporar efectos fijados y aleatorios para representar con precisión estructuras de datos no independientes. Un efecto fijo es un parámetro que no varía. Los efectos aleatorios son variables aleatorias, en nuestro caso, una distribución normal. En una regresión lineal (las de toda la vida) los datos son variables aleatorias y los parámetros son efectos fijos. Ahora los datos son aleatorios y los parámetros son variables aleatorias al primer nivel y fijas al nivel más alto.

# La fórmula general del modelo que tenemos es:
# 
# $$
# T_i(t) = \beta_R R(t) + \sum_{s=1}^{S_1} \beta_T^{-s} T_i(t-s) + \sum_{s=1}^{S_2} \beta_{\langle T \rangle}^{-s} \langle T(t-s) \rangle + \beta_i + \epsilon_i(t)
# $$
# 
# donde $T_i(t)$ es el esfuerzo de la ronda $t$ del jugador $i$. $R(t)$ es el estado del bosque virtual, $T_i(t-s)$ es el esfuerzo pasado en el tiempo $t-s$.$<T_i(t-s)>$ el esfuerzo promedio de todos menos del i. $S_1$ y $S_2$ es el número de pasos atrás que miramos, $\beta_i$ es el término constante del modelo (un fixed effect, específico de cada jugador) y $\epsilon_i(t)$ son los residuos normalizados también propios de cada jugador.
# 

# In[ ]:


import statsmodels.formula.api as smf

# FUNCIÓN QUE AJUSTA SEGÚN EL MODELO MIXED LM 

def mixed_model(df):   
    formula = "effort  ~ resources + T_1 + T_2 + T_3 + T_4 + T_5 + avg_Ti"
    model = smf.mixedlm(formula,df, groups=df["player"])
    resultado = model.fit()
    return resultado


# In[ ]:


import seaborn as sb
import numpy as np

def coef_err_regresion(resultado, title):
    """
    Genera los coeficientes y errores estándar de una regresión.
    
    Parámetros:
    - resultado: Modelo de regresión (ej. resultado_c, resultado_s)
    - title: Título del gráfico

    Retorna:
    - coeficientes: Parámetros del modelo
    - errores_estandar: Errores estándar de los coeficientes
    - title: Título proporcionado
    """
    
    coeficientes = resultado.params
    errores_estandar = resultado.bse
    return coeficientes, errores_estandar, title

def print_resultados(df):
    
    """
    Printea los coeficientes y errores estándar de la regresión
    
    """

    print("\n📊 Diferencia de coeficientes entre regresiones:")
    for var in df.index:
        coef = df.loc[var, 'Coeficiente']
        error = df.loc[var, 'Error Estándar']
        print(f"🔹 {var}: Diferencia = {coef:.4f}, Error Estándar = {error:.4f}")
        
        
def plot_regresion(df,title):
    """
    
    Dado un data frame y el titulo del gráfico de la regresión, devuelve un gráfico con los coeficientes y sus respectivos
    errores
    
    """
    y_lim_neg=-0.1
    y_lim_pos=0.5
    ylabel= "Valor"
    resultado = mixed_model(df)
    coeficientes, errores_estandar,title = coef_err_regresion(resultado,title)

    # Crear un DataFrame con los coeficientes y sus errores estándar
    df_coeficientes = pd.DataFrame({
        'Coeficiente': coeficientes,
        'Error Estándar': errores_estandar
    })

    etiquetas = {
        'resources' : r'$\beta_{R}$',
        'T_1': r'$\beta_{T}^{-1}$',
        'T_2': r'$\beta_{T}^{-2}$',
        'T_3': r'$\beta_{T}^{-3}$',
        'T_4': r'$\beta_{T}^{-4}$',
        'T_5': r'$\beta_{T}^{-5}$',
        'avg_Ti': r'$\beta_{<T>}^{-1}$' 
    }

    # Asegurarse de que el índice tenga los nombres correctos de las variables
    df_coeficientes['Variable'] = df_coeficientes.index

    # Eliminar el término intercepto
    df_coeficientes = df_coeficientes[df_coeficientes['Variable'] != 'Intercept']

    # Ordenar las variables según el orden que mencionaste
    orden_variables = ['resources', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'avg_Ti']
    df_coeficientes = df_coeficientes[df_coeficientes['Variable'].isin(orden_variables)]
    df_coeficientes = df_coeficientes.set_index('Variable').loc[orden_variables]
    
    print_resultados(df_coeficientes)
    grafica_regresion(df_coeficientes,title,y_lim_pos,y_lim_neg, etiquetas, ylabel)
        
    return resultado

def plot_diferencia_regresion(df1, df2, title):
    y_lim_neg=-0.3
    y_lim_pos=0.3
    ylabel = r"$\beta_{estudiantes}-\beta_{España}$"
    # Obtener los resultados de las regresiones
    resultado1 = mixed_model(df1)
    resultado2 = mixed_model(df2)
    
    # Obtener coeficientes y errores estándar
    coef1, err1, title= coef_err_regresion(resultado1, title)
    coef2, err2, title = coef_err_regresion(resultado2, title)
    
    # Calcular la diferencia entre coeficientes y la propagación del error
    coef_dif = coef1 - coef2
    err_dif = np.sqrt(err1**2 + err2**2)

    # Crear un DataFrame con los coeficientes y errores estándar de la diferencia
    df_coef_dif = pd.DataFrame({
        'Coeficiente': coef_dif,
        'Error Estándar': err_dif
    })

    etiquetas = {
        'resources': r'$\Delta \beta_{R}$',
        'T_1': r'$\Delta \beta_{T}^{-1}$',
        'T_2': r'$\Delta \beta_{T}^{-2}$',
        'T_3': r'$\Delta \beta_{T}^{-3}$',
        'T_4': r'$\Delta \beta_{T}^{-4}$',
        'T_5': r'$\Delta \beta_{T}^{-5}$',
        'avg_Ti': r'$\Delta \beta_{<T>}^{-1}$'
    }

    # Asegurar que el índice tenga nombres correctos
    df_coef_dif['Variable'] = df_coef_dif.index

    # Eliminar el término intercepto
    df_coef_dif = df_coef_dif[df_coef_dif['Variable'] != 'Intercept']

    # Ordenar las variables según el orden especificado
    orden_variables = ['resources', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'avg_Ti']
    df_coef_dif = df_coef_dif[df_coef_dif['Variable'].isin(orden_variables)]
    df_coef_dif = df_coef_dif.set_index('Variable').loc[orden_variables]

    print_resultados(df_coef_dif)
    grafica_regresion(df_coef_dif,title,y_lim_pos,y_lim_neg,etiquetas, ylabel)
    
def plot_scattering (resultado,title):
    residuos = resultado.resid
    valores_ajustados = resultado.fittedvalues
    plt.title(title)
    plt.xlim(-2.5,2.5)
    plt.scatter(valores_ajustados,residuos,alpha=0.5)
    
def grafica_regresion(df,title,y_lim_pos,y_lim_neg, etiquetas, ylabel):  
     # Configuración de la gráfica
    plt.figure(figsize=(10, 6))

    # Crear gráfico de puntos con barras de error en el eje X
    plt.errorbar(df.index, df['Coeficiente'],
                 yerr=df['Error Estándar'], fmt='o', color='skyblue', 
                 ecolor='black', capsize=5, markersize=8)

    plt.xticks(ticks=range(len(df.index)), labels=[etiquetas[v] for v in df.index],fontsize=23)
    plt.yticks(fontsize=23)
    plt.ylim(y_lim_neg,y_lim_pos)
    # Añadir títulos y etiquetas
    #plt.title(title,fontsize=22)
    plt.ylabel(ylabel,fontsize=25)
    plt.grid(False)
    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()


# <h4> Datos antiguos

# In[ ]:


import pandas as pd
# Cargar archivo TSV
df_regresion = pd.read_csv("experimental_data.tsv", sep="\t")

#seleccionamos el pais
df_regresion_China = df_regresion[df_regresion['country']=='China'].copy()
df_regresion_Spain = df_regresion[df_regresion['country']=='Spain'].copy()
#España 

df_regresion_China.rename(columns={
    'session.code':'session',
    'participant.code':'player',
    'player.harvesting_time':'effort',
    'group.n_resources' : 'resources',
    'group.id_in_subsession':'group',
    'subsession.round_number': 'round' 
}, inplace=True)

df_regresion_Spain.rename(columns={
    'session.code':'session',
    'participant.code':'player',
    'player.harvesting_time':'effort',
    'group.n_resources' : 'resources',
    'group.id_in_subsession':'group',
    'subsession.round_number': 'round' 
}, inplace=True)


#reordenamos
df_regresion_China = df_regresion_China [['session','player','group','round','effort','resources']]
df_regresion_Spain = df_regresion_Spain [['session','player','group','round','effort','resources']]
#df_regresion.head()


# EL grupo lo necesitamos para hacer el agrupamiento

# - El estado del bosque $R(t)$ es la columna "group.resource_patches", que es una matriz donde los positivos suman a la cantidad de árboles, así $R$ es la suma de números positivos en ese vector. Además, el estado del bosque es diferente para cada grupo (cada uno juega con su propio bosque).
# - $T_i(t)$ variable dependiente, es el esfuerzo de la ronda concreta de cada jugador, en este caso, "player.harvested", distinguido por la ronda de la sesión que sea -> array con los esfuerzos de cada participante del grupo en la ronda t.
# - $T_i(t-s)$ es esta misma columna pero en la ronda previa (-s) y habrá que sumarlo a todos los pasos hacia atrás que hagamos $S_1$, esto se hace con la función $.shift(s)$ 
# - $\langle T_i(t-s) \rangle$ es el promedio del resto en la ronda $t-s$ y sumada a los diferentes $s$.
# 

# Ahora ya podemos preparar los datos para que hagamos la regresión. Necesitamos las columnas de los pasos anteriores, hasta S1 y S2 rondas atrás además del promedio del resto en el paso anterior

# Tiempos anteriores

# In[ ]:


df_regresion_China = df_regresion_China.sort_values(by=['round','group','player']).reset_index(drop=True)
df_regresion_Spain = df_regresion_Spain.sort_values(by=['round','group','player']).reset_index(drop=True)


# In[ ]:


S1=5
#columnas de los tiempos anteriores para ese jugador 
for s in range(1,S1+1):
    df_regresion_China[f"T_{s}"] = df_regresion_China.groupby("player")["effort"].shift(s)
    df_regresion_Spain[f"T_{s}"] = df_regresion_Spain.groupby("player")["effort"].shift(s)

df_regresion_China["avg_Ti"]=df_regresion_China.groupby(["round","group"])["effort"].transform(lambda x: (x.sum()-x)/(len(x)-1)).shift(1) 
df_regresion_Spain["avg_Ti"]=df_regresion_Spain.groupby(["round","group"])["effort"].transform(lambda x: (x.sum()-x)/(len(x)-1)).shift(1) 

df_regresion_China.dropna(inplace=True)
df_regresion_Spain.dropna(inplace=True)


# <h4> Estandarización de las variables </h4>

# Buscamos unas variables que tengan una media de 0 y desviación estándar de 1. Esto es necesario pues los órdenes de las mismas son diferentes y pueden afectar a la regresión (lo hacen si no se estandariza). 

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #se inicializa el scaler

#cuidado que no distingue por grupos, se cogen todas las variables implicadas pues la regresión se hace con todas ellas
X_i = ["effort","resources", "T_1", "T_2", "T_3", "T_4", "T_5", "avg_Ti"]  
df_regresion_China[X_i] = scaler.fit_transform(df_regresion_China[X_i]) 
df_regresion_Spain[X_i] = scaler.fit_transform(df_regresion_Spain[X_i]) 


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Establecer el estilo de los gráficos
sns.set(style="whitegrid")

# Seleccionamos las columnas que fueron estandarizadas
columns_to_plot = ["effort","resources", "T_1", "T_2", "T_3", "T_4", "T_5", "avg_Ti"]

# Crear una figura con varios subgráficos
fig, axes = plt.subplots(len(columns_to_plot), 2, figsize=(12, len(columns_to_plot) * 4))

for i, col in enumerate(columns_to_plot):
    # Histograma de la variable estandarizada
    sns.histplot(df_regresion_Spain[col], kde=False, bins=30, ax=axes[i, 0], color='skyblue', edgecolor='black')
    axes[i, 0].set_title(f'Histograma de {col}')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Frecuencia')
    
    # Gráfico de densidad (Kernel Density Estimate) para observar la distribución
    sns.kdeplot(df_regresion_Spain[col], ax=axes[i, 1], color='red', fill=True)
    axes[i, 1].set_title(f'Gráfico de Densidad de {col}')
    axes[i, 1].set_xlabel(col)
    axes[i, 1].set_ylabel('Densidad')

plt.tight_layout()
plt.show()


# Teniendo las variables implicadas en la regresión, hay que ponerlas de forma adecuada a la regresión.

# In[ ]:


# Obtener el resultado de la regresión
res = plot_regresion(df_regresion_Spain, "Regresión España")

# Usarlo en el gráfico de dispersión
plot_scattering(res, "Residuos España")
plt.show()


# In[ ]:


# Obtener el resultado de la regresión
res = plot_regresion(df_regresion_China, "Regresión China")

# Usarlo en el gráfico de dispersión
plot_scattering(res, "Residuos China")
plt.show()


# In[ ]:


# Obtener el resultado de la regresión
res = plot_diferencia_regresion(df_regresion_China, df_regresion_Spain, "España/China")


# Para ver como se comporta el modelo, podemos ver las observaciones frente a las predicciones y hacer un scattering. Si ajusta bien, los puntos deben agruparse en torno a la diagonal.
# 
# - Obtenemos las predicciones 
# - Calculamos los residuos estandarizados 

# <h4> Datos nuevos

# In[ ]:


import pandas as pd 
df1 = pd.read_csv("ganadores/data_1.csv")
df2 = pd.read_csv("ganadores/data_2.csv")
df3 = pd.read_csv("ganadores/data_3.csv")
#unimos para tener los id que nos interesan en uno solo 
df_participants = pd.concat([df1[["ID"]],df2[["ID"]],df3[["ID"]]], ignore_index=True)
#print(df1["ID"],df2["ID"],df3["ID"])


# In[ ]:


datos = 'pgg_dynamic_resource (accessed 2024-04-18).xlsx'
#guardo todos los datos del excel en un DataFrame 
df = pd.read_excel(datos, engine='openpyxl')
df_regresion_all = df[['participant.code', 'player.harvesting_time','group.id_in_subsession', 'subsession.round_number', 'group.n_resources']]
df_regresion = df_regresion_all[df_regresion_all["participant.code"].isin(df_participants["ID"])]
#df_regresion.head()


# In[ ]:


df_regresion.rename(columns={
    'participant.code':'player',
    'player.harvesting_time':'effort',
    'group.n_resources' : 'resources',
    'group.id_in_subsession':'group',
    'subsession.round_number': 'round' 
}, inplace=True)

#reordenamos
df_regresion = df_regresion[['player','group','round','effort','resources']]
#df_regresion.head()


# In[ ]:


df_regresion = df_regresion.sort_values(by=['round','group','player']).reset_index(drop=True)
#df_regresion.head()


# In[ ]:


S1=5
#columnas de los tiempos anteriores para ese jugador 
for s in range(1,S1+1):
    df_regresion[f"T_{s}"] = df_regresion.groupby("player")["effort"].shift(s)

df_regresion["avg_Ti"]=df_regresion.groupby(["round","group"])["effort"].transform(lambda x: (x.sum()-x)/(len(x)-1)).shift(1) 

df_regresion.dropna(inplace=True)

#renombramos las cosas
df_regresion.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #se inicializa el scaler

#cuidado que no distingue por grupos, se cogen todas las variables implicadas pues la regresión se hace con todas ellas
X_i = ["effort","resources", "T_1", "T_2", "T_3", "T_4", "T_5", "avg_Ti"]  
df_regresion[X_i] = scaler.fit_transform(df_regresion[X_i]) 


# In[ ]:


# Obtener el resultado de la regresión
res = plot_regresion(df_regresion, "Regresión Estudiantes")

# Usarlo en el gráfico de dispersión
plot_scattering(res, "Residuos Estudiantes")
plt.show()


# Lo que aumenta considerablemente en comparación con lo que sucedía con los resultados previos es lo que afecta el tiempo promedio del resto, indicano una alta influencia del resto de participantes. Los recursos disminuyen.

# In[ ]:


res = plot_diferencia_regresion(df_regresion,df_regresion_Spain,"Estudiantes/España")


# In[ ]:


res = plot_diferencia_regresion(df_regresion, df_regresion_China,"Estudiantes/China")


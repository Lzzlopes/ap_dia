import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import numpy as np
import warnings

# Ignorar warnings para melhor visualização no terminal
warnings.filterwarnings("ignore")

print("Iniciando o sistema de apoio ao diagnóstico de Apendicite...")
print("-" * 60)

# --- 1. Carregamento e Preparação dos Dados ---
try:
    # Carrega o dataset principal do arquivo XLSX
    excel_file = pd.ExcelFile('app_data.xlsx')
    sheet_names = excel_file.sheet_names

    df_all_cases = None
    if 'All cases' in sheet_names:
        df_all_cases = pd.read_excel(excel_file, sheet_name='All cases')
        print("Aba 'All cases' carregada com sucesso!")
    elif 'Sheet1' in sheet_names:
        df_all_cases = pd.read_excel(excel_file, sheet_name='Sheet1')
        print("Aba 'Sheet1' carregada com sucesso (assumida como 'All cases')!")
    elif len(sheet_names) == 1:
        df_all_cases = pd.read_excel(excel_file, sheet_name=sheet_names[0])
        print(f"Única aba '{sheet_names[0]}' carregada com sucesso (assumida como 'All cases')!")
    else:
        print(f"ERRO: Não foi possível encontrar uma aba 'All cases' ou 'Sheet1' no arquivo. Abas disponíveis: {sheet_names}")
        exit()

    if df_all_cases is None:
        print("ERRO: Não foi possível carregar os dados de 'All cases' do arquivo Excel.")
        exit()

    print("Dados carregados com sucesso do arquivo Excel!")
except FileNotFoundError:
    print("ERRO: Certifique-se de que 'app_data.xlsx' está na mesma pasta do script.")
    exit()
except Exception as e:
    print(f"ERRO ao carregar o arquivo Excel: {e}")
    exit()

# Identificação das features (X) e da variável alvo (y)
target_column_name = 'Severity' # Nome da coluna alvo (ajuste se necessário)

if target_column_name not in df_all_cases.columns:
    print(f"ERRO: A coluna alvo '{target_column_name}' não foi encontrada no dataset.")
    print("Colunas disponíveis no dataset:")
    print(df_all_cases.columns.tolist())
    print("Por favor, revise o nome da coluna alvo no script.")
    exit()

# Removendo colunas que não são features numéricas ou que são identificadores/textos não úteis para o modelo
features = ['Age', 'Alvarado_Score', 'Paedriatic_Appendicitis_Score', 'Appendix_Diameter',
            'Body_Temperature', 'WBC_Count', 'CRP'] # Features numéricas baseadas nas colunas fornecidas

# Certifica-se de que todas as features existem e são numéricas
numerical_features = [f for f in features if f in df_all_cases.columns and pd.api.types.is_numeric_dtype(df_all_cases[f])]

if not numerical_features:
    print("ERRO: Nenhuma feature numérica válida encontrada com os nomes sugeridos. Por favor, ajuste a lista 'features' no script.")
    print("Colunas numéricas detectadas no dataset:")
    print(df_all_cases.select_dtypes(include=np.number).columns.tolist())
    exit()

initial_rows = df_all_cases.shape[0]
df_all_cases.dropna(subset=[target_column_name], inplace=True)
rows_after_dropna = df_all_cases.shape[0]
if initial_rows > rows_after_dropna:
    print(f"\nForam removidas {initial_rows - rows_after_dropna} linhas com valores ausentes na coluna '{target_column_name}'.")

X = df_all_cases[numerical_features]
y = df_all_cases[target_column_name]

if X.isnull().sum().any():
    print("\nDetectados valores ausentes nas features. Preenchendo com a média da coluna...")
    X = X.fillna(X.mean())

if y.dtype == 'object':
    print("\nCodificando a variável alvo (classes de apendicite) para valores numéricos...")
    class_mapping = {cls: i for i, cls in enumerate(y.unique())}
    inverse_class_mapping = {i: cls for cls, i in class_mapping.items()}
    y = y.map(class_mapping)
    print(f"Mapeamento de classes: {class_mapping}")
else:
    inverse_class_mapping = {i: i for i in y.unique()}
    if all(isinstance(val, int) for val in y.unique()):
        print("\nA variável alvo já é numérica. Assumindo mapeamento 0, 1, 2...")
        if len(y.unique()) == 3 and all(c in [0, 1, 2] for c in y.unique()):
            inverse_class_mapping = {0: 'Leve', 1: 'Moderada', 2: 'Grave'}
            print(f"Mapeamento assumido (verifique e ajuste): {inverse_class_mapping}")
        else:
            print("Mapeamento inverso genérico criado para classes numéricas.")

print("\nShape dos dados (features X, alvo y):", X.shape, y.shape)
print("Classes da variável alvo e suas contagens antes do balanceamento:")
print(y.value_counts())
print("-" * 60)

print("Realizando a normalização dos dados (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("Dados normalizados com sucesso!")
print("-" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {}
accuracies = {}

# Modelo 1: Sem Balanceamento
print("Treinando Modelo 1: RandomForest sem Balanceamento...")
model_no_balance = RandomForestClassifier(random_state=42)
scores_no_balance = cross_val_score(model_no_balance, X_scaled_df, y, cv=cv, scoring='accuracy')
accuracies['No_Balance'] = scores_no_balance.mean()
models['No_Balance'] = model_no_balance.fit(X_scaled_df, y)

print(f"  Acurácia Cross-Validation (Sem Balanceamento): {accuracies['No_Balance']:.4f}")
print("-" * 60)

# Modelo 2: Com SMOTE (Over-sampling)
print("Treinando Modelo 2: RandomForest com SMOTE (Over-sampling)...")
smote = SMOTE(random_state=42, k_neighbors=1)
X_smote, y_smote = smote.fit_resample(X_scaled_df, y)
model_smote = RandomForestClassifier(random_state=42)
scores_smote = cross_val_score(model_smote, X_smote, y_smote, cv=cv, scoring='accuracy')
accuracies['SMOTE'] = scores_smote.mean()
models['SMOTE'] = model_smote.fit(X_smote, y_smote)

print(f"  Acurácia Cross-Validation (SMOTE): {accuracies['SMOTE']:.4f}")
print(f"  Contagem de classes após SMOTE:\n{y_smote.value_counts()}")
print("-" * 60)

# Modelo 3: Com RandomUnderSampler (Under-sampling)
print("Treinando Modelo 3: RandomForest com RandomUnderSampler (Under-sampling)...")
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_scaled_df, y)
model_rus = RandomForestClassifier(random_state=42)
scores_rus = cross_val_score(model_rus, X_rus, y_rus, cv=cv, scoring='accuracy')
accuracies['RandomUnderSampler'] = scores_rus.mean()
models['RandomUnderSampler'] = model_rus.fit(X_rus, y_rus)

print(f"  Acurácia Cross-Validation (RandomUnderSampler): {accuracies['RandomUnderSampler']:.4f}")
print(f"  Contagem de classes após RandomUnderSampler:\n{y_rus.value_counts()}")
print("-" * 60)


print("\nResultados das acurácias dos treinamentos:")
for name, acc in accuracies.items():
    print(f"  Modelo {name}: Acurácia = {acc:.4f}")
print("-" * 60)

# --- 4. Sistema de Inferência para um Novo Paciente ---

def get_diagnosis_info(predicted_class_num, inverse_map):
    """
    Retorna o diagnóstico, classificação e tratamento com base na classe prevista.
    """
    predicted_class_str = inverse_map.get(predicted_class_num, "Desconhecido")

    diagnosis = "Apendicite Confirmada"
    classification = predicted_class_str
    treatment = "Consultar um médico para tratamento específico."

    if "uncomplicated" in predicted_class_str or "Uncomplicated" in predicted_class_str: # Supondo que 'uncomplicated' seja leve
        treatment = "Normalmente, pode ser tratada com antibióticos ou cirurgia laparoscópica. Requer avaliação médica urgente."
    elif "complicated" in predicted_class_str or "Complicated" in predicted_class_str: # Supondo que 'complicated' seja grave
        treatment = "Emergência médica. Requer cirurgia imediata para evitar complicações sérias como ruptura. Procure socorro urgente!"
    elif "Leve" in predicted_class_str or "leve" in predicted_class_str:
        treatment = "Normalmente, pode ser tratada com antibióticos ou cirurgia laparoscópica. Requer avaliação médica urgente."
    elif "Moderada" in predicted_class_str or "moderada" in predicted_class_str:
        treatment = "Geralmente requer cirurgia (apendicectomia). Consulte um médico imediatamente."
    elif "Grave" in predicted_class_str or "grave" in predicted_class_str:
        treatment = "Emergência médica. Requer cirurgia imediata para evitar complicações sérias como ruptura. Procure socorro urgente!"
    elif "0" == str(predicted_class_str) and len(inverse_map) == 3 and all(isinstance(v, (int, str)) for v in inverse_map.values()):
        if 0 in inverse_map.keys() and ('uncomplicated' in inverse_map.values() or 'Leve' in inverse_map.values()):
            treatment = "Possível apendicite leve (não complicada). Avaliação médica recomendada."
        else:
            treatment = "Possível apendicite, grau de severidade 0. Avaliação médica recomendada."
    elif "1" == str(predicted_class_str) and len(inverse_map) == 3 and all(isinstance(v, (int, str)) for v in inverse_map.values()):
        if 1 in inverse_map.keys() and 'Moderada' in inverse_map.values(): # Se 1 for moderada
            treatment = "Possível apendicite moderada. Procure ajuda médica."
        else:
            treatment = "Possível apendicite, grau de severidade 1. Procure ajuda médica."
    elif "2" == str(predicted_class_str) and len(inverse_map) == 3 and all(isinstance(v, (int, str)) for v in inverse_map.values()):
        if 2 in inverse_map.keys() and ('complicated' in inverse_map.values() or 'Grave' in inverse_map.values()):
            treatment = "Possível apendicite grave (complicada). Emergência!"
        else:
            treatment = "Possível apendicite, grau de severidade 2. Emergência!"

    return diagnosis, classification, treatment

print("\n--- Sistema de Inferência de Apendicite ---")
print("Para inferir o diagnóstico de um novo paciente, por favor, insira os dados solicitados.")
print(f"Features esperadas para o paciente: {', '.join(numerical_features)}")
print("-" * 60)

def get_patient_data():
    """
    Coleta os dados do novo paciente via entrada do usuário.
    """
    patient_data = {}
    for feature in numerical_features:
        while True:
            try:
                value = float(input(f"  Digite o valor para '{feature}': "))
                patient_data[feature] = value
                break
            except ValueError:
                print("  Entrada inválida. Por favor, digite um número.")
    return pd.DataFrame([patient_data], columns=numerical_features)

# Loop principal para inferência
while True:
    new_patient_df = get_patient_data()

    # Normaliza os dados do novo paciente usando o mesmo scaler dos dados de treinamento
    new_patient_scaled = scaler.transform(new_patient_df)
    new_patient_scaled_df = pd.DataFrame(new_patient_scaled, columns=numerical_features)

    print("\nRealizando inferência com os 3 modelos treinados...")
    predictions = {}
    for name, model in models.items():
        pred_num = model.predict(new_patient_scaled_df)[0]
        predictions[name] = pred_num
        print(f"  Modelo {name} previu a classe numérica: {pred_num}")

    best_model_name = max(accuracies, key=accuracies.get)
    final_predicted_class_num = predictions[best_model_name]

    print("\n--- Resultado da Inferência ---")
    print(f"Baseado no modelo com maior acurácia ('{best_model_name}'), a previsão final é:")

    diagnosis_text, classification_text, treatment_text = get_diagnosis_info(final_predicted_class_num, inverse_class_mapping)

    print(f"  Diagnóstico: {diagnosis_text}")
    print(f"  Classificação da Apendicite: {classification_text}")
    print(f"  Tratamento Sugerido: {treatment_text}")

    print("\n--- Acurácias dos Treinamentos ---")
    for name, acc in accuracies.items():
        print(f"  Modelo {name}: Acurácia = {acc:.4f}")

    print("-" * 60)

    continue_inference = input("Deseja inferir para outro paciente? (s/n): ").lower()
    if continue_inference != 's':
        print("\nObrigado por usar o sistema. Encerrando...")
        break
    print("-" * 60)

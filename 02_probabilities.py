import pandas as pandas

from scipy.stats import chi2_contingency
from data_preparation import load_data

dataframe = load_data()

def analyze_events(A, B, name_A, name_B):
	n = len(A)

	p_A = A.mean()
	p_B = B.mean()
	p_A_and_B = (A & B).mean()
	p_A_or_B = (A | B).mean()
	p_B_given_A = (A & B).sum() / A.sum()
	p_B_given_not_A = (~A & B).sum() / (~A).sum()
	p_A_given_B = (A & B).sum() / B.sum()

	table = pandas.crosstab(
		A.map({True: name_A, False: f"Não {name_A}"}),
		B.map({True: name_B, False: f"Não {name_B}"}),
		margins=True, margins_name="Total"
	)
	table.index.name = None
	table.columns.name = None

	print(f"\n=== A: {name_A} | B: {name_B} (n = {n}) ===")
	print(table)
	print(f"\nP(A) = {A.sum()}/{n} = {p_A:.4f}")
	print(f"P(B) = {B.sum()}/{n} = {p_B:.4f}")
	print(f"P(A ∩ B) = {(A & B).sum()}/{n} = {p_A_and_B:.4f}")
	print(f"P(A ∪ B) = {p_A:.4f} + {p_B:.4f} - {p_A_and_B:.4f} = {p_A_or_B:.4f}")
	print(f"P(B | A) = {(A & B).sum()}/{A.sum()} = {p_B_given_A:.4f}")
	print(f"P(B | Aᶜ) = {(~A & B).sum()}/{(~A).sum()} = {p_B_given_not_A:.4f}")
	print(f"P(A | B) = {(A & B).sum()}/{B.sum()} = {p_A_given_B:.4f}")

	print("\nTeste de independência:")
	print(f"P(A) * P(B) = {p_A * p_B:.4f} vs P(A ∩ B) = {p_A_and_B:.4f}")

	chi2, p_value, dof, expected = chi2_contingency(pandas.crosstab(A, B))
	print(f"Qui-quadrado = {chi2:.4f}, gl = {dof}, p-valor = {p_value:.2e}")
	print("Conclusão:", "rejeita-se a independência (a = 0,05)" if p_value < .05 else "não há evidência para rejeitar a independência (a = 0,05)")

# Execução
era_2020s = dataframe["era"] == "2020-2025"

analyze_events(era_2020s, dataframe["is_short"], "Era 2020+", "Menos de 3 min")
analyze_events(era_2020s, dataframe["artist_count"] > 1, "Era 2020+", "Colaboração")
analyze_events(era_2020s, dataframe["explicit"] == True, "Era 2020+", "Explícita")
analyze_events(era_2020s, dataframe["album_type"] == "single", "Era 2020+", "Single")
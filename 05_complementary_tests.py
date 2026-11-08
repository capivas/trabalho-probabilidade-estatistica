import numpy as numpy
import pandas as pandas

from scipy import stats
from data_preparation import load_data

dataframe = load_data()

recent = dataframe["era"] == "2020-2025"
duration_old = dataframe.loc[~recent, "duration_min"]
duration_new = dataframe.loc[recent, "duration_min"]

print("=== 1. Diferença de duração entre as eras ===")
print(f"2010-2019: n = {len(duration_old)} | média = {duration_old.mean():.4f} | mediana = {duration_old.median():.4f}")
print(f"2020-2025: n = {len(duration_new)} | média = {duration_new.mean():.4f} | mediana = {duration_new.median():.4f}")

u_stat, u_p = stats.mannwhitneyu(duration_old, duration_new, alternative="two-sided")
prob_superiority = u_stat / (len(duration_old) * len(duration_new))
rank_biserial = 2 * prob_superiority - 1
print("\nMann-Whitney U (bilateral):")
print(f"U = {u_stat:.1f} | p-valor = {u_p:.2e}")
print(f"P(música de 2010-2019 > música de 2020-2025) = {prob_superiority:.4f}")
print(f"Correlação bisserial de postos = {rank_biserial:.4f}")

t_stat, t_p = stats.ttest_ind(duration_old, duration_new, equal_var=False)
mean_diff = duration_old.mean() - duration_new.mean()
var_old = duration_old.var(ddof=1) / len(duration_old)
var_new = duration_new.var(ddof=1) / len(duration_new)
standard_error = numpy.sqrt(var_old + var_new)
welch_df = (var_old + var_new) ** 2 / (var_old ** 2 / (len(duration_old) - 1) + var_new ** 2 / (len(duration_new) - 1))
t_critical = stats.t.ppf(0.975, welch_df)
ci_low = mean_diff - t_critical * standard_error
ci_high = mean_diff + t_critical * standard_error
print("\nTeste t de Welch (bilateral):")
print(f"t = {t_stat:.4f} | gl = {welch_df:.1f} | p-valor = {t_p:.2e}")
print(f"Diferença de médias = {mean_diff:.4f} min ({mean_diff * 60:.1f} s)")
print(f"IC 95% = [{ci_low:.4f}; {ci_high:.4f}] min ([{ci_low * 60:.1f}; {ci_high * 60:.1f}] s)")

print("\n=== 2. Testes qui-quadrado com correção de Bonferroni ===")
events = {
	"Duração < 3 min": dataframe["is_short"],
	"Colaboração (2+ artistas)": dataframe["artist_count"] > 1,
	"Conteúdo explícito": dataframe["explicit"] == True,
	"Lançamento como single": dataframe["album_type"] == "single",
}
n_tests = len(events)
alpha = 0.05
print(f"Número de testes = {n_tests} | α corrigido = {alpha / n_tests:.4f}")

rows = []
for name, event in events.items():
	table = pandas.crosstab(recent, event)
	chi2, p_value, dof, expected = stats.chi2_contingency(table)
	chi2_uncorrected = stats.chi2_contingency(table, correction=False)[0]
	phi = numpy.sqrt(chi2_uncorrected / len(dataframe))
	p_bonferroni = min(p_value * n_tests, 1.0)
	rows.append({
		"evento": name,
		"chi2": round(chi2, 2),
		"p_valor": f"{p_value:.2e}",
		"p_bonferroni": f"{p_bonferroni:.2e}",
		"phi": round(phi, 4),
		"conclusao": "Dependentes" if p_bonferroni < alpha else "Independência não rejeitada",
	})
print(pandas.DataFrame(rows).to_string(index=False))

print("\n=== 3. Validação cruzada (10 partes) do classificador Naive Bayes ===")
is_short = dataframe["is_short"].to_numpy()
is_explicit = (dataframe["explicit"] == True).to_numpy()
target = recent.to_numpy()

def naive_bayes_predict(train_short, train_explicit, train_target, test_short, test_explicit):
	prior = train_target.mean()
	likelihoods = {}
	for era_value in (True, False):
		mask = train_target == era_value
		likelihoods[era_value] = (train_short[mask].mean(), train_explicit[mask].mean())

	def likelihood(era_value, short, explicit):
		p_short, p_explicit = likelihoods[era_value]
		return (p_short if short else 1 - p_short) * (p_explicit if explicit else 1 - p_explicit)

	predictions = []
	for short, explicit in zip(test_short, test_explicit):
		numerator = likelihood(True, short, explicit) * prior
		denominator = numerator + likelihood(False, short, explicit) * (1 - prior)
		predictions.append(numerator / denominator > 0.5)
	return numpy.array(predictions)

rng = numpy.random.default_rng(42)
indices = rng.permutation(len(dataframe))
folds = numpy.array_split(indices, 10)

accuracies = []
baseline_accuracies = []
for test_index in folds:
	train_index = numpy.setdiff1d(indices, test_index)
	predictions = naive_bayes_predict(
		is_short[train_index], is_explicit[train_index], target[train_index],
		is_short[test_index], is_explicit[test_index],
	)
	accuracies.append((predictions == target[test_index]).mean())
	majority_class = target[train_index].mean() > 0.5
	baseline_accuracies.append((target[test_index] == majority_class).mean())

print(f"Acurácia média = {numpy.mean(accuracies):.4f} (desvio padrão = {numpy.std(accuracies, ddof=1):.4f})")
print(f"Acurácia da referência = {numpy.mean(baseline_accuracies):.4f} (desvio padrão = {numpy.std(baseline_accuracies, ddof=1):.4f})")
print(f"Acurácia por parte: {[round(float(a), 4) for a in accuracies]}")

print("\n=== 4. Análise de sensibilidade: sem músicas natalinas ===")
HOLIDAY_PATTERN = (r"christmas|xmas|jingle|holly jolly|santa|sleigh|rudolph|mistletoe|navidad|"
                   r"snow|winter wonderland|wonderful time|underneath the tree|rockin.? around")
is_holiday = dataframe["track_name"].str.lower().str.contains(HOLIDAY_PATTERN, regex=True)

print("Músicas identificadas como natalinas (conferir manualmente):")
holiday_songs = dataframe.loc[is_holiday, ["top_year", "track_name", "artist_name", "duration_min"]]
print(holiday_songs.sort_values("top_year").to_string(index=False))
print("\nMúsicas natalinas por era:")
print(pandas.crosstab(dataframe["era"], is_holiday, margins=True))
print("\nMúsicas natalinas com menos de 3 min, por era:")
print(dataframe[is_holiday].groupby("era")["is_short"].sum())

def compare_eras(data, label):
	recent_mask = data["era"] == "2020-2025"
	old = data.loc[~recent_mask, "duration_min"]
	new = data.loc[recent_mask, "duration_min"]

	table = pandas.crosstab(recent_mask, data["is_short"])
	chi2, p_chi2, _, _ = stats.chi2_contingency(table)
	phi = numpy.sqrt(stats.chi2_contingency(table, correction=False)[0] / len(data))

	u, p_u = stats.mannwhitneyu(old, new, alternative="two-sided")
	t, p_t = stats.ttest_ind(old, new, equal_var=False)
	diff = old.mean() - new.mean()
	v_old, v_new = old.var(ddof=1) / len(old), new.var(ddof=1) / len(new)
	df = (v_old + v_new) ** 2 / (v_old ** 2 / (len(old) - 1) + v_new ** 2 / (len(new) - 1))
	margin = stats.t.ppf(0.975, df) * numpy.sqrt(v_old + v_new)

	print(f"\n--- {label} (n = {len(data)}: {len(old)} + {len(new)}) ---")
	print(f"P(< 3 min | 2010-2019) = {data.loc[~recent_mask, 'is_short'].mean():.4f} | "
	      f"P(< 3 min | 2020-2025) = {data.loc[recent_mask, 'is_short'].mean():.4f}")
	print(f"Qui-quadrado = {chi2:.2f} | p-valor = {p_chi2:.2e} | phi = {phi:.4f}")
	print(f"Mann-Whitney: p-valor = {p_u:.2e} | P(2010-2019 > 2020-2025) = {u / (len(old) * len(new)):.4f}")
	print(f"Welch: diferença = {diff * 60:.1f} s | IC 95% = [{(diff - margin) * 60:.1f}; {(diff + margin) * 60:.1f}] s | p-valor = {p_t:.2e}")


compare_eras(dataframe, "Com todas as músicas")
compare_eras(dataframe[~is_holiday], "Sem as músicas natalinas")
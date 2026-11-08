import os
import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as pyplot

from data_preparation import load_data, comma_format, get_output_dir

dataframe = load_data()

figures_dir = get_output_dir("figures")

def bayes(prior, likelihood_h, likelihood_not_h):
	evidence = likelihood_h * prior + likelihood_not_h * (1 - prior)
	return likelihood_h * prior / evidence, evidence

H = dataframe["era"] == "2020-2025"
is_short = dataframe["is_short"]
is_explicit = dataframe["explicit"] == True

prior = H.mean()

like_short_h = is_short[H].mean()
like_short_not_h = is_short[~H].mean()
posterior_short, p_short = bayes(prior, like_short_h, like_short_not_h)

print("=== 1. Atualização com a evidência: duração < 3 min ===")
print(f"Priori P(H) = {prior:.4f}")
print(f"Verossimilhança P(E|H) = {like_short_h:.4f} | P(E|Hᶜ) = {like_short_not_h:.4f}")
print(f"Razão de verossimilhança = {like_short_h / like_short_not_h:.4f}")
print(f"P(E) = {like_short_h:.4f} x {prior:.4f} + {like_short_not_h:.4f} x {1 - prior:.4f} = {p_short:.4f}")
print(f"Posteriori P(H|E) = {like_short_h * prior:.4f} / {p_short:.4f} = {posterior_short:.4f}")

like_explicit_h = is_explicit[H].mean()
like_explicit_not_h = is_explicit[~H].mean()
posterior_both, p_explicit = bayes(posterior_short, like_explicit_h, like_explicit_not_h)

print("\n=== 2. Segunda evidência: conteúdo explícito (priori = posteriori anterior) ===")
print(f"Priori = {posterior_short:.4f}")
print(f"Verossimilhança P(E₂|H) = {like_explicit_h:.4f} | P(E₂|Hᶜ) = {like_explicit_not_h:.4f}")
print(f"Razão de verossimilhança = {like_explicit_h / like_explicit_not_h:.4f}")
print(f"Posteriori P(H|E₁, E₂) = {posterior_both:.4f}")

print("\n=== 3. Sensibilidade à priori (evidência: duração < 3 min) ===")
for alternative_prior in [.10, .25, prior, .5, .75]:
	alternative_posterior, _ = bayes(alternative_prior, like_short_h, like_short_not_h)
	print(f"Priori = {alternative_prior:.4f} -> posteriori = {alternative_posterior:.4f}")

priors = numpy.linspace(.001, .999, 300)
posteriors_short = [bayes(p, like_short_h, like_short_not_h)[0] for p in priors]
posteriors_both = [bayes(bayes(p, like_short_h, like_short_not_h)[0], like_explicit_h, like_explicit_not_h)[0] for p in priors]

figure, ax = pyplot.subplots(figsize=(7, 5))
ax.plot(priors, priors, linestyle="--", color="gray", label="Sem evidência (posteriori = priori)")
ax.plot(priors, posteriors_short, linewidth=2, label="Evidência: duração < 3 min")
ax.plot(priors, posteriors_both, linewidth=2, label="Evidências: duração < 3 min e explícita")
ax.axvline(prior, linestyle=":", color="black", label=f"Priori adotada ({comma_format(prior, 3)})")
ax.set_xlabel("Probabilidade a priori de a música ser da era 2020–2025")
ax.set_ylabel("Probabilidade a posteriori")
ax.xaxis.set_major_formatter(lambda value, _: comma_format(value, 1))
ax.yaxis.set_major_formatter(lambda value, _: comma_format(value, 1))
ax.legend(fontsize=8)
figure.tight_layout()
figure.savefig(os.path.join(figures_dir, "bayes_sensibilidade_priori.png"), dpi=300)

print("\n=== 4. Naive Bayes: posteriori calculada x proporção observada ===")
rows = []
for short_value in [True, False]:
	for explicit_value in [True, False]:
		ls_h = like_short_h if short_value else 1 - like_short_h
		ls_not_h = like_short_not_h if short_value else 1 - like_short_not_h
		le_h = like_explicit_h if explicit_value else 1 - like_explicit_h
		le_not_h = like_explicit_not_h if explicit_value else 1 - like_explicit_not_h
		naive_posterior, _ = bayes(prior, ls_h * le_h, ls_not_h * le_not_h)

		group = (is_short == short_value) & (is_explicit == explicit_value)
		rows.append({
			"duracao_menor_3": short_value,
			"explicita": explicit_value,
			"n": group.sum(),
			"posteriori_naive": round(naive_posterior, 4),
			"proporcao_observada": round(H[group].mean(), 4),
			"classificacao": "2020-2025" if naive_posterior > .5 else "2010-2019"
		})

naive_table = pandas.DataFrame(rows)
print(naive_table.to_string(index=False))

predicted_recent = pandas.Series(False, index=dataframe.index)
for row in rows:
	if row["classificacao"] == "2020-2025":
		group = (is_short == row["duracao_menor_3"]) & (is_explicit == row["explicita"])
		predicted_recent[group] = True

accuracy = (predicted_recent == H).mean()
baseline = max(H.mean(), 1 - H.mean())
print(f"\nAcurácia do classificador Naive Bayes = {accuracy:.4f}")
print(f"Acurácia de referência (sempre prever a mais frequente) = {baseline:.4f}")
import os
import numpy as numpy
import matplotlib.pyplot as pyplot

from scipy import stats
from data_preparation import load_data, comma_format, get_output_dir, SHORT_LIMIT

dataframe = load_data()

figures_dir = get_output_dir("figures")

ERAS = ["2010-2019", "2020-2025"]

N_PER_YEAR = 100

# Distribuição discreta: Binomial
figure, axes = pyplot.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, era in zip(axes, ERAS):
	era_data = dataframe[dataframe["era"] == era]
	p = era_data["is_short"].mean()
	yearly_counts = era_data.groupby("top_year")["is_short"].sum()

	expected_mean = N_PER_YEAR * p
	expected_var = N_PER_YEAR * p * (1 - p)
	low, high = stats.binom.interval(.95, N_PER_YEAR, p)
	outside = ((yearly_counts < low) | (yearly_counts > high)).sum()
	p_at_least_20 = stats.binom.sf(19, N_PER_YEAR, p)

	print(f"\n=== Binomial | Era {era} | X ~ Bin(n = {N_PER_YEAR}, p = {p:.4f}) ===")
	print("Músicas curtas por ano:", yearly_counts.to_dict())
	print(f"Média: esperada = {expected_mean:.2f} | observada = {yearly_counts.mean():.2f}")
	print(f"Variância: esperada = {expected_var:.2f} | observada = {yearly_counts.var(ddof=1):.2f}")
	print(f"Intervalo de 95% da binomial: [{low:.0f}, {high:.0f}] | anos fora do intervalo: {outside} de {len(yearly_counts)}")
	print(f"P(X ≥ 20) = {p_at_least_20:.2e}")

	k = numpy.arange(0, 61)


	p_label = comma_format(p, 4)
	ax.bar(k, stats.binom.pmf(k, N_PER_YEAR, p), alpha=.6, label=(f"Bin(100; {p_label})"))
	ax.yaxis.set_major_formatter(lambda value, _: comma_format(value))

	stack_level = {}
	for count in yearly_counts:
		level = stack_level.get(count, 0)
		stack_level[count] = level + 1
		ax.scatter(count, .003 + level * .007, marker="^", s=50, color="red", zorder=3)
	ax.scatter([], [], marker="^", s=50, color="red", label="Contagens anuais observadas")
	ax.set_title(f"Era {era}")
	ax.set_xlabel("Músicas com menos de 3 min no Top 100")
	ax.legend(fontsize=8)

axes[0].set_ylabel("Probabilidade")
figure.tight_layout()
figure.savefig(os.path.join(figures_dir, "binomial_musicas_curtas.png"), dpi=300)

# Distribuição contínua: Normal
all_durations = dataframe["duration_min"]
bins = numpy.linspace(all_durations.min(), all_durations.max(), 31)

figure, axes = pyplot.subplots(2, 2, figsize=(11, 8))
for column, era in enumerate(ERAS):
	duration = dataframe.loc[dataframe["era"] == era, "duration_min"]
	mean = duration.mean()
	sigma = duration.std(ddof=1)
	skewness = stats.skew(duration)
	shapiro_stat, shapiro_p = stats.shapiro(duration)
	p_normal = stats.norm.cdf(SHORT_LIMIT, mean, sigma)
	p_observed = (duration < SHORT_LIMIT).mean()

	print(f"\n=== Normal | Era {era} (n = {len(duration)}) ===")
	print(f"μ = {mean:.4f} min | σ = {sigma:.4f} min | assimetria = {skewness:.4f}")
	print(f"Mínimo = {duration.min():.2f} min | mediana = {duration.median():.2f} min | máximo = {duration.max():.2f} min")
	print(f"Shapiro-Wilk: W = {shapiro_stat:.4f}, p-valor = {shapiro_p:.2e}")
	print(f"P(X < 3) pela normal = {p_normal:.4f} | proporção observada = {p_observed:.4f}")

	long_songs = (duration > 6).sum()
	print(f"Músicas com mais de 6 min = {long_songs} | assimetria sem elas = {stats.skew(duration[duration <= 6]):.4f}")
	
	hist_ax = axes[0, column]
	hist_ax.hist(duration, bins=bins, density=True, alpha=.6, label="Observado")
	x = numpy.linspace(all_durations.min(), all_durations.max(), 300)
	hist_ax.plot(x, stats.norm.pdf(x, mean, sigma), linewidth=2, label=f"Normal(μ = {comma_format(mean)}; σ = {comma_format(sigma)})")
	hist_ax.axvline(SHORT_LIMIT, linestyle="--", color="gray", label=f"{SHORT_LIMIT} minutos")
	hist_ax.set_title(f"Era {era}")
	hist_ax.set_xlabel("Duração (min)")
	hist_ax.set_ylabel("Densidade")
	hist_ax.xaxis.set_major_formatter(lambda value, _: comma_format(value, 1))
	hist_ax.yaxis.set_major_formatter(lambda value, _: comma_format(value))
	hist_ax.legend(fontsize=8)

	qq_ax = axes[1, column]
	stats.probplot(duration, dist="norm", plot=qq_ax)
	qq_ax.set_title(f"Gráfico QQ - Era {era}")
	qq_ax.set_xlabel("Quantis teóricos da normal")
	qq_ax.set_ylabel("Duração observada (min)")
	qq_ax.xaxis.set_major_formatter(lambda value, _: comma_format(value, 1))
	qq_ax.yaxis.set_major_formatter(lambda value, _: comma_format(value, 1))

figure.tight_layout()
figure.savefig(os.path.join(figures_dir, "normal_duracao.png"), dpi=300)
import plotly.express as px
import numpy as np
import pandas as pd


def main():
    results = []

    for index in range(20):
        data = np.load(f"output/metrics/results_{index}.npy")
        results.append(data)

    shortest_result = min([len(result) for result in results])
    trimmed_results = [result[:shortest_result] for result in results]
    trimmed_results = np.array(trimmed_results)
    mean_result = np.mean(trimmed_results, axis=0)

    df = pd.DataFrame(dict(
        episode=np.arange(mean_result.size),
        rewards=mean_result,
    ))

    fig = px.line(df, x="episode", y="rewards")
    fig.write_html("output/plots/results.html", auto_open=True)


if __name__ == '__main__':
    main()

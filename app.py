mean_absolute_error        view = view[view["transmission"].astype(str) == trans_filter]
    st.write(f"Matching records: {len(view):,}")
    st.dataframe(view.head(500), use_container_width=True)
    st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"), "filtered_cars.csv", "text/csv")
    fig, ax = plt.subplots()
    ax.hist(view["selling_price"], bins=25)
    ax.set_title("Selling price distribution")
    ax.set_xlabel("Selling price (₹)")
    ax.set_ylabel("Number of cars")
    st.pyplot(fig)
    plt.close(fig)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

scores = []

for model_name, trained_model in models.items():
    predictions = trained_model.predict(X_test)

    scores.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R²": r2_score(y_test, predictions)
    })

scores_df = pd.DataFrame(scores)

with tab3:
    st.subheader("Compare model performance")
    st.caption("Lower MAE/RMSE is better; higher R² is generally better.")
    st.dataframe(scores_df.round(3), use_container_width=True)
    fig, ax = plt.subplots()
    ax.bar(scores_df["Model"], scores_df["RMSE"])
    ax.set_ylabel("RMSE (₹)")
    ax.set_title("RMSE comparison (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)
    plt.close(fig)
    st.success(f"Selected model: {best_name} — lowest test RMSE.")

import shap


class Shap:
    def __init__(self, model, df_xtrain, df_ytrain):
        self.model = model
        self.df_xtrain = df_xtrain
        self.df_ytrain = df_ytrain
        self.shap_values = shap.TreeExplainer(self.model.model).shap_values(
            self.df_xtrain[self.model.selected_features], self.df_ytrain
        )

    def detail_plot(self):
        shap.summary_plot(
            self.shap_values,
            self.df_xtrain[self.model.selected_features],
            max_display=20,
        )

    def summary_plot(self):
        shap.summary_plot(
            self.shap_values,
            self.df_xtrain[self.model.selected_features],
            plot_type="bar",
            max_display=20,
        )

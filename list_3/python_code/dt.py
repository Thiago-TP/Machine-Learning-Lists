import optuna
from sklearn.tree import DecisionTreeClassifier


def suggest_dt_hyperparams(trial: optuna.Trial):
    return {
        "max_depth": trial.suggest_int("depth", 2, 20),
        "min_samples_split": trial.suggest_int("split", 2, 20),
    }


def build_decision_tree(context, params):
    return DecisionTreeClassifier(
        max_depth=params["max_depth"], min_samples_split=params["min_samples_split"]
    )


def train_sklearn_model(context, params, model):
    model.fit(context.X_train, context.y_train)
    preds = model.predict(context.X_val)
    val_loss = 1.0 - accuracy_score(context.y_val, preds)
    return val_loss

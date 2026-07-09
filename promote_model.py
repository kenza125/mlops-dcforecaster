"""
Assigne un alias (ex: "production", "staging") à une version du modèle
DCForecaster enregistré dans le MLflow Model Registry.

Les "stages" (Staging/Production/Archived) sont dépréciés depuis MLflow 2.9.
On utilise désormais les alias, qui sont plus flexibles : un alias est un
pointeur nommé et mutable vers une version précise. Réassigner un alias à
une nouvelle version retire automatiquement l'ancien pointage — c'est ce
qui sert de base au rollback : réassigner l'alias à une version antérieure.

Usage :
    python promote_model.py --version 1 --alias production
    python promote_model.py --version 2 --alias staging
    python promote_model.py --list
    python promote_model.py --remove-alias staging
"""

import argparse

from mlflow.tracking import MlflowClient

# Le projet utilise le tracking store local par défaut (dossier ./mlruns
# généré par track.py). Adapte cette URI si tu passes à un serveur MLflow
# distant ou à un backend SQLite (ex: "sqlite:///mlflow.db").
MLFLOW_TRACKING_URI = "file:./mlruns"
REGISTERED_MODEL_NAME = "DCForecaster_v2"


def promote(version: str, alias: str) -> None:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=alias,
        version=version,
    )

    print(f"'{REGISTERED_MODEL_NAME}' v{version} a maintenant l'alias '@{alias}'")


def list_versions() -> None:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        print(f"Aucune version trouvée pour '{REGISTERED_MODEL_NAME}'.")
        return

    registered_model = client.get_registered_model(REGISTERED_MODEL_NAME)
    aliases_by_version = {}
    for alias, ver in (registered_model.aliases or {}).items():
        aliases_by_version.setdefault(ver, []).append(alias)

    print(f"Versions de '{REGISTERED_MODEL_NAME}' :")
    for v in sorted(versions, key=lambda x: int(x.version)):
        aliases = aliases_by_version.get(v.version, [])
        aliases_str = ", ".join(f"@{a}" for a in aliases) if aliases else "(aucun alias)"
        print(f"  v{v.version} — {aliases_str} — run_id: {v.run_id}")


def remove_alias(alias: str) -> None:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    client.delete_registered_model_alias(name=REGISTERED_MODEL_NAME, alias=alias)
    print(f"Alias '@{alias}' supprimé de '{REGISTERED_MODEL_NAME}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gérer les alias du modèle DCForecaster dans le Registry")
    parser.add_argument("--version", type=str, help="Numéro de version à aliaser")
    parser.add_argument("--alias", type=str, default="production", help="Nom de l'alias (ex: production, staging)")
    parser.add_argument("--list", action="store_true", help="Lister toutes les versions et leurs alias")
    parser.add_argument("--remove-alias", type=str, help="Supprimer un alias existant")
    args = parser.parse_args()

    if args.list:
        list_versions()
    elif args.remove_alias:
        remove_alias(args.remove_alias)
    elif args.version:
        promote(args.version, args.alias)
    else:
        parser.print_help()
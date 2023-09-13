from databricks.sdk.runtime import *


# Below are initialization related functions
def get_current_url():
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()


def get_username() -> str:  # Get the user's username
    return (
        dbutils()
        .notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .tags()
        .apply("user")
        .lower()
        .split("@")[0]
        .replace(".", "_")
    )


def get_pat():
    return (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .apiToken()
        .get()
    )


def get_request_headers() -> str:
    return {
        "Authorization": f"""Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}"""
    }


def get_instance() -> str:
    return (
        dbutils()
        .notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .tags()
        .apply("browserHostName")
    )
